"""
PCC — Prototype-Matched Crowd Completion (annotation-free instance discovery).

Turns DETR's Hungarian duplicate-suppression (its worst structural weakness on
dense same-category crowds) into a discovery signal:
  frame-0 committed track -> DINOv2 masked prototype -> cosine-sim map over the
  frame-0 dense DINOv2 grid -> peak-find spatially-disjoint local maxima that are
  (a) above SIM_THR and (b) NOT covered by any existing track (coverage gate) ->
  seed SAM2 (point prompt) at each peak -> new mask track propagated full video ->
  committed under the SAME committed-selection gate, scored with the SAME R/Q.

Arms produced in ONE pass:
  base      = frame-0 committed only (A0)
  pcc       = frame-0 + PCC discovery (A1)
  pccrand   = frame-0 + RANDOM point-grid seeding at matched seed count (ablation)
  tideg     = frame-0 + gated temporal re-detection (TIDE)              [if DO_TIDE=1]
  tide_pcc  = frame-0 + gated TIDE + PCC (full discovery framework)     [A2 headline]

Env: ROOT, CUDA_VISIBLE_DEVICES, MAXF(60), MAXVIDS(0), TAG,
     DO_TIDE(1), COVER_MAX(0.5), DISC_PERSIST_MIN(0.7), DISC_COMMIT_COVER_MAX(0.3),
     SIM_THR(0.60), MIN_DIST(40), MAX_SEEDS(12), SEED(0)
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
from owdetr_detector import OpenWorldDetector
from metrics import compute_f_score
import torch, cv2, numpy as np
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

ROOT = os.environ["ROOT"]
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
W, H = TARGET_SIZE
MAXF = int(os.environ.get("MAXF", "60")); MAXVIDS = int(os.environ.get("MAXVIDS", "0"))
TAG = os.environ.get("TAG", "pcc")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5
COVER_MAX = float(os.environ.get("COVER_MAX", "0.5"))
DISC_PERSIST_MIN = float(os.environ.get("DISC_PERSIST_MIN", "0.7"))
DISC_COMMIT_COVER_MAX = float(os.environ.get("DISC_COMMIT_COVER_MAX", "0.3"))
DO_TIDE = os.environ.get("DO_TIDE", "1") == "1"
SIM_THR = float(os.environ.get("SIM_THR", "0.60"))
MIN_DIST = int(os.environ.get("MIN_DIST", "40"))
MAX_SEEDS = int(os.environ.get("MAX_SEEDS", "12"))
RNG = np.random.default_rng(int(os.environ.get("SEED", "0")))

print(f"Loading DETR + SAM2 Large + DINOv2... ROOT={ROOT} DO_TIDE={DO_TIDE} "
      f"SIM_THR={SIM_THR} MIN_DIST={MIN_DIST} MAX_SEEDS={MAX_SEEDS}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval().to(DEVICE)
for p in dino.parameters(): p.requires_grad_(False)
print("Models loaded.", flush=True)

_NF = 0

def get_boxes(frame_rgb):
    d = det.detect(frame_rgb)
    out = []
    for b in list(d.get("known_boxes", [])) + list(d.get("unknown_boxes", [])):
        out.append(list(b.tolist() if hasattr(b, "tolist") else b))
    return out

def _clip_box(bx):
    x1, y1, x2, y2 = [float(v) for v in bx[:4]]
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(x2, W)); y2 = max(y1+1, min(y2, H))
    return x1, y1, x2, y2

def _mk_pred(k): return {i: [np.zeros((H, W), np.uint8) for _ in range(_NF)] for i in range(k)}

def _run(prompts, kind, anchor=0, bidir=False):
    """prompts: list of box (x1,y1,x2,y2) if kind=='box' else point (x,y)."""
    pred = _mk_pred(len(prompts)); conf = {i: [] for i in range(len(prompts))}
    if not prompts: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=_TMPD, offload_video_to_cpu=True)
        for i, pr in enumerate(prompts):
            if kind == "box":
                x1, y1, x2, y2 = pr
                sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i,
                                           box=np.array([[x1, y1, x2, y2]], np.float32))
            else:
                x, y = pr
                sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i,
                                           points=np.array([[x, y]], np.float32),
                                           labels=np.array([1], np.int32))
        passes = (False, True) if bidir else (False,)
        for reverse in passes:
            for fi, oids, masks in sam2.propagate_in_video(st, start_frame_idx=int(anchor), reverse=reverse):
                if fi < 0 or fi >= _NF: continue
                for oid, ml in zip(oids, masks):
                    logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    pred[int(oid)][fi] = m
                    pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf

# ---- DINOv2 dense frame-0 features + masked prototypes ----
def dino_grid(frame_rgb):
    Hd = (H // 14) * 14; Wd = (W // 14) * 14
    im = cv2.resize(frame_rgb, (Wd, Hd)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.inference_mode():
        f = dino.forward_features(t)["x_norm_patchtokens"][0]  # [Np, C]
    gh, gw = Hd // 14, Wd // 14
    return f.reshape(gh, gw, -1), gh, gw  # grid features

def proto_from_mask0(f, gh, gw, mask0):
    mk = cv2.resize(mask0, (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
    if mk.sum() == 0: return None
    fm = f[torch.from_numpy(mk).to(f.device)]  # [K,C]
    return F.normalize(fm.mean(0), dim=0)

def pcc_seeds(frame0, f0_masks_at0, cover0):
    """cover0: bool [H,W] union of committed frame-0 masks. Returns list of (x,y) peaks."""
    if not f0_masks_at0: return [], 0
    f, gh, gw = dino_grid(frame0)
    fn = F.normalize(f.reshape(-1, f.shape[-1]), dim=1)  # [gh*gw, C]
    protos = [p for p in (proto_from_mask0(f, gh, gw, m) for m in f0_masks_at0) if p is not None]
    if not protos: return [], gh*gw
    P = F.normalize(torch.stack(protos), dim=1)          # [Nproto, C]
    sim = (fn @ P.t()).max(1).values.reshape(gh, gw)     # per-patch max cosine over prototypes
    simmap = cv2.resize(sim.float().cpu().numpy(), (W, H), interpolation=cv2.INTER_CUBIC)
    # coverage gate: exclude pixels already covered by existing tracks
    work = simmap.copy(); work[cover0] = -1.0
    peaks = []
    for _ in range(MAX_SEEDS):
        idx = int(np.argmax(work)); y, x = divmod(idx, W)
        if work[y, x] < SIM_THR: break
        peaks.append((x, y))
        y0, y1 = max(0, y-MIN_DIST), min(H, y+MIN_DIST+1)
        x0, x1 = max(0, x-MIN_DIST), min(W, x+MIN_DIST+1)
        work[y0:y1, x0:x1] = -1.0   # suppress disk (square) around peak
    return peaks, float((simmap > SIM_THR).mean())

def rand_seeds(cover0, k):
    """k random points in the UNCOVERED region (matched-count control for PCC)."""
    ys, xs = np.where(~cover0)
    if len(xs) == 0 or k == 0: return []
    sel = RNG.choice(len(xs), size=min(k, len(xs)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in sel]

JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

MET = ["greedy", "commit_bij", "prec", "rec", "ntrk", "ncommit", "Rstrict", "Qstrict"]
ARMS = ["base", "pcc", "pccrand"] + (["tideg", "tide_pcc"] if DO_TIDE else [])
agg = {a: {k: [] for k in MET} for a in ARMS}
# per-arm hit accounting by origin: {arm: {"new":[Q..], "f0":[Q..], "nhit_new":int}}
hitacc = {a: {"newQ": [], "f0Q": [], "n_new": 0, "n_f0": 0} for a in ARMS}
per_seq = {}

for si, vid in enumerate(vids):
    t0 = time.time()
    fps = sorted(glob.glob(os.path.join(JROOT, vid, "*.jpg")))
    astems = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(AROOT, vid, "*.png"))}
    paired = [(p, astems[os.path.splitext(os.path.basename(p))[0]]) for p in fps
              if os.path.splitext(os.path.basename(p))[0] in astems]
    paired = paired[:MAXF]
    if len(paired) < 2:
        print(f"  [{si+1}] {vid}: <2 frames, skip", flush=True); continue
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), TARGET_SIZE) for fp, _ in paired]
    gtm = [cv2.resize(np.array(Image.open(ap)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for _, ap in paired]
    n = len(frames); _NF = n
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {vid}: no GT, skip", flush=True); continue
    boxes0 = [_clip_box(b) for b in get_boxes(frames[0])]
    if not boxes0:
        print(f"  [{si+1}] {vid}: no det f0, skip", flush=True); continue

    _TMPD = tempfile.mkdtemp(prefix="pcc_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{_TMPD}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    # ---- frame-0 forward propagation ----
    predF, confF = _run(boxes0, "box", anchor=0, bidir=False)

    # ---- signatures over a growing pool; build helpers incrementally ----
    def sig_of(masks, cf):
        ne = [int(m.sum()) for m in masks if m.sum() > 0]
        return dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                    quality=(len(ne)/n)*(float(np.mean(cf)) if cf else 0.0))

    # committed frame-0 tracks (for prototypes, coverage gate, tide gate)
    f0sig = {i: sig_of(predF[i], confF[i]) for i in predF}
    f0cand = [i for i in predF if f0sig[i]["persistence"] >= PERSIST_MIN and 0 < f0sig[i]["mean_area"] < AREA_MAX]
    f0cand.sort(key=lambda i: f0sig[i]["quality"], reverse=True)
    comm_f0_local, comm_masks = [], []
    for i in f0cand:
        di = np.stack([m[::4, ::4] for m in predF[i]]).astype(bool)
        if all(np.logical_and(di, cj).sum()/max(np.logical_or(di, cj).sum(), 1) < DEDUP_IOU for cj in comm_masks):
            comm_f0_local.append(i); comm_masks.append(di)
    # frame-0 committed masks at frame 0 (full res) + coverage union
    f0_masks_at0 = [predF[i][0] for i in comm_f0_local if predF[i][0].sum() > 0]
    cover0 = np.zeros((H, W), bool)
    for m in f0_masks_at0: cover0 |= (m > 0)

    # ---- PCC discovery (frame-0 spatial prong) ----
    peaks, simfrac = pcc_seeds(frames[0], f0_masks_at0, cover0)
    predP, confP = _run(peaks, "point", anchor=0, bidir=False)
    # ---- random matched-count control ----
    rpts = rand_seeds(cover0, len(peaks))
    predR, confR = _run(rpts, "point", anchor=0, bidir=False)

    # ---- TIDE temporal prong (quartile re-detection + bidir) ----
    disc_pred, disc_conf, kept = {}, {}, 0
    if DO_TIDE:
        redet = sorted(set(f for f in (n//4, n//2, (3*n)//4) if 0 < f < n))
        for kf in redet:
            cover = np.zeros((H, W), np.uint8)
            for i in predF: cover |= predF[i][kf]
            for t in disc_pred.values(): cover |= t[kf]
            newb = []
            for b in get_boxes(frames[kf]):
                x1, y1, x2, y2 = [int(round(v)) for v in _clip_box(b)]
                x2 = min(W, max(x1+1, x2)); y2 = min(H, max(y1+1, y2))
                area = (x2-x1)*(y2-y1)
                if area <= 0: continue
                if int(cover[y1:y2, x1:x2].sum())/area < COVER_MAX: newb.append(_clip_box(b))
            if not newb: continue
            bp, bc = _run(newb, "box", anchor=kf, bidir=True)
            for j in bp:
                disc_pred[kept] = bp[j]; disc_conf[kept] = bc[j]; kept += 1
    shutil.rmtree(_TMPD, ignore_errors=True)

    # ---- assemble superset pool with origin tags ----
    pred, conf, origin = {}, {}, {}
    for i in predF: k = f"f0_{i}"; pred[k] = predF[i]; conf[k] = confF[i]; origin[k] = "f0"
    for j in predP: k = f"p_{j}";  pred[k] = predP[j]; conf[k] = confP[j]; origin[k] = "pcc"
    for j in predR: k = f"r_{j}";  pred[k] = predR[j]; conf[k] = confR[j]; origin[k] = "rand"
    for j in disc_pred: k = f"d_{j}"; pred[k] = disc_pred[j]; conf[k] = disc_conf[j]; origin[k] = "disc"
    tids = list(pred.keys())
    f0_ids = [t for t in tids if origin[t] == "f0"]
    p_ids  = [t for t in tids if origin[t] == "pcc"]
    r_ids  = [t for t in tids if origin[t] == "rand"]
    d_ids  = [t for t in tids if origin[t] == "disc"]

    sig = {t: sig_of(pred[t], conf.get(t)) for t in tids}
    dstack = {t: np.stack([m[::4, ::4] for m in pred[t]]).astype(bool) for t in tids}
    def tiou_d(a, b):
        ia = dstack[a]; ib = dstack[b]
        inter = np.logical_and(ia, ib).sum(); union = np.logical_or(ia, ib).sum()
        return inter/union if union > 0 else 0.0

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    Jrow = {}
    for t in tids:
        pb = np.stack(pred[t]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        Jrow[t] = np.array(out)
    _fc = {}
    def bF(t, g):
        k = (t, g)
        if k not in _fc:
            _fc[k] = float(np.mean([compute_f_score(pred[t][i], gt_seqs[g][i]) for i in range(n)]))
        return _fc[k]

    def select_committed(subset):
        cand = [t for t in subset if sig[t]["persistence"] >= PERSIST_MIN and 0 < sig[t]["mean_area"] < AREA_MAX]
        cand.sort(key=lambda t: sig[t]["quality"], reverse=True)
        comm = []
        for t in cand:
            if all(tiou_d(t, k) < DEDUP_IOU for k in comm): comm.append(t)
        return comm

    def greedy_of(subset):
        if not subset: return 0.0
        Js = np.stack([Jrow[t] for t in subset])
        out = []
        for gi, g in enumerate(gt_ids):
            bt = int(np.argmax(Js[:, gi])); out.append((Js[bt, gi] + bF(subset[bt], g)) / 2)
        return float(np.mean(out))

    def metrics(subset_greedy, committed):
        m = dict(greedy=greedy_of(subset_greedy), commit_bij=0.0, prec=0.0, rec=0.0,
                 ntrk=len(subset_greedy), ncommit=len(committed), Rstrict=0.0, Qstrict=0.0, hits=[])
        if committed:
            Jsub = np.stack([Jrow[t] for t in committed])
            ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sh = 0.0
            for r, c in zip(ri, ci):
                q = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2; per_gt[c] = q
                if Jsub[r, c] > 0.5:
                    nhit += 1; sh += q; m["hits"].append((origin[committed[r]], float(q)))
            m["commit_bij"] = float(np.mean(per_gt)); m["prec"] = nhit/len(committed)
            m["rec"] = nhit/len(gt_ids); m["Rstrict"] = nhit/len(gt_ids); m["Qstrict"] = (sh/nhit) if nhit else 0.0
        return m

    # tide gate (uses downsampled committed-f0 union)
    if comm_f0_local and DO_TIDE:
        cf0_keys = [f"f0_{i}" for i in comm_f0_local]
        union_d = np.logical_or.reduce([dstack[t] for t in cf0_keys])
    else:
        union_d = np.zeros_like(next(iter(dstack.values())))
    def gate_disc(persist_thr, cover_thr):
        keep = []
        for t in d_ids:
            if sig[t]["persistence"] < persist_thr: continue
            if not (0 < sig[t]["mean_area"] < AREA_MAX): continue
            dt = dstack[t]; area = int(dt.sum())
            inter = int(np.logical_and(dt, union_d).sum())
            ov = inter/area if area else 1.0
            if ov < cover_thr: keep.append(t)
        return keep

    res = {}
    res["base"]    = metrics(f0_ids, select_committed(f0_ids))
    res["pcc"]     = metrics(f0_ids + p_ids, select_committed(f0_ids + p_ids))
    res["pccrand"] = metrics(f0_ids + r_ids, select_committed(f0_ids + r_ids))
    if DO_TIDE:
        gk = gate_disc(DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX)
        res["tideg"]    = metrics(f0_ids + gk, select_committed(f0_ids + gk))
        res["tide_pcc"] = metrics(f0_ids + gk + p_ids, select_committed(f0_ids + gk + p_ids))
    else:
        gk = []

    for a in ARMS:
        for k in MET: agg[a][k].append(res[a][k])
        for (o, q) in res[a]["hits"]:
            if o == "f0": hitacc[a]["f0Q"].append(q); hitacc[a]["n_f0"] += 1
            else:         hitacc[a]["newQ"].append(q); hitacc[a]["n_new"] += 1

    per_seq[vid] = {a: {k: round(res[a][k], 4) for k in MET} for a in ARMS}
    per_seq[vid].update(n_peaks=len(peaks), n_pcc_commit=res["pcc"]["ncommit"]-res["base"]["ncommit"],
                        n_disc=kept, n_gated=len(gk), ngt=len(gt_ids), nframe=n, sim_frac=round(simfrac, 4))
    b = res["base"]; pc = res["pcc"]
    extra = ""
    if DO_TIDE: extra = f" tpcc R{res['tide_pcc']['Rstrict']:.2f} JF{res['tide_pcc']['commit_bij']:.3f}"
    print(f"  [{si+1:3d}] {vid:20s} R {b['Rstrict']:.2f}->pcc{pc['Rstrict']:.2f} "
          f"JF {b['commit_bij']:.3f}->pcc{pc['commit_bij']:.3f} "
          f"peaks={len(peaks)} ncommit {b['ncommit']}->{pc['ncommit']} gt={len(gt_ids)}{extra} "
          f"t={time.time()-t0:.0f}s", flush=True)

    if (si+1) % 5 == 0 or si == len(vids)-1:
        def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
        summ = {a: {k: Mv(a, k) for k in MET} for a in ARMS}
        for a in ARMS:
            ha = hitacc[a]
            summ[a]["hits_new"] = ha["n_new"]; summ[a]["hits_f0"] = ha["n_f0"]
            summ[a]["Q_new"] = float(np.mean(ha["newQ"])) if ha["newQ"] else 0.0
            summ[a]["Q_f0"] = float(np.mean(ha["f0Q"])) if ha["f0Q"] else 0.0
        out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS, DO_TIDE=DO_TIDE,
                   SIM_THR=SIM_THR, MIN_DIST=MIN_DIST, MAX_SEEDS=MAX_SEEDS,
                   n=len(agg["base"]["greedy"])), summary=summ, per_seq=per_seq)
        od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)

def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
print(f"\n=== PCC on {TAG} ({len(agg['base']['greedy'])} vids, MAXF={MAXF}) ===", flush=True)
for a in ARMS:
    print(f"[{a:9s}] R={Mv(a,'Rstrict')*100:5.2f} commitJF={Mv(a,'commit_bij')*100:5.2f} "
          f"greedy={Mv(a,'greedy')*100:5.2f} prec={Mv(a,'prec')*100:4.1f} rec={Mv(a,'rec')*100:5.2f} "
          f"trk={Mv(a,'ntrk'):5.1f} commit={Mv(a,'ncommit'):4.1f} Qstrict={Mv(a,'Qstrict')*100:5.2f}", flush=True)
for a in ARMS:
    ha = hitacc[a]
    qn = np.mean(ha["newQ"])*100 if ha["newQ"] else 0.0; qf = np.mean(ha["f0Q"])*100 if ha["f0Q"] else 0.0
    print(f"[{a:9s}] committed hits: new-origin={ha['n_new']} (Q={qn:.2f}) | f0={ha['n_f0']} (Q={qf:.2f})", flush=True)
print("Saved.", flush=True)
