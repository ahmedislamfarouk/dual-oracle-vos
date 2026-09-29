"""
TIDE v2 — adds an ANNOTATION-FREE ADAPTIVE COMMIT GATE to kill the DAVIS
(saturated-scene) flood/regression while keeping the MOSE discovery gain.

Gate idea: a discovery track (seeded off frame-0, backward+forward propagated) is
allowed into the committed set only if it is genuinely a NEW object:
  (1) PERSISTENT NON-OVERLAP: across all frames where it exists, the fraction of its
      area that overlaps the union of committed frame-0 tracks is < DISC_COMMIT_COVER_MAX
      (not just at the detection frame — sustained over the whole track), AND
  (2) STRICTER PERSISTENCE: its temporal persistence >= DISC_PERSIST_MIN (> the 0.5
      bar used for frame-0 tracks), since a speculative off-frame track must earn its
      commit.
On DAVIS (objects already found at frame-0) discovery tracks re-cover existing
objects -> high overlap -> rejected -> no flood. On MOSE (objects hidden at frame-0)
they occupy genuinely uncovered regions -> low overlap -> accepted.

Propagation is done ONCE per video; the gate is a cheap post-hoc commit filter, so
this script sweeps a GRID of (DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX) in one run to
produce the DAVIS/MOSE tradeoff curve, and also reports:
  base   = frame-0 only
  tide   = frame-0 + ALL discovery, gate OFF (the +10pp/flood config, ablation)
  tideg  = frame-0 + gated discovery at the env-selected primary gate

Env: ROOT, CUDA_VISIBLE_DEVICES, MAXF(60), MAXVIDS(0), TAG, COVER_MAX(0.5)[detect-gate],
     DISC_PERSIST_MIN(0.7), DISC_COMMIT_COVER_MAX(0.3)[primary gate], SWEEP(1/0)
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
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
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

ROOT = os.environ["ROOT"]
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60")); MAXVIDS = int(os.environ.get("MAXVIDS", "0"))
TAG = os.environ.get("TAG", "tideg")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5
COVER_MAX = float(os.environ.get("COVER_MAX", "0.5"))
DISC_PERSIST_MIN = float(os.environ.get("DISC_PERSIST_MIN", "0.7"))
DISC_COMMIT_COVER_MAX = float(os.environ.get("DISC_COMMIT_COVER_MAX", "0.3"))
SWEEP = os.environ.get("SWEEP", "1") == "1"
W, H = TARGET_SIZE
GRID_P = [0.6, 0.7, 0.8]
GRID_C = [0.2, 0.3, 0.4]

def temporal_iou(a, b):
    inter = union = 0
    for x, y in zip(a, b):
        inter += np.logical_and(x, y).sum(); union += np.logical_or(x, y).sum()
    return inter/union if union > 0 else 0.0

print(f"Loading DETR + SAM2 Large... ROOT={ROOT} COVER_MAX={COVER_MAX} "
      f"primary gate P>={DISC_PERSIST_MIN} cover<{DISC_COMMIT_COVER_MAX} SWEEP={SWEEP}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)


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

def propagate_forward(tmpd, boxes):
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes))}
    conf = {i: [] for i in range(len(boxes))}
    if not boxes: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i, box=np.array([[x1, y1, x2, y2]], np.float32))
        for fi, oids, masks in sam2.propagate_in_video(st):
            if fi >= n: break
            for oid, ml in zip(oids, masks):
                logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                pred[int(oid)][fi] = m
                pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf

def propagate_bidir(tmpd, boxes_at, anchor):
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes_at))}
    conf = {i: [] for i in range(len(boxes_at))}
    if not boxes_at: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes_at):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i, box=np.array([[x1, y1, x2, y2]], np.float32))
        for reverse in (False, True):
            for fi, oids, masks in sam2.propagate_in_video(st, start_frame_idx=int(anchor), reverse=reverse):
                if fi < 0 or fi >= n: continue
                for oid, ml in zip(oids, masks):
                    logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    pred[int(oid)][fi] = m
                    pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf


JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

MET = ["greedy", "commit_bij", "prec", "rec", "ntrk", "ncommit", "Rstrict", "Qstrict"]
arm_names = ["base", "tide", "tideg"] + ([f"g_{p}_{c}" for p in GRID_P for c in GRID_C] if SWEEP else [])
agg = {a: {k: [] for k in MET} for a in arm_names}
disc_Q, f0_Q = [], []; disc_hits = 0; f0_hits = 0
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

    tmpd = tempfile.mkdtemp(prefix="tideg_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    predF, confF = propagate_forward(tmpd, boxes0)
    redet = sorted(set(f for f in (n//4, n//2, (3*n)//4) if 0 < f < n))
    disc_pred, disc_conf, kept = {}, {}, 0
    for kf in redet:
        cover = np.zeros((H, W), np.uint8)
        for t in predF.values(): cover |= t[kf]
        for t in disc_pred.values(): cover |= t[kf]
        newb = []
        for b in get_boxes(frames[kf]):
            x1, y1, x2, y2 = [int(round(v)) for v in _clip_box(b)]
            x2 = min(W, max(x1+1, x2)); y2 = min(H, max(y1+1, y2))
            area = (x2-x1)*(y2-y1)
            if area <= 0: continue
            if int(cover[y1:y2, x1:x2].sum())/area < COVER_MAX: newb.append(_clip_box(b))
        if not newb: continue
        bp, bc = propagate_bidir(tmpd, newb, kf)
        for j in bp:
            disc_pred[kept] = bp[j]; disc_conf[kept] = bc[j]; kept += 1
    shutil.rmtree(tmpd, ignore_errors=True)

    # ---- assemble superset pool with origin tags ----
    pred, conf, origin = {}, {}, {}
    for i in predF: k = f"f0_{i}"; pred[k] = predF[i]; conf[k] = confF[i]; origin[k] = "f0"
    for j in disc_pred: k = f"d_{j}"; pred[k] = disc_pred[j]; conf[k] = disc_conf[j]; origin[k] = "disc"
    tids = list(pred.keys())
    f0_ids = [t for t in tids if origin[t] == "f0"]
    d_ids  = [t for t in tids if origin[t] == "disc"]

    sig = {}
    for t in tids:
        ne = [int(m.sum()) for m in pred[t] if m.sum() > 0]
        sig[t] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                      quality=(len(ne)/n)*(float(np.mean(conf[t])) if conf.get(t) else 0.0))

    # OPT: downsampled bool stacks for fast temporal-IoU dedup + gate coverage
    # (J&F vs GT below stays full-res; only track-vs-track set ops are downsampled)
    dstack = {t: np.stack([m[::4, ::4] for m in pred[t]]).astype(bool) for t in tids}
    def tiou_d(a, b):
        ia = dstack[a]; ib = dstack[b]
        inter = np.logical_and(ia, ib).sum(); union = np.logical_or(ia, ib).sum()
        return inter/union if union > 0 else 0.0

    # ---- scorer (compute J over superset once) ----
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
        Js = np.stack([Jrow[t] for t in subset])  # [T,G]
        out = []
        for gi, g in enumerate(gt_ids):
            bt = int(np.argmax(Js[:, gi])); out.append((Js[bt, gi] + bF(subset[bt], g)) / 2)
        return float(np.mean(out))

    def metrics(subset_greedy, committed):
        m = dict(greedy=greedy_of(subset_greedy), commit_bij=0.0, prec=0.0, rec=0.0,
                 ntrk=len(subset_greedy), ncommit=len(committed), Rstrict=0.0, Qstrict=0.0, hits=[])
        if committed:
            Jsub = np.stack([Jrow[t] for t in committed])  # [P,G]
            ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sh = 0.0
            for r, c in zip(ri, ci):
                q = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2; per_gt[c] = q
                if Jsub[r, c] > 0.5:
                    nhit += 1; sh += q; m["hits"].append((origin[committed[r]], float(q)))
            m["commit_bij"] = float(np.mean(per_gt)); m["prec"] = nhit/len(committed)
            m["rec"] = nhit/len(gt_ids); m["Rstrict"] = nhit/len(gt_ids); m["Qstrict"] = (sh/nhit) if nhit else 0.0
        return m

    # committed frame-0 union (for gate) — downsampled to match dstack
    comm_f0 = select_committed(f0_ids)
    if comm_f0:
        union_d = np.logical_or.reduce([dstack[t] for t in comm_f0])  # [n, h2, w2] bool
    else:
        union_d = np.zeros_like(next(iter(dstack.values())))

    def gate_disc(persist_thr, cover_thr):
        keep = []
        for t in d_ids:
            if sig[t]["persistence"] < persist_thr: continue
            if not (0 < sig[t]["mean_area"] < AREA_MAX): continue
            dt = dstack[t]; area = int(dt.sum())
            inter = int(np.logical_and(dt, union_d).sum())
            ov = inter/area if area else 1.0   # persistent overlap fraction over lifetime
            if ov < cover_thr: keep.append(t)
        return keep

    res = {}
    res["base"]  = metrics(f0_ids, comm_f0)
    res["tide"]  = metrics(tids, select_committed(tids))
    gk = gate_disc(DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX)
    sub_g = f0_ids + gk
    res["tideg"] = metrics(sub_g, select_committed(sub_g))
    if SWEEP:
        for p in GRID_P:
            for c in GRID_C:
                gg = gate_disc(p, c); sg = f0_ids + gg
                res[f"g_{p}_{c}"] = metrics(sg, select_committed(sg))

    for a in arm_names:
        for k in MET: agg[a][k].append(res[a][k])
    for (o, q) in res["tideg"]["hits"]:
        if o == "disc": disc_Q.append(q); disc_hits += 1
        else: f0_Q.append(q); f0_hits += 1

    per_seq[vid] = {a: {k: round(res[a][k], 4) for k in MET} for a in ["base", "tide", "tideg"]}
    per_seq[vid].update(n_disc=kept, n_gated=len(gk), ngt=len(gt_ids), nframe=n)
    b, td, tg = res["base"], res["tide"], res["tideg"]
    print(f"  [{si+1:3d}] {vid:20s} R {b['Rstrict']:.2f}->{tg['Rstrict']:.2f} "
          f"JF {b['commit_bij']:.3f}->g{tg['commit_bij']:.3f}(off{td['commit_bij']:.3f}) "
          f"commit {b['ncommit']}->g{tg['ncommit']}(off{td['ncommit']}) disc={kept} gated={len(gk)} "
          f"gt={len(gt_ids)} t={time.time()-t0:.0f}s", flush=True)

    if (si+1) % 5 == 0 or si == len(vids)-1:
        def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
        summ = {a: {k: Mv(a, k) for k in MET} for a in arm_names}
        summ["off_frame0"] = dict(disc_hits=disc_hits, f0_hits=f0_hits,
                                  disc_Q=float(np.mean(disc_Q)) if disc_Q else 0.0,
                                  f0_Q=float(np.mean(f0_Q)) if f0_Q else 0.0)
        out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS, COVER_MAX=COVER_MAX,
                   DISC_PERSIST_MIN=DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX=DISC_COMMIT_COVER_MAX,
                   n=len(agg["base"]["greedy"])), summary=summ, per_seq=per_seq)
        od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)

def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
print(f"\n=== TIDE-v2 GATED on {TAG} ({len(agg['base']['greedy'])} vids, MAXF={MAXF}) ===", flush=True)
for a in ["base", "tide", "tideg"]:
    print(f"[{a:6s}] R={Mv(a,'Rstrict')*100:5.2f} commitJF={Mv(a,'commit_bij')*100:5.2f} "
          f"greedy={Mv(a,'greedy')*100:5.2f} prec={Mv(a,'prec')*100:4.1f} "
          f"trk={Mv(a,'ntrk'):5.1f} commit={Mv(a,'ncommit'):4.1f} Qstrict={Mv(a,'Qstrict')*100:5.2f}", flush=True)
if SWEEP:
    print("--- gate sweep (persist, cover) -> commitJF / commit-count ---", flush=True)
    for p in GRID_P:
        for c in GRID_C:
            a = f"g_{p}_{c}"
            print(f"  P>={p} cover<{c}: JF={Mv(a,'commit_bij')*100:5.2f} R={Mv(a,'Rstrict')*100:5.2f} "
                  f"commit={Mv(a,'ncommit'):4.1f} prec={Mv(a,'prec')*100:4.1f}", flush=True)
print(f"[off-frame0 tideg] disc hits={disc_hits} (Q={np.mean(disc_Q)*100 if disc_Q else 0:.2f}) | "
      f"f0 hits={f0_hits} (Q={np.mean(f0_Q)*100 if f0_Q else 0:.2f})", flush=True)
print("Saved.", flush=True)
