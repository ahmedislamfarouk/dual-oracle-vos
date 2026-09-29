"""
EXP3 — TIDE COMPONENT ABLATION on MOSE. Isolates which of {backward propagation,
coverage gate, commitment gate} drives the discovery gain over frame-0-only.

Variants (all share frame-0 forward pool; discovery = sparse re-detection at n/4,n/2,3n/4):
  V0 base : frame-0 only.
  V1 fwd  : + discovery, cover-gated boxes, FORWARD-only propagation, no commit gate.
  V2 bidir: + discovery, cover-gated boxes, BIDIRECTIONAL propagation, no commit gate.
  V3 nocov: + discovery, ALL re-detection boxes (NO coverage gate), bidirectional, no commit gate.
  V4 full : + discovery, cover-gated boxes, bidirectional, + commitment gate (gate_disc).

Comparisons: V2-V1 = backward propagation; V2-V3 = coverage gate; V4-V2 = commitment gate.
Reports per variant: committed bijective J&F, R (Rstrict), greedy, prec, tracks/vid, commit/vid.
Env: ROOT, CUDA_VISIBLE_DEVICES, MAXF(60), MAXVIDS(40), TAG, COVER_MAX(0.5),
     DISC_PERSIST_MIN(0.7), DISC_COMMIT_COVER_MAX(0.3).
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
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
W, H = TARGET_SIZE
MAXF = int(os.environ.get("MAXF", "60")); MAXVIDS = int(os.environ.get("MAXVIDS", "40"))
TAG = os.environ.get("TAG", "tide_ablate")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5
COVER_MAX = float(os.environ.get("COVER_MAX", "0.5"))
DISC_PERSIST_MIN = float(os.environ.get("DISC_PERSIST_MIN", "0.7"))
DISC_COMMIT_COVER_MAX = float(os.environ.get("DISC_COMMIT_COVER_MAX", "0.3"))

def get_boxes(frame_rgb):
    d = det.detect(frame_rgb); out = []
    for b in list(d.get("known_boxes", [])) + list(d.get("unknown_boxes", [])):
        out.append(list(b.tolist() if hasattr(b, "tolist") else b))
    return out

def _clip_box(bx):
    x1, y1, x2, y2 = [float(v) for v in bx[:4]]
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(x2, W)); y2 = max(y1+1, min(y2, H))
    return x1, y1, x2, y2

def propagate_forward0(tmpd, boxes, n):
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

def propagate_anchor(tmpd, boxes_at, anchor, n, reverses):
    """Propagate boxes seeded at `anchor`. reverses in {(False,), (False,True)}."""
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes_at))}
    conf = {i: [] for i in range(len(boxes_at))}
    if not boxes_at: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes_at):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i, box=np.array([[x1, y1, x2, y2]], np.float32))
        for reverse in reverses:
            for fi, oids, masks in sam2.propagate_in_video(st, start_frame_idx=int(anchor), reverse=reverse):
                if fi < 0 or fi >= n: continue
                for oid, ml in zip(oids, masks):
                    logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    pred[int(oid)][fi] = m
                    pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf

print(f"Loading DETR + SAM2 Large... ROOT={ROOT} COVER_MAX={COVER_MAX} "
      f"commit-gate P>={DISC_PERSIST_MIN} cover<{DISC_COMMIT_COVER_MAX}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

MET = ["greedy", "commit_bij", "prec", "rec", "ntrk", "ncommit", "Rstrict", "Qstrict"]
ARMS = ["V0", "V1", "V2", "V3", "V4"]
agg = {a: {k: [] for k in MET} for a in ARMS}
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
    n = len(frames)
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {vid}: no GT, skip", flush=True); continue
    boxes0 = [_clip_box(b) for b in get_boxes(frames[0])]
    if not boxes0:
        print(f"  [{si+1}] {vid}: no det f0, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="tabl_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    predF, confF = propagate_forward0(tmpd, boxes0, n)

    redet = sorted(set(f for f in (n//4, n//2, (3*n)//4) if 0 < f < n))
    # accumulate discovery tracks. keys: b_ (bidir), f_ (forward). gated flag per box.
    pred, conf, origin, gated = {}, {}, {}, {}
    for i in predF: k = f"f0_{i}"; pred[k] = predF[i]; conf[k] = confF[i]; origin[k] = "f0"
    # running cover (full-res union) of f0 forward + accepted (cover-gated) bidir disc at each frame
    cover_accum = {kf: np.zeros((H, W), np.uint8) for kf in redet}
    kept_bidir_gated = []   # keys of gated bidir tracks, for cover accumulation across frames
    idx = 0
    for kf in redet:
        allb = [_clip_box(b) for b in get_boxes(frames[kf])]
        if not allb: continue
        bp, bc = propagate_anchor(tmpd, allb, kf, n, (False, True))   # bidir, all boxes
        fp, fc = propagate_anchor(tmpd, allb, kf, n, (False,))         # forward-only, all boxes
        # cover at frame kf = union of f0 forward masks + accepted gated bidir tracks so far
        cover = np.zeros((H, W), np.uint8)
        for i in predF: cover |= predF[i][kf]
        for gk in kept_bidir_gated: cover |= pred[gk][kf]
        for j, b in enumerate(allb):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x2 = min(W, max(x1+1, x2)); y2 = min(H, max(y1+1, y2))
            area = (x2-x1)*(y2-y1)
            g = (area > 0) and (int(cover[y1:y2, x1:x2].sum())/area < COVER_MAX)
            bk = f"b_{kf}_{idx}"; fk = f"f_{kf}_{idx}"
            pred[bk] = bp[j]; conf[bk] = bc[j]; origin[bk] = "b"; gated[bk] = g
            pred[fk] = fp[j]; conf[fk] = fc[j]; origin[fk] = "f"; gated[fk] = g
            if g: kept_bidir_gated.append(bk)
            idx += 1
    shutil.rmtree(tmpd, ignore_errors=True)

    tids = list(pred.keys())
    f0_ids = [t for t in tids if origin[t] == "f0"]

    sig = {}
    for t in tids:
        ne = [int(m.sum()) for m in pred[t] if m.sum() > 0]
        sig[t] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                      quality=(len(ne)/n)*(float(np.mean(conf[t])) if conf.get(t) else 0.0))
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
        Js = np.stack([Jrow[t] for t in subset]); out = []
        for gi, g in enumerate(gt_ids):
            bt = int(np.argmax(Js[:, gi])); out.append((Js[bt, gi] + bF(subset[bt], g)) / 2)
        return float(np.mean(out))

    def metrics(subset_greedy, committed):
        m = dict(greedy=greedy_of(subset_greedy), commit_bij=0.0, prec=0.0, rec=0.0,
                 ntrk=len(subset_greedy), ncommit=len(committed), Rstrict=0.0, Qstrict=0.0)
        if committed:
            Jsub = np.stack([Jrow[t] for t in committed])
            ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sh = 0.0
            for r, c in zip(ri, ci):
                q = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2; per_gt[c] = q
                if Jsub[r, c] > 0.5: nhit += 1; sh += q
            m["commit_bij"] = float(np.mean(per_gt)); m["prec"] = nhit/len(committed)
            m["rec"] = nhit/len(gt_ids); m["Rstrict"] = nhit/len(gt_ids); m["Qstrict"] = (sh/nhit) if nhit else 0.0
        return m

    comm_f0 = select_committed(f0_ids)
    union_d = np.logical_or.reduce([dstack[t] for t in comm_f0]) if comm_f0 else np.zeros_like(next(iter(dstack.values())))

    b_gated  = [t for t in tids if origin[t] == "b" and gated[t]]
    b_all    = [t for t in tids if origin[t] == "b"]
    f_gated  = [t for t in tids if origin[t] == "f" and gated[t]]

    def gate_disc(disc_keys, persist_thr, cover_thr):
        keep = []
        for t in disc_keys:
            if sig[t]["persistence"] < persist_thr: continue
            if not (0 < sig[t]["mean_area"] < AREA_MAX): continue
            dt = dstack[t]; area = int(dt.sum())
            inter = int(np.logical_and(dt, union_d).sum())
            ov = inter/area if area else 1.0
            if ov < cover_thr: keep.append(t)
        return keep

    v4_disc = gate_disc(b_gated, DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX)

    res = {}
    res["V0"] = metrics(f0_ids, comm_f0)
    s1 = f0_ids + f_gated; res["V1"] = metrics(s1, select_committed(s1))
    s2 = f0_ids + b_gated; res["V2"] = metrics(s2, select_committed(s2))
    s3 = f0_ids + b_all;   res["V3"] = metrics(s3, select_committed(s3))
    s4 = f0_ids + v4_disc; res["V4"] = metrics(s4, select_committed(s4))

    for a in ARMS:
        for k in MET: agg[a][k].append(res[a][k])
    per_seq[vid] = {a: {k: round(res[a][k], 4) for k in MET} for a in ARMS}
    per_seq[vid].update(ngt=len(gt_ids), nframe=n, n_disc_all=len(b_all), n_disc_gated=len(b_gated), n_v4=len(v4_disc))
    print(f"  [{si+1:3d}] {vid:16s} JF V0={res['V0']['commit_bij']:.3f} V1={res['V1']['commit_bij']:.3f} "
          f"V2={res['V2']['commit_bij']:.3f} V3={res['V3']['commit_bij']:.3f} V4={res['V4']['commit_bij']:.3f} | "
          f"R V0={res['V0']['Rstrict']:.2f} V4={res['V4']['Rstrict']:.2f} "
          f"discAll={len(b_all)} gated={len(b_gated)} v4={len(v4_disc)} gt={len(gt_ids)} t={time.time()-t0:.0f}s", flush=True)

    if (si+1) % 3 == 0 or si == len(vids)-1:
        def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
        summ = {a: {k: Mv(a, k) for k in MET} for a in ARMS}
        out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS, COVER_MAX=COVER_MAX,
                   DISC_PERSIST_MIN=DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX=DISC_COMMIT_COVER_MAX,
                   n=len(agg["V0"]["greedy"])), summary=summ, per_seq=per_seq)
        od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)

def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
print(f"\n=== TIDE ABLATION on {TAG} ({len(agg['V0']['greedy'])} vids, MAXF={MAXF}) ===", flush=True)
desc = {"V0": "base f0-only", "V1": "fwd-only gated", "V2": "bidir gated",
        "V3": "bidir NO-cover-gate", "V4": "full (bidir+cover+commit gate)"}
for a in ARMS:
    print(f"[{a} {desc[a]:32s}] commitJF={Mv(a,'commit_bij')*100:5.2f} R={Mv(a,'Rstrict')*100:5.2f} "
          f"greedy={Mv(a,'greedy')*100:5.2f} prec={Mv(a,'prec')*100:4.1f} "
          f"trk={Mv(a,'ntrk'):5.1f} commit={Mv(a,'ncommit'):4.1f}", flush=True)
print(f"backward-prop effect  (V2-V1): {(Mv('V2','commit_bij')-Mv('V1','commit_bij'))*100:+.2f} JF", flush=True)
print(f"coverage-gate effect  (V2-V3): {(Mv('V2','commit_bij')-Mv('V3','commit_bij'))*100:+.2f} JF", flush=True)
print(f"commitment-gate effect(V4-V2): {(Mv('V4','commit_bij')-Mv('V2','commit_bij'))*100:+.2f} JF", flush=True)
print("Saved.", flush=True)
