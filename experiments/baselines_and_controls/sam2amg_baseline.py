"""
SAM2 AUTOMATIC-MASK-GENERATION (AMG) ANNOTATION-FREE BASELINE.

Identical to MC-VOS except the frame-0 detector: instead of DETR class-agnostic
objectness boxes, we use SAM2's automatic mask generator to propose candidate objects
on frame 0, take each proposal's bounding box, and run the SAME SAM2-Large propagation
and the SAME dual-metric scoring (greedy best-per-GT + bijective Hungarian + detection
precision/recall) on the official DAVIS-2017 val-30. This isolates whether DETR
objectness specifically matters, or whether a generic SAM2 proposer works as well.
Reports the same columns as the protocol-robustness table for a direct comparison.
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
from metrics import compute_f_score
import torch, cv2, numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import tempfile, shutil

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60")); NSEQ = int(os.environ.get("NSEQ", "30"))
MAX_CAND = int(os.environ.get("MAX_CAND", "80"))   # cap AMG proposals by score (runtime bound)
PPS = int(os.environ.get("PPS", "24"))             # AMG points_per_side

print("Loading SAM2 Large (image AMG + video predictor)...", flush=True)
sam2_img = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE, apply_postprocessing=False)
amg = SAM2AutomaticMaskGenerator(sam2_img, points_per_side=PPS,
                                 pred_iou_thresh=0.7, stability_score_thresh=0.85,
                                 min_mask_region_area=100)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as _f:
    val_seqs = [ln.strip() for ln in _f if ln.strip()][:NSEQ]

agg = {"greedy": [], "bij": [], "prec": [], "rec": [], "ntracks": [], "ngt": []}
per_seq = {}
TPFP = dict(tp=0, fp=0, fn=0)

for si, seq in enumerate(val_seqs):
    t0 = time.time()
    fps = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mps = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    if not fps or not mps: continue
    n = min(len(fps), len(mps), MAXF)
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in fps[:n]]
    gtm = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mps[:n]]
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids: continue

    # ---- AMG proposals on frame 0 -> boxes ----
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks0 = amg.generate(frames[0])
    masks0 = sorted(masks0, key=lambda d: d.get("predicted_iou", 0)*d.get("stability_score", 0), reverse=True)[:MAX_CAND]
    boxes = []
    for i, md in enumerate(masks0):
        x, y, w, h = md["bbox"]
        boxes.append(([float(x), float(y), float(x+w), float(y+h)], f"a{i}"))
    if not boxes:
        print(f"  [{si+1}] {seq}: no AMG proposals, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="amg_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    omap, i2o = {}, {}
    for _, iid in boxes:
        oid = hash(iid) % 100000
        while oid in omap and omap[oid] != iid: oid = (oid+1) % 100000
        omap[oid] = iid; i2o[iid] = oid

    pred = {iid: [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8) for _ in range(n)] for _, iid in boxes}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
            for bx, iid in boxes:
                x1, y1, x2, y2 = bx
                x1 = max(0, min(x1, TARGET_SIZE[0]-1)); y1 = max(0, min(y1, TARGET_SIZE[1]-1))
                x2 = max(x1+1, min(x2, TARGET_SIZE[0])); y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
                sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i2o[iid],
                                           box=np.array([[x1, y1, x2, y2]], dtype=np.float32))
            for fi, oids, masks in sam2.propagate_in_video(st):
                if fi >= n: break
                for oid, ml in zip(oids, masks):
                    iid = omap.get(int(oid))
                    if iid is None: continue
                    m = (ml[0].float().cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
                        m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    pred[iid][fi] = m
            sam2.reset_state(st)
    except Exception as e:
        print(f"  SAM2 err {seq}: {e}", flush=True)
    shutil.rmtree(tmpd, ignore_errors=True)

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    track_ids = [iid for _, iid in boxes]
    def track_J(iid):
        pb = np.stack(pred[iid]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        return out
    _fc = {}
    def bF(iid, g):
        k = (iid, g)
        if k not in _fc:
            _fc[k] = float(np.mean([compute_f_score(pred[iid][i], gt_seqs[g][i]) for i in range(n)]))
        return _fc[k]
    Jall = np.array([track_J(iid) for iid in track_ids])

    # greedy best-per-GT
    g_jf = []
    for gi, g in enumerate(gt_ids):
        bt = int(np.argmax(Jall[:, gi])); g_jf.append((Jall[bt, gi] + bF(track_ids[bt], g)) / 2)
    greedy = float(np.mean(g_jf))

    # bijective Hungarian (all tracks) + precision/recall
    ri, ci = linear_sum_assignment(-Jall)   # may match a subset of tracks (n_gt assignments)
    per_gt = np.zeros(len(gt_ids)); nhit = 0; matchedJ = {}
    for r, c in zip(ri, ci):
        matchedJ[c] = Jall[r, c]
        per_gt[c] = (Jall[r, c] + bF(track_ids[r], gt_ids[c])) / 2
        if Jall[r, c] > 0.5: nhit += 1
    bij = float(np.mean(per_gt))
    tp = nhit; fn = len(gt_ids) - nhit; fp = len(track_ids) - nhit
    TPFP["tp"] += tp; TPFP["fp"] += fp; TPFP["fn"] += fn
    prec = tp/(tp+fp) if (tp+fp) else 0.0; rec = tp/(tp+fn) if (tp+fn) else 0.0

    agg["greedy"].append(greedy); agg["bij"].append(bij); agg["prec"].append(prec)
    agg["rec"].append(rec); agg["ntracks"].append(len(track_ids)); agg["ngt"].append(len(gt_ids))
    per_seq[seq] = dict(greedy=round(greedy, 4), bij=round(bij, 4), prec=round(prec, 3),
                        rec=round(rec, 3), ntracks=len(track_ids), ngt=len(gt_ids))
    print(f"  [{si+1:2d}] {seq:20s} greedy={greedy:.3f} bij={bij:.3f} P={prec:.2f} R={rec:.2f} "
          f"tracks={len(track_ids)} gt={len(gt_ids)} t={time.time()-t0:.0f}s", flush=True)

O = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
gp = TPFP["tp"]/(TPFP["tp"]+TPFP["fp"]) if (TPFP["tp"]+TPFP["fp"]) else 0.0
gr = TPFP["tp"]/(TPFP["tp"]+TPFP["fn"]) if (TPFP["tp"]+TPFP["fn"]) else 0.0
print("\n=== SAM2-AMG BASELINE (official DAVIS-2017 val-30, SAM2 Large) ===", flush=True)
print(f"sequences        : {len(agg['greedy'])}", flush=True)
print(f"GREEDY  J&F      : {O('greedy')*100:.2f}%   (cf. MC-VOS/DETR 83.44%)", flush=True)
print(f"BIJECTIVE J&F    : {O('bij')*100:.2f}%   (cf. MC-VOS/DETR 83.07%)", flush=True)
print(f"detection P / R  : {gp*100:.1f}% / {gr*100:.1f}%   (global; cf. MC-VOS 4.9% / 83.6%)", flush=True)
print(f"mean tracks/video: {O('ntracks'):.1f}   (cf. MC-VOS 34.4; GT objs {O('ngt'):.1f})", flush=True)
print(f"AMG cfg: points_per_side={PPS}, max_cand={MAX_CAND}", flush=True)
out = dict(config=dict(PPS=PPS, MAX_CAND=MAX_CAND, MAXF=MAXF, NSEQ=NSEQ),
           summary=dict(n=len(agg["greedy"]), greedy=O("greedy"), bij=O("bij"),
                        det_precision=gp, det_recall=gr, mean_tracks=O("ntracks"), mean_gt=O("ngt")),
           per_seq=per_seq)
os.makedirs("/path/to/ow_vis/results/sam2amg_baseline", exist_ok=True)
with open("/path/to/ow_vis/results/sam2amg_baseline/results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/sam2amg_baseline/results.json", flush=True)
