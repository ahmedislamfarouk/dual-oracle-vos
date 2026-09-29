"""
GT-FREE TRACK SELECTION EXPERIMENT (defuses the "proposal-recall in disguise" critique).

Same all-boxes MC-VOS pipeline as eval_davis_v2_large.py (DETR all-boxes -> SAM2 Large
propagation), but instead of letting the GT pick the best track per object, the system
COMMITS to a set of tracks using ONLY GT-free signals:
  - temporal persistence (fraction of frames the track mask is non-empty)
  - SAM2 mask confidence (mean positive logit over the mask)
  - temporal-mask-IoU dedup (same temporal-coherence principle as the memory bank):
    keep a track only if it does not heavily overlap an already-kept, higher-quality track.

We then score the COMMITTED set two ways and compare to the GT-best-match ORACLE:
  - greedy oracle J&F  : for each GT, best track over ALL tracks  (== paper headline)
  - committed bijective: one-to-one Hungarian (committed tracks vs GT), unmatched GT = 0
  - committed precision / recall at J>0.5

Retention = committed_bijective_JF / greedy_oracle_JF.  High retention + improved
precision = the method's own components select objects without GT.
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
# Work around a broken torchao version probe in transformers (version.parse(None)).
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
import torch, cv2, numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)            # (W, H)
HW = TARGET_SIZE[0] * TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60"))     # frames per seq (match harness)
NSEQ = int(os.environ.get("NSEQ", "30"))     # sequences

# GT-free selection hyperparameters (fixed a priori, NOT tuned on GT)
PERSIST_MIN = 0.5     # track must be present in >=50% of frames
AREA_MAX    = 0.50    # drop background-spanning tracks (>50% of frame)
DEDUP_IOU   = 0.50    # drop a track that overlaps a kept higher-quality track >0.5

def temporal_iou(a_masks, b_masks):
    inter = union = 0
    for a, b in zip(a_masks, b_masks):
        ab = np.logical_and(a, b).sum()
        au = np.logical_or(a, b).sum()
        inter += ab; union += au
    return inter / union if union > 0 else 0.0

def pair_jf(pred_seq, gt_seq):
    js = [compute_j_score(pred_seq[i], gt_seq[i]) for i in range(len(gt_seq))]
    fs = [compute_f_score(pred_seq[i], gt_seq[i]) for i in range(len(gt_seq))]
    mj, mf = float(np.mean(js)), float(np.mean(fs))
    return mj, mf

print("Loading DETR + SAM2 Large...", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

# Use the OFFICIAL DAVIS-2017 val split (val.txt), not alphabetical listdir of the
# 90-seq TrainVal 480p folder. OFFICIAL=1 (default) reads val.txt; OFFICIAL=0 keeps
# the legacy listdir[:NSEQ] behaviour for direct comparison.
if os.environ.get("OFFICIAL", "1") == "1":
    with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as _f:
        val_seqs = [ln.strip() for ln in _f if ln.strip()][:NSEQ]
else:
    val_seqs = sorted(os.listdir(f"{DAVIS_ROOT}/JPEGImages/480p"))[:NSEQ]

agg = {"oracle_jf": [], "oracle_j": [], "oracle_f": [],
       "commit_bij_jf": [], "commit_bij_j": [], "commit_bij_f": [], "prec": [], "rec": [],
       "n_tracks": [], "n_commit": [], "n_gt": []}
per_seq = {}

for si, seq in enumerate(val_seqs):
    t0 = time.time()
    fps = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mps = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    if not fps or not mps:
        print(f"  [{si+1}] {seq}: missing, skip", flush=True); continue
    n = min(len(fps), len(mps), MAXF)
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in fps[:n]]
    gtm = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mps[:n]]
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {seq}: no GT, skip", flush=True); continue

    d0 = det.detect(frames[0])
    boxes = []
    for i, b in enumerate(d0.get("known_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"k{i}"))
    for i, b in enumerate(d0.get("unknown_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"u{i}"))
    if not boxes:
        print(f"  [{si+1}] {seq}: no det, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="gtf_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    omap, i2o = {}, {}
    for _, iid in boxes:
        oid = hash(iid) % 10000
        while oid in omap and omap[oid] != iid:
            oid = (oid + 1) % 10000
        omap[oid] = iid; i2o[iid] = oid

    pred = {iid: [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8) for _ in range(n)] for _, iid in boxes}
    conf = {iid: [] for _, iid in boxes}   # per-frame mean positive logit
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
            for bx, iid in boxes:
                x1, y1, x2, y2 = [float(v) for v in bx[:4]]
                x1 = max(0, min(x1, TARGET_SIZE[0]-1)); y1 = max(0, min(y1, TARGET_SIZE[1]-1))
                x2 = max(x1+1, min(x2, TARGET_SIZE[0])); y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
                sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i2o[iid],
                                           box=np.array([[x1, y1, x2, y2]], dtype=np.float32))
            for fi, oids, masks in sam2.propagate_in_video(st):
                if fi >= n: break
                for oid, ml in zip(oids, masks):
                    iid = omap.get(int(oid))
                    if iid is None: continue
                    logit = ml[0].float().cpu().numpy()
                    m = (logit > 0).astype(np.uint8)
                    if m.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
                        m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    pred[iid][fi] = m
                    pos = logit[logit > 0]
                    conf[iid].append(float(pos.mean()) if pos.size else 0.0)
            sam2.reset_state(st)
    except Exception as e:
        print(f"  SAM2 err {seq}: {e}", flush=True)
    shutil.rmtree(tmpd, ignore_errors=True)

    # ---- GT-free per-track signals ----
    sig = {}
    for _, iid in boxes:
        areas = [int(m.sum()) for m in pred[iid]]
        ne = [a for a in areas if a > 0]
        persistence = len(ne) / n
        mean_area = (np.mean(ne) / HW) if ne else 0.0
        mean_conf = float(np.mean(conf[iid])) if conf[iid] else 0.0
        sig[iid] = dict(persistence=persistence, mean_area=mean_area, mean_conf=mean_conf,
                        quality=persistence * mean_conf)

    # ---- GT-free commit: filter + quality-sorted temporal-IoU dedup ----
    cand = [iid for _, iid in boxes
            if sig[iid]["persistence"] >= PERSIST_MIN and 0 < sig[iid]["mean_area"] < AREA_MAX]
    cand.sort(key=lambda i: sig[i]["quality"], reverse=True)
    committed = []
    for iid in cand:
        if all(temporal_iou(pred[iid], pred[k]) < DEDUP_IOU for k in committed):
            committed.append(iid)

    # ---- scoring (fast: vectorized region-J for selection, boundary-F only on chosen pairs) ----
    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}

    def track_J_vs_gts(iid):
        # mean-over-frames region-J of one track against every GT object (DAVIS convention:
        # both-empty frame -> 1, one-empty -> 0). Vectorized; matches compute_j_score.
        pb = np.stack(pred[iid]).astype(bool).reshape(n, -1)
        out = []
        for g in gt_ids:
            gb = gt_bool[g]
            inter = np.logical_and(pb, gb).sum(1)
            union = np.logical_or(pb, gb).sum(1)
            jf = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
            out.append(float(jf.mean()))
        return out

    def boundary_F(iid, g):
        fs = [compute_f_score(pred[iid][i], gt_seqs[g][i]) for i in range(n)]
        return float(np.mean(fs))

    track_ids = [iid for _, iid in boxes]
    Jall = np.array([track_J_vs_gts(iid) for iid in track_ids])   # [n_tracks, n_gt]

    # greedy oracle over ALL tracks (paper protocol): best track per GT by J, J&F reported
    o_jf = []; o_j = []; o_f = []
    for gi, g in enumerate(gt_ids):
        bt = int(np.argmax(Jall[:, gi]))
        bj = Jall[bt, gi]
        bf = boundary_F(track_ids[bt], g)
        o_jf.append((bj + bf) / 2); o_j.append(bj); o_f.append(bf)
    oracle_jf = float(np.mean(o_jf))
    oracle_j = float(np.mean(o_j)); oracle_f = float(np.mean(o_f))

    # committed bijective Hungarian (assignment by J, DAVIS-standard; J&F reported)
    P = committed
    if P:
        pidx = [track_ids.index(iid) for iid in P]
        Jsub = Jall[pidx]                                          # [n_commit, n_gt]
        ri, ci = linear_sum_assignment(-Jsub)
        per_gt = np.zeros(len(gt_ids))
        per_gt_j = np.zeros(len(gt_ids)); per_gt_f = np.zeros(len(gt_ids))
        n_hit = 0; sumhit = 0.0
        for r, c in zip(ri, ci):
            jval = Jsub[r, c]
            fval = boundary_F(P[r], gt_ids[c])
            per_gt[c] = (jval + fval) / 2
            per_gt_j[c] = jval; per_gt_f[c] = fval
            if jval > 0.5: n_hit += 1; sumhit += per_gt[c]
        commit_bij = float(np.mean(per_gt))         # unmatched GT contribute 0
        commit_bij_j = float(np.mean(per_gt_j)); commit_bij_f = float(np.mean(per_gt_f))
        nmatch=int((per_gt>0).sum()); Rdisc=nmatch/len(gt_ids); Qcond=float(per_gt[per_gt>0].mean()) if nmatch else 0.0
        Rstrict=n_hit/len(gt_ids); Qstrict=(sumhit/n_hit) if n_hit else 0.0
        prec = n_hit / len(P)
        rec = n_hit / len(gt_ids)
    else:
        commit_bij = prec = rec = 0.0
        commit_bij_j = commit_bij_f = 0.0
        nmatch=0; Rdisc=Qcond=Rstrict=Qstrict=0.0

    agg["oracle_jf"].append(oracle_jf); agg["commit_bij_jf"].append(commit_bij)
    agg["oracle_j"].append(oracle_j); agg["oracle_f"].append(oracle_f)
    agg["commit_bij_j"].append(commit_bij_j); agg["commit_bij_f"].append(commit_bij_f)
    agg["prec"].append(prec); agg["rec"].append(rec)
    agg["n_tracks"].append(len(boxes)); agg["n_commit"].append(len(P)); agg["n_gt"].append(len(gt_ids))
    per_seq[seq] = dict(oracle_jf=round(oracle_jf, 4), oracle_j=round(oracle_j, 4),
                        oracle_f=round(oracle_f, 4), commit_bij_j=round(commit_bij_j, 4),
                        commit_bij_f=round(commit_bij_f, 4), commit_bij_jf=round(commit_bij, 4),
                        prec=round(prec, 3), rec=round(rec, 3), nmatch=nmatch, Rdisc=round(Rdisc,4), Qcond=round(Qcond,4), Rstrict=round(Rstrict,4), Qstrict=round(Qstrict,4),
                        n_tracks=len(boxes), n_commit=len(P), n_gt=len(gt_ids))
    print(f"  [{si+1:2d}] {seq:20s} oracle={oracle_jf:.3f} commitBij={commit_bij:.3f} "
          f"P={prec:.2f} R={rec:.2f} tracks={len(boxes)} commit={len(P)} gt={len(gt_ids)} "
          f"t={time.time()-t0:.0f}s", flush=True)

O = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
oracle = O("oracle_jf"); commit = O("commit_bij_jf")
retention = commit / oracle if oracle > 0 else 0.0
print("\n=== GT-FREE SELECTION RESULTS (DAVIS-17 val, all-boxes, SAM2 Large) ===", flush=True)
print(f"sequences            : {len(agg['oracle_jf'])}", flush=True)
print(f"GREEDY ORACLE  J&F   : {oracle*100:.2f}%   (GT picks best track/object -- paper protocol)", flush=True)
print(f"GT-FREE COMMIT J&F   : {commit*100:.2f}%   (bijective Hungarian on committed set)", flush=True)
print(f"RETENTION            : {retention*100:.1f}% of oracle", flush=True)
print(f"committed precision  : {O('prec')*100:.1f}%   (vs ~4.9% all-tracks in paper)", flush=True)
print(f"committed recall     : {O('rec')*100:.1f}%", flush=True)
print(f"mean tracks/video    : {O('n_tracks'):.1f} -> committed {O('n_commit'):.1f}  (GT objs {O('n_gt'):.1f})", flush=True)

out = dict(config=dict(PERSIST_MIN=PERSIST_MIN, AREA_MAX=AREA_MAX, DEDUP_IOU=DEDUP_IOU, MAXF=MAXF, NSEQ=NSEQ),
           summary=dict(n=len(agg["oracle_jf"]), oracle_jf=oracle, commit_bij_jf=commit,
                        oracle_j=float(np.mean(agg["oracle_j"])), oracle_f=float(np.mean(agg["oracle_f"])),
                        commit_bij_j=float(np.mean(agg["commit_bij_j"])), commit_bij_f=float(np.mean(agg["commit_bij_f"])),
                        retention=retention,
                        precision=O("prec"), recall=O("rec"),
                        mean_tracks=O("n_tracks"), mean_commit=O("n_commit"), mean_gt=O("n_gt")),
           per_seq=per_seq)
OUT_DIR = os.environ.get("OUT_DIR", "/path/to/ow_vis/results/sq_davis_fullseq_jf_20260827")
if os.path.exists(os.path.join(OUT_DIR, "results.json")):
    raise SystemExit(f"refusing to overwrite existing {OUT_DIR}/results.json")
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {OUT_DIR}/results.json", flush=True)
