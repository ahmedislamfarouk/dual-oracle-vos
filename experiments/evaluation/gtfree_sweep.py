"""
GT-FREE SELECTOR THRESHOLD SENSITIVITY SWEEP.

Same all-boxes DETR->SAM2 predictions as gtfree_select_eval.py, but per sequence we
cache the (threshold-independent) per-track signals + masks + J-vs-GT matrix ONCE, then
re-run the GT-free commitment + bijective scoring across a grid of the three thresholds
(persistence, area-max, dedup-IoU). Goal: show retention & precision are STABLE, i.e.
the a-priori thresholds were not cherry-picked. One-at-a-time slices + full-grid range.
"""
import sys, os, json, glob, time, itertools
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

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60")); NSEQ = int(os.environ.get("NSEQ", "30"))

# default operating point (the a-priori values reported in the paper)
DEF = dict(P=0.5, A=0.5, D=0.5)
# one-at-a-time grids
GRID_P = [0.3, 0.4, 0.5, 0.6, 0.7]
GRID_A = [0.4, 0.5, 0.6]
GRID_D = [0.4, 0.5, 0.6]
# full configs = all one-at-a-time points (dedup default unless swept), de-duplicated
CONFIGS = {}
for p in GRID_P: CONFIGS[(p, DEF["A"], DEF["D"])] = None
for a in GRID_A: CONFIGS[(DEF["P"], a, DEF["D"])] = None
for d in GRID_D: CONFIGS[(DEF["P"], DEF["A"], d)] = None
CONFIGS = list(CONFIGS.keys())

def temporal_iou(a_masks, b_masks):
    inter = union = 0
    for a, b in zip(a_masks, b_masks):
        inter += np.logical_and(a, b).sum(); union += np.logical_or(a, b).sum()
    return inter/union if union > 0 else 0.0

print("Loading DETR + SAM2 Large...", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as _f:
    val_seqs = [ln.strip() for ln in _f if ln.strip()][:NSEQ]

# per-config accumulators
acc = {c: {"commit": [], "prec": [], "rec": [], "ncommit": []} for c in CONFIGS}
oracle_all = []

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

    d0 = det.detect(frames[0])
    boxes = []
    for i, b in enumerate(d0.get("known_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"k{i}"))
    for i, b in enumerate(d0.get("unknown_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"u{i}"))
    if not boxes: continue

    tmpd = tempfile.mkdtemp(prefix="swp_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    omap, i2o = {}, {}
    for _, iid in boxes:
        oid = hash(iid) % 10000
        while oid in omap and omap[oid] != iid: oid = (oid+1) % 10000
        omap[oid] = iid; i2o[iid] = oid

    pred = {iid: [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8) for _ in range(n)] for _, iid in boxes}
    conf = {iid: [] for _, iid in boxes}
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
                    pos = logit[logit > 0]; conf[iid].append(float(pos.mean()) if pos.size else 0.0)
            sam2.reset_state(st)
    except Exception as e:
        print(f"  SAM2 err {seq}: {e}", flush=True)
    shutil.rmtree(tmpd, ignore_errors=True)

    sig = {}
    for _, iid in boxes:
        ne = [int(m.sum()) for m in pred[iid] if m.sum() > 0]
        sig[iid] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                        quality=(len(ne)/n) * (float(np.mean(conf[iid])) if conf[iid] else 0.0))

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    track_ids = [iid for _, iid in boxes]
    def track_J(iid):
        pb = np.stack(pred[iid]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        return out
    Jall = np.array([track_J(iid) for iid in track_ids])
    _fcache = {}
    def bF(iid, g):
        k = (iid, g)
        if k not in _fcache:
            _fcache[k] = float(np.mean([compute_f_score(pred[iid][i], gt_seqs[g][i]) for i in range(n)]))
        return _fcache[k]

    # oracle (threshold-independent)
    o_jf = []
    for gi, g in enumerate(gt_ids):
        bt = int(np.argmax(Jall[:, gi])); o_jf.append((Jall[bt, gi] + bF(track_ids[bt], g)) / 2)
    oracle_all.append(float(np.mean(o_jf)))

    # pairwise temporal-IoU cached ONCE per seq (config-independent): the dedup decision
    # depends on the threshold D, but the IoU itself does not. tIoU = sum_t|a∩b| / sum_t|a∪b|,
    # and sum_t|a∪b| = area(a)+area(b)-sum_t|a∩b|.
    flat = {iid: np.stack(pred[iid]).astype(bool).reshape(n, -1) for iid in track_ids}
    asum = {iid: int(flat[iid].sum()) for iid in track_ids}
    _tcache = {}
    def TIOU(i, j):
        k = (i, j) if i < j else (j, i)
        if k not in _tcache:
            inter = int(np.logical_and(flat[i], flat[j]).sum())
            uni = asum[i] + asum[j] - inter
            _tcache[k] = inter/uni if uni > 0 else 0.0
        return _tcache[k]

    # sweep configs
    tidx = {iid: k for k, iid in enumerate(track_ids)}
    for (P, A, D) in CONFIGS:
        cand = [iid for _, iid in boxes if sig[iid]["persistence"] >= P and 0 < sig[iid]["mean_area"] < A]
        cand.sort(key=lambda i: sig[i]["quality"], reverse=True)
        committed = []
        for iid in cand:
            if all(TIOU(iid, k) < D for k in committed): committed.append(iid)
        if committed:
            Jsub = Jall[[tidx[i] for i in committed]]
            ri, ci = linear_sum_assignment(-Jsub)
            per_gt = np.zeros(len(gt_ids)); nhit = 0
            for r, c in zip(ri, ci):
                per_gt[c] = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2
                if Jsub[r, c] > 0.5: nhit += 1
            cb, pr, rc = float(np.mean(per_gt)), nhit/len(committed), nhit/len(gt_ids)
        else:
            cb = pr = rc = 0.0
        acc[(P, A, D)]["commit"].append(cb); acc[(P, A, D)]["prec"].append(pr)
        acc[(P, A, D)]["rec"].append(rc); acc[(P, A, D)]["ncommit"].append(len(committed))
    print(f"  [{si+1:2d}] {seq:20s} oracle={oracle_all[-1]:.3f} t={time.time()-t0:.0f}s", flush=True)

ORACLE = float(np.mean(oracle_all))
def row(c):
    a = acc[c]; cb = float(np.mean(a["commit"]))
    return dict(commit=cb, retention=cb/ORACLE, prec=float(np.mean(a["prec"])),
                rec=float(np.mean(a["rec"])), ncommit=float(np.mean(a["ncommit"])))
res = {f"P{P}_A{A}_D{D}": row((P, A, D)) for (P, A, D) in CONFIGS}
print(f"\n=== SENSITIVITY SWEEP (official val-30, oracle J&F={ORACLE*100:.2f}%) ===", flush=True)
print("default P=0.5 A=0.5 D=0.5 :", res["P0.5_A0.5_D0.5"], flush=True)
def slc(name, fixedfmt, items):
    print(f"\n-- vary {name} ({fixedfmt}) --", flush=True)
    for c in items:
        r = res[c]; print(f"  {c:16s} retention={r['retention']*100:5.1f}%  prec={r['prec']*100:5.1f}%  "
                           f"rec={r['rec']*100:5.1f}%  commit={r['ncommit']:.1f}", flush=True)
slc("persistence", "A=0.5,D=0.5", [f"P{p}_A0.5_D0.5" for p in GRID_P])
slc("area-max",    "P=0.5,D=0.5", [f"P0.5_A{a}_D0.5" for a in GRID_A])
slc("dedup-IoU",   "P=0.5,A=0.5", [f"P0.5_A0.5_D{d}" for d in GRID_D])
rets = [res[k]["retention"] for k in res]; precs = [res[k]["prec"] for k in res]
print(f"\nRETENTION across all {len(res)} configs: min={min(rets)*100:.1f}% max={max(rets)*100:.1f}% (spread {(max(rets)-min(rets))*100:.1f}pp)", flush=True)
print(f"PRECISION across all configs: min={min(precs)*100:.1f}% max={max(precs)*100:.1f}%", flush=True)
out = dict(oracle_jf=ORACLE, default=DEF, configs=res,
           retention_range=[min(rets), max(rets)], precision_range=[min(precs), max(precs)])
os.makedirs("/path/to/ow_vis/results/gtfree_sweep", exist_ok=True)
with open("/path/to/ow_vis/results/gtfree_sweep/results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/gtfree_sweep/results.json", flush=True)
