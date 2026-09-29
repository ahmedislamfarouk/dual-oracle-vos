"""
OW-VIS DAVIS 2017 val: TOP-K candidate pruning ablation (MEASURED, not extrapolated).
Runs the full all-boxes pipeline once per sequence (SAM2 propagation of every candidate),
records each track's objectness/confidence score, then evaluates J&F using only the
top-K highest-scoring tracks for K in {1,3,5,10,20,all}. Answers directly: how much
accuracy is lost by pruning the candidate pool? (Section 4.x currently extrapolates this.)
Reports both greedy best-per-GT (paper metric) and bijective Hungarian (FP-penalized).
GPU: 1
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
from scipy.optimize import linear_sum_assignment
import torch, cv2, numpy as np
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480)
K_LIST = [1, 3, 5, 10, 20, 9999]
OUT_DIR = "/path/to/ow_vis/results/topk_val30"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print(f"Top-K pruning ablation on {len(VAL_SEQS)} val seqs, K={K_LIST}", flush=True)


def jf_for_subset(Jmat, Fmat, track_ids):
    """greedy + bijective J&F using only the given track columns."""
    if len(track_ids) == 0:
        n_gt = Jmat.shape[0]
        return 0.0, 0.0
    Js = Jmat[:, track_ids]; Fs = Fmat[:, track_ids]
    gi = Js.argmax(axis=1)
    n_gt = Js.shape[0]
    Jg = float(Js[np.arange(n_gt), gi].mean()); Fg = float(Fs[np.arange(n_gt), gi].mean())
    rows, cols = linear_sum_assignment(-Js)
    Jb = np.zeros(n_gt); Fb = np.zeros(n_gt)
    for r, c in zip(rows, cols): Jb[r] = Js[r, c]; Fb[r] = Fs[r, c]
    return (Jg + Fg) / 2.0, (float(Jb.mean()) + float(Fb.mean())) / 2.0


def eval_sequence(seq):
    frame_paths = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mask_paths  = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    n = min(len(frame_paths), len(mask_paths))
    if n == 0: return None
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]
    h, w = frames[0].shape[:2]
    gt_ids = sorted(set(int(x) for m in gt_masks for x in np.unique(m[m > 0])))
    if not gt_ids: return None

    res = det.detect(frames[0])
    ub = list(res.get("unknown_boxes", [])); us = list(res.get("unknown_scores", []))
    kb = list(res.get("known_boxes", []));   ks = list(res.get("known_scores", []))
    all_boxes  = ub + kb
    all_scores = list(us) + list(ks)         # unified rank key (objectness / class conf)
    n_boxes = len(all_boxes)
    if n_boxes == 0: return None
    order = list(np.argsort(-np.array(all_scores)))   # high score first

    tmpdir = tempfile.mkdtemp(prefix=f"topk_{seq}_")
    for i, fr in enumerate(frames): cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    pred_by_track = {i: [np.zeros((h, w), np.uint8) for _ in range(n)] for i in range(n_boxes)}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, box in enumerate(all_boxes):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1)); x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=i, box=np.array([[x1,y1,x2,y2]], dtype=np.float32))
            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n: break
                for oi, ml in zip(obj_ids, masks):
                    m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_by_track[int(oi)][frame_idx] = m
            sam2.reset_state(state)
    except Exception as e:
        print(f"  SAM2 error {seq}: {e}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    n_gt = len(gt_ids)
    Jmat = np.zeros((n_gt, n_boxes)); Fmat = np.zeros((n_gt, n_boxes))
    for gi2, gt_id in enumerate(gt_ids):
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]
        for ti in range(n_boxes):
            tm = pred_by_track[ti]
            Jmat[gi2, ti] = float(np.mean([compute_j_score(tm[i], gt_seq[i]) for i in range(n)]))
            Fmat[gi2, ti] = float(np.mean([compute_f_score(tm[i], gt_seq[i]) for i in range(n)]))

    out = {"n_gt": n_gt, "n_boxes": n_boxes}
    for K in K_LIST:
        sel = order[:K]
        g, b = jf_for_subset(Jmat, Fmat, sel)
        kk = "all" if K >= n_boxes else str(K)
        out[f"greedy_K{kk}"] = round(g, 4); out[f"bij_K{kk}"] = round(b, 4)
    msg = " ".join(f"K{('all' if K>=n_boxes else K)}={out.get('greedy_K'+('all' if K>=n_boxes else str(K)),0):.3f}" for K in K_LIST)
    print(f"  {seq}: n_gt={n_gt} n_box={n_boxes} | greedy {msg}", flush=True)
    return out


results = {}
t0 = time.time()
for si, seq in enumerate(VAL_SEQS):
    print(f"[{si+1}/{len(VAL_SEQS)}] {seq}", flush=True)
    r = eval_sequence(seq)
    if r: results[seq] = r
    # aggregate (use fixed K labels; 'all' handled per-seq so aggregate the numeric Ks + all)
    agg = {}
    for K in K_LIST:
        for proto in ["greedy", "bij"]:
            vals = []
            for v in results.values():
                key_all = f"{proto}_Kall"; key_k = f"{proto}_K{K}"
                if K >= v["n_boxes"]:
                    if key_all in v: vals.append(v[key_all])
                elif key_k in v: vals.append(v[key_k])
            label = "all" if K == 9999 else K
            if vals: agg[f"{proto}_K{label}"] = round(float(np.mean(vals)), 4)
    summary = {"per_seq": results, "agg_mean": agg, "n_seqs": len(results), "K_LIST": K_LIST}
    with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(summary, fp, indent=2)

summary["total_s"] = round(time.time()-t0, 1)
with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(summary, fp, indent=2)
print("\n=== TOP-K PRUNING FINAL (mean over seqs) ===", flush=True)
for K in K_LIST:
    label = "all" if K == 9999 else K
    print(f"  K={label:<4} greedy J&F={agg.get('greedy_K'+str(label),0):.4f}  bijective J&F={agg.get('bij_K'+str(label),0):.4f}", flush=True)
