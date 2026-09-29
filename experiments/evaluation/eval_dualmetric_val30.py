"""
OW-VIS DAVIS 2017 val (30 seqs): SAME pipeline as eval_allboxes_val30.py (83.37%),
but reports THREE evaluation protocols side by side so the inflation of the
greedy oracle metric can be quantified for reviewers:

  (A) greedy  : best-match-per-GT, reuse allowed, FPs free  (the paper's current metric)
  (B) bijective: one-to-one Hungarian GT<->track assignment (DAVIS-unsupervised protocol);
                 unmatched GT score 0, so missing/duplicate tracks ARE penalized
  (C) detection P/R: TP = bijective pair with J>=0.5, FP = leftover tracks, FN = leftover GT

Builds the full per-video J/F matrix [n_gt x n_track] once and derives all three.
GPU: 1
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
MATCH_J_THRESH = 0.5
OUT_DIR    = "/path/to/ow_vis/results/dualmetric_val30"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
print(f"Evaluating {len(VAL_SEQS)} val seqs: greedy + bijective + detection P/R", flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


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
    all_boxes = list(res.get("unknown_boxes", [])) + list(res.get("known_boxes", []))
    n_boxes = len(all_boxes)
    n_gt = len(gt_ids)
    if n_boxes == 0:
        return {"J_greedy": 0.0, "F_greedy": 0.0, "JF_greedy": 0.0,
                "J_bij": 0.0, "F_bij": 0.0, "JF_bij": 0.0,
                "TP": 0, "FP": 0, "FN": n_gt, "n_gt": n_gt, "n_track": 0}

    tmpdir = tempfile.mkdtemp(prefix=f"dual_{seq}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    pred_by_track = {i: [np.zeros((h, w), np.uint8) for _ in range(n)] for i in range(n_boxes)}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, box in enumerate(all_boxes):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
                x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=i,
                    box=np.array([[x1,y1,x2,y2]], dtype=np.float32))
            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n: break
                for oi, ml in zip(obj_ids, masks):
                    m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_by_track[int(oi)][frame_idx] = m
            sam2.reset_state(state)
    except Exception as e:
        print(f"  SAM2 error on {seq}: {e}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Full J/F matrices [n_gt x n_track]
    Jmat = np.zeros((n_gt, n_boxes)); Fmat = np.zeros((n_gt, n_boxes))
    for gi, gt_id in enumerate(gt_ids):
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]
        for ti in range(n_boxes):
            tmasks = pred_by_track[ti]
            Jmat[gi, ti] = float(np.mean([compute_j_score(tmasks[i], gt_seq[i]) for i in range(n)]))
            Fmat[gi, ti] = float(np.mean([compute_f_score(tmasks[i], gt_seq[i]) for i in range(n)]))

    # (A) greedy best-per-GT
    gi_idx = Jmat.argmax(axis=1)
    J_greedy = float(Jmat[np.arange(n_gt), gi_idx].mean())
    F_greedy = float(Fmat[np.arange(n_gt), gi_idx].mean())

    # (B) bijective Hungarian (maximize J); unmatched GT -> 0
    rows, cols = linear_sum_assignment(-Jmat)
    Jb = np.zeros(n_gt); Fb = np.zeros(n_gt)
    matched_pairs = []
    for r, c in zip(rows, cols):
        Jb[r] = Jmat[r, c]; Fb[r] = Fmat[r, c]
        matched_pairs.append((r, c, Jmat[r, c]))
    J_bij = float(Jb.mean()); F_bij = float(Fb.mean())

    # (C) detection precision/recall at J>=0.5 over the bijective assignment
    TP = sum(1 for _, _, j in matched_pairs if j >= MATCH_J_THRESH)
    FP = n_boxes - TP
    FN = n_gt - TP
    print(f"  {seq}: n_gt={n_gt} n_track={n_boxes} | greedy JF={(J_greedy+F_greedy)/2:.3f} "
          f"bij JF={(J_bij+F_bij)/2:.3f} | TP={TP} FP={FP} FN={FN}", flush=True)
    return {"J_greedy": round(J_greedy,4), "F_greedy": round(F_greedy,4), "JF_greedy": round((J_greedy+F_greedy)/2,4),
            "J_bij": round(J_bij,4), "F_bij": round(F_bij,4), "JF_bij": round((J_bij+F_bij)/2,4),
            "TP": TP, "FP": FP, "FN": FN, "n_gt": n_gt, "n_track": n_boxes}


results = {}
t0 = time.time()
for si, seq in enumerate(VAL_SEQS):
    print(f"\n[{si+1}/{len(VAL_SEQS)}] {seq}", flush=True)
    r = eval_sequence(seq)
    if r: results[seq] = r
    def agg(key): return round(float(np.mean([v[key] for v in results.values()])), 4) if results else 0.0
    tot_tp = sum(v["TP"] for v in results.values()); tot_fp = sum(v["FP"] for v in results.values()); tot_fn = sum(v["FN"] for v in results.values())
    out = {"per_seq": results,
           "mean_JF_greedy": agg("JF_greedy"), "mean_J_greedy": agg("J_greedy"), "mean_F_greedy": agg("F_greedy"),
           "mean_JF_bij": agg("JF_bij"), "mean_J_bij": agg("J_bij"), "mean_F_bij": agg("F_bij"),
           "det_precision": round(tot_tp/(tot_tp+tot_fp), 4) if (tot_tp+tot_fp) else 0.0,
           "det_recall": round(tot_tp/(tot_tp+tot_fn), 4) if (tot_tp+tot_fn) else 0.0,
           "TP": tot_tp, "FP": tot_fp, "FN": tot_fn,
           "mean_tracks_per_video": round(float(np.mean([v["n_track"] for v in results.values()])), 1) if results else 0.0,
           "n_seqs": len(results), "match_j_thresh": MATCH_J_THRESH}
    with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(out, fp, indent=2)

out["total_s"] = round(time.time()-t0, 1)
with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(out, fp, indent=2)
print(f"\n=== DUAL-METRIC FINAL ===", flush=True)
print(f"  greedy  J&F = {out['mean_JF_greedy']:.4f}  (paper's current oracle metric)", flush=True)
print(f"  bijective J&F = {out['mean_JF_bij']:.4f}  (DAVIS-unsupervised protocol, FPs penalized)", flush=True)
print(f"  inflation gap = {(out['mean_JF_greedy']-out['mean_JF_bij'])*100:+.2f}pp", flush=True)
print(f"  detection precision = {out['det_precision']:.4f}  recall = {out['det_recall']:.4f}  "
      f"(TP={out['TP']} FP={out['FP']} FN={out['FN']}, mean {out['mean_tracks_per_video']} tracks/video)", flush=True)
