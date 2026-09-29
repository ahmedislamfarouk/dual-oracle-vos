"""
OW-VIS dual-metric eval on MOSE (open-world VOS benchmark, train split w/ full GT).
Mirrors eval_dualmetric_val30.py exactly (greedy + bijective + detection P/R);
only the dataset paths, sequence selection, and an optional frame cap differ.
GPU: 1
"""
import sys, os, json, glob, tempfile, shutil, time, random
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

MOSE_ROOT  = os.environ.get("MOSE_ROOT", "/path/to/ow_vis/mose_eval/MOSE/train")
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
MATCH_J_THRESH = 0.5
N_SEQS     = int(os.environ.get("MOSE_NSEQS", "30"))
FRAME_CAP  = int(os.environ.get("MOSE_FRAMECAP", "100"))   # cap frames/video to bound runtime
SEED       = int(os.environ.get("MOSE_SEED", "1234"))
OUT_DIR    = os.environ.get("MOSE_OUT", "/path/to/ow_vis/results/mose_dualmetric")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_ROOT = f"{MOSE_ROOT}/JPEGImages"
ANN_ROOT = f"{MOSE_ROOT}/Annotations"
all_seqs = sorted(d for d in os.listdir(IMG_ROOT) if os.path.isdir(f"{IMG_ROOT}/{d}"))
random.seed(SEED)
SEQS = sorted(random.sample(all_seqs, min(N_SEQS, len(all_seqs))))
print(f"MOSE: {len(all_seqs)} total train videos; evaluating fixed seeded subset of {len(SEQS)} "
      f"(seed={SEED}, frame_cap={FRAME_CAP}): greedy + bijective + detection P/R", flush=True)
with open(f"{OUT_DIR}/SEQS.txt", "w") as f: f.write("\n".join(SEQS))

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def eval_sequence(seq):
    frame_paths = sorted(glob.glob(f"{IMG_ROOT}/{seq}/*.jpg"))[:FRAME_CAP]
    mask_paths  = sorted(glob.glob(f"{ANN_ROOT}/{seq}/*.png"))[:FRAME_CAP]
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

    tmpdir = tempfile.mkdtemp(prefix=f"mose_{seq}_")
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

    Jmat = np.zeros((n_gt, n_boxes)); Fmat = np.zeros((n_gt, n_boxes))
    for gi, gt_id in enumerate(gt_ids):
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]
        for ti in range(n_boxes):
            tmasks = pred_by_track[ti]
            Jmat[gi, ti] = float(np.mean([compute_j_score(tmasks[i], gt_seq[i]) for i in range(n)]))
            Fmat[gi, ti] = float(np.mean([compute_f_score(tmasks[i], gt_seq[i]) for i in range(n)]))

    gi_idx = Jmat.argmax(axis=1)
    J_greedy = float(Jmat[np.arange(n_gt), gi_idx].mean())
    F_greedy = float(Fmat[np.arange(n_gt), gi_idx].mean())

    rows, cols = linear_sum_assignment(-Jmat)
    Jb = np.zeros(n_gt); Fb = np.zeros(n_gt)
    matched_pairs = []
    for r, c in zip(rows, cols):
        Jb[r] = Jmat[r, c]; Fb[r] = Fmat[r, c]
        matched_pairs.append((r, c, Jmat[r, c]))
    J_bij = float(Jb.mean()); F_bij = float(Fb.mean())

    TP = sum(1 for _, _, j in matched_pairs if j >= MATCH_J_THRESH)
    FP = n_boxes - TP
    FN = n_gt - TP
    print(f"  {seq}: nf={n} n_gt={n_gt} n_track={n_boxes} | greedy JF={(J_greedy+F_greedy)/2:.3f} "
          f"bij JF={(J_bij+F_bij)/2:.3f} | TP={TP} FP={FP} FN={FN}", flush=True)
    return {"J_greedy": round(J_greedy,4), "F_greedy": round(F_greedy,4), "JF_greedy": round((J_greedy+F_greedy)/2,4),
            "J_bij": round(J_bij,4), "F_bij": round(F_bij,4), "JF_bij": round((J_bij+F_bij)/2,4),
            "TP": TP, "FP": FP, "FN": FN, "n_gt": n_gt, "n_track": n_boxes}


results = {}
t0 = time.time()
for si, seq in enumerate(SEQS):
    print(f"\n[{si+1}/{len(SEQS)}] {seq}", flush=True)
    try:
        r = eval_sequence(seq)
    except Exception as e:
        print(f"  FATAL on {seq}: {e}", flush=True); r = None
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
           "n_seqs": len(results), "match_j_thresh": MATCH_J_THRESH,
           "frame_cap": FRAME_CAP, "seed": SEED, "dataset": "MOSE-train"}
    with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(out, fp, indent=2)

out["total_s"] = round(time.time()-t0, 1)
with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(out, fp, indent=2)
open(f"{OUT_DIR}/MOSE_SCORING_DONE", "w").write("done")
print(f"\n=== MOSE DUAL-METRIC FINAL ===", flush=True)
print(f"  greedy  J&F = {out['mean_JF_greedy']:.4f}", flush=True)
print(f"  bijective J&F = {out['mean_JF_bij']:.4f}", flush=True)
print(f"  inflation gap = {(out['mean_JF_greedy']-out['mean_JF_bij'])*100:+.2f}pp", flush=True)
print(f"  detection precision = {out['det_precision']:.4f}  recall = {out['det_recall']:.4f}  "
      f"(TP={out['TP']} FP={out['FP']} FN={out['FN']}, mean {out['mean_tracks_per_video']} tracks/video)", flush=True)
