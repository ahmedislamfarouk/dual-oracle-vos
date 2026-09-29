"""
SegTrack v2 Evaluation: SAM2 Large + threshold=0.25 + all 14 sequences.
Uses the same SAM2-based propagation approach as DAVIS eval (not pipeline box-only).
"""
import sys, os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")

import os as _os
_os.chdir("/path/to/ow_vis")

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
import torch, cv2, numpy as np, time, tempfile, shutil
from PIL import Image

SEGTRACK_ROOT = "/tmp/segtrack/SegTrackv2"
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE    = "cuda"
THRESHOLD = 0.25
MAX_FRAMES = 60

# All 14 SegTrack sequences
ALL_SEQS = ["birdfall", "bird_of_paradise", "bmx", "cheetah", "drift",
            "frog", "girl", "hummingbird", "monkey", "monkeydog",
            "parachute", "penguin", "soldier", "worm"]

print(f"Loading OpenWorldDetector (threshold={THRESHOLD})...", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=THRESHOLD, objectness_threshold=0.05)
print("Loading SAM2 LARGE video predictor...", flush=True)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

def load_sequence(seq):
    img_dir = os.path.join(SEGTRACK_ROOT, "JPEGImages", seq)
    gt_dir  = os.path.join(SEGTRACK_ROOT, "GroundTruth", seq)
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.bmp")) +
                       glob.glob(os.path.join(img_dir, "*.jpg")))
    gt_files  = sorted(glob.glob(os.path.join(gt_dir, "*.png")) +
                       glob.glob(os.path.join(gt_dir, "*.bmp")))
    n = min(len(img_files), len(gt_files), MAX_FRAMES)
    frames, gt_masks = [], []
    for i in range(n):
        img = cv2.imread(img_files[i])
        gt  = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gt_masks.append((gt > 0).astype(np.uint8))
    return frames, gt_masks

def eval_sequence_sam2(seq, frames, gt_masks):
    """Use SAM2 propagation for higher quality masks (same as DAVIS eval)."""
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    n = len(frames)

    # Detect all objects in frame 0
    frame0_result = det.detect(frames[0])
    all_boxes_frame0 = []
    known_boxes  = frame0_result.get("known_boxes", [])
    unknown_boxes = frame0_result.get("unknown_boxes", [])
    for i, b in enumerate(known_boxes):
        bl = b.tolist() if hasattr(b, 'tolist') else list(b)
        all_boxes_frame0.append((bl, f"known_{i}"))
    for i, b in enumerate(unknown_boxes):
        bl = b.tolist() if hasattr(b, 'tolist') else list(b)
        all_boxes_frame0.append((bl, f"unknown_{i}"))

    if not all_boxes_frame0:
        print(f"    No detections in frame 0, using empty masks", flush=True)
        j_scores = [compute_j_score(np.zeros((h,w), np.uint8), g) for g in gt_masks]
        f_scores = [compute_f_score(np.zeros((h,w), np.uint8), g) for g in gt_masks]
        return {"J": float(np.mean(j_scores)), "F": float(np.mean(f_scores)), "n_frames": n, "n_boxes": 0}

    # Write frames to temp dir
    tmpdir = tempfile.mkdtemp(prefix=f"segtrack_{seq}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    obj_id_map = {}
    inst_id_to_obj_id = {}
    for box_data, inst_id in all_boxes_frame0:
        obj_id = hash(inst_id) % 10000
        while obj_id in obj_id_map and obj_id_map[obj_id] != inst_id:
            obj_id = (obj_id + 1) % 10000
        obj_id_map[obj_id] = inst_id
        inst_id_to_obj_id[inst_id] = obj_id

    pred_masks = {inst_id: [np.zeros((h, w), np.uint8) for _ in range(n)]
                  for _, inst_id in all_boxes_frame0}

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for box_data, inst_id in all_boxes_frame0:
                x1, y1, x2, y2 = [float(v) for v in box_data[:4]]
                x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w)); y2 = max(y1+1, min(y2, h))
                box_arr = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                sam2.add_new_points_or_box(state, frame_idx=0,
                    obj_id=inst_id_to_obj_id[inst_id], box=box_arr)

            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n:
                    break
                for obj_id, mask_logits in zip(obj_ids, masks):
                    inst_id = obj_id_map.get(int(obj_id))
                    if inst_id is None:
                        continue
                    m = (mask_logits[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_masks[inst_id][frame_idx] = m
            sam2.reset_state(state)
    except Exception as e:
        print(f"    SAM2 error: {e}", flush=True)
        for box_data, inst_id in all_boxes_frame0:
            x1, y1, x2, y2 = [int(float(v)) for v in box_data[:4]]
            x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
            x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
            bm = np.zeros((h,w), np.uint8); bm[y1:y2,x1:x2]=1
            for i in range(n): pred_masks[inst_id][i] = bm

    shutil.rmtree(tmpdir, ignore_errors=True)

    # Build combined mask per frame (union of all instances = foreground)
    j_scores, f_scores = [], []
    for i in range(n):
        combined = np.zeros((h, w), np.uint8)
        for inst_id in pred_masks:
            combined = np.maximum(combined, pred_masks[inst_id][i])
        gt = gt_masks[i]
        if gt.shape != combined.shape:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        j_scores.append(compute_j_score(combined, gt))
        f_scores.append(compute_f_score(combined, gt))

    return {
        "J": float(np.mean(j_scores)),
        "F": float(np.mean(f_scores)),
        "n_frames": n,
        "n_boxes": len(all_boxes_frame0),
        "n_known": len(known_boxes),
        "n_unknown": len(unknown_boxes)
    }

all_results = {}
t0_total = time.time()

for seq in ALL_SEQS:
    t0 = time.time()
    print(f"\nEvaluating {seq}...", flush=True)
    frames, gt_masks = load_sequence(seq)
    if not frames:
        print(f"  {seq}: no frames, skipping", flush=True)
        continue
    r = eval_sequence_sam2(seq, frames, gt_masks)
    if r:
        jf = (r["J"] + r["F"]) / 2
        elapsed = time.time() - t0
        all_results[seq] = {
            "J": round(r["J"], 4), "F": round(r["F"], 4), "JF": round(jf, 4),
            "n_frames": r["n_frames"], "n_boxes": r.get("n_boxes", 0),
            "elapsed_s": round(elapsed, 1)
        }
        print(f"  J={r['J']:.4f}  F={r['F']:.4f}  JF={jf:.4f}  frames={r['n_frames']}  boxes={r.get('n_boxes',0)}  t={elapsed:.1f}s", flush=True)

if all_results:
    mean_J  = float(np.mean([v["J"]  for v in all_results.values()]))
    mean_F  = float(np.mean([v["F"]  for v in all_results.values()]))
    mean_JF = float(np.mean([v["JF"] for v in all_results.values()]))
    total_elapsed = time.time() - t0_total
    print(f"\n=== SEGTRACK RESULTS (SAM2 Large, threshold={THRESHOLD}, all {len(all_results)} seqs) ===", flush=True)
    print(f"J={mean_J:.4f}  F={mean_F:.4f}  J&F={mean_JF:.4f}", flush=True)
    print(f"Total time: {total_elapsed/60:.1f} min", flush=True)

    output = {
        "model": "sam2.1_hiera_large",
        "threshold": THRESHOLD,
        "overall": {"J": round(mean_J, 4), "F": round(mean_F, 4), "JF": round(mean_JF, 4)},
        "n_sequences": len(all_results),
        "total_elapsed_min": round(total_elapsed/60, 1),
        "per_sequence": all_results
    }
    os.makedirs("/path/to/ow_vis/results/segtrack_large", exist_ok=True)
    out = "/path/to/ow_vis/results/segtrack_large/results.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out}", flush=True)
