"""
SegTrack v2 Evaluation on OW-VIS Pipeline.
Evaluates J (IoU) and F (boundary) score against GT masks on 5 sequences.
"""
import sys, os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")

import torch
import cv2
import numpy as np

from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from metrics import compute_j_score, compute_f_score
from pipeline import OWVISPipeline

SEGTRACK_ROOT = "/tmp/segtrack/SegTrackv2"
CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"

# Select 5 diverse sequences with clear GT masks
TARGET_SEQS = ["birdfall", "frog", "parachute", "bird_of_paradise", "worm"]
# Limit frames per sequence for speed
MAX_FRAMES = 50

def build_pipeline(threshold=0.4):
    bank = UnknownInstanceMemoryBank(promote_after_obs=5, prune_after_frames=30)
    emb = CropEmbedder().to(DEVICE)
    try:
        ck = torch.load("/path/to/ow_vis/checkpoints/crop_embedder_best.pt",
                        map_location=DEVICE)
        emb.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        print(f"  [CropEmbedder init: {e}]")
    return OWVISPipeline(
        CKPT, "sam2.1_hiera_base_plus", bank, emb,
        unknown_score_threshold=threshold, use_owdetr=True, device=DEVICE
    )

def load_sequence(seq):
    img_dir = os.path.join(SEGTRACK_ROOT, "JPEGImages", seq)
    gt_dir = os.path.join(SEGTRACK_ROOT, "GroundTruth", seq)
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")) +
                       glob.glob(os.path.join(img_dir, "*.bmp")) +
                       glob.glob(os.path.join(img_dir, "*.jpg")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")) +
                      glob.glob(os.path.join(gt_dir, "*.bmp")))
    
    # Align by index (same ordering)
    n = min(len(img_files), len(gt_files), MAX_FRAMES)
    
    frames = []
    gt_masks = []
    for i in range(n):
        img = cv2.imread(img_files[i])
        gt = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_bin = (gt > 0).astype(np.uint8)
        frames.append(img_rgb)
        gt_masks.append(gt_bin)
    
    return frames, gt_masks

def boxes_to_mask(boxes, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask[y1:y2, x1:x2] = 1
    return mask

def eval_sequence(seq, pipe):
    frames, gt_masks = load_sequence(seq)
    if not frames:
        print(f"  {seq}: no frames loaded, skip")
        return None
    
    j_scores, f_scores = [], []
    for i, (frame, gt) in enumerate(zip(frames, gt_masks)):
        h, w = frame.shape[:2]
        res = pipe.process_frame(frame, i, [])
        
        # Combine all predictions into one mask (unknown + known boxes)
        all_boxes = []
        if len(res["unknown_boxes"]) > 0:
            all_boxes.extend(res["unknown_boxes"].tolist())
        if len(res["known_boxes"]) > 0:
            all_boxes.extend(res["known_boxes"].tolist())
        
        if all_boxes:
            pred_mask = boxes_to_mask(all_boxes, (h, w))
        else:
            pred_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Resize GT to match pred (in case of size mismatch)
        if gt.shape != pred_mask.shape:
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        
        j_scores.append(compute_j_score(pred_mask, gt))
        f_scores.append(compute_f_score(pred_mask, gt))
    
    return {
        "sequence": seq,
        "n_frames": len(frames),
        "J": round(float(np.mean(j_scores)), 4),
        "F": round(float(np.mean(f_scores)), 4),
        "JF": round(float(np.mean(j_scores + f_scores)), 4),
    }

# ---- Main ----
pipe = build_pipeline()
results = []
for seq in TARGET_SEQS:
    print(f"\nEvaluating {seq}...")
    r = eval_sequence(seq, pipe)
    if r:
        results.append(r)
        print(f"  J={r['J']:.4f}  F={r['F']:.4f}  JF={r['JF']:.4f}  ({r['n_frames']} frames)")

if results:
    mean_J = float(np.mean([r["J"] for r in results]))
    mean_F = float(np.mean([r["F"] for r in results]))
    mean_JF = float(np.mean([r["JF"] for r in results]))
    
    output = {
        "overall": {"J": round(mean_J, 4), "F": round(mean_F, 4), "JF": round(mean_JF, 4)},
        "n_sequences": len(results),
        "per_sequence": {r["sequence"]: r for r in results}
    }
    
    print(f"\nMean over {len(results)} sequences:")
    print(f"  J={mean_J:.4f}  F={mean_F:.4f}  JF={mean_JF:.4f}")
    
    os.makedirs("/path/to/ow_vis/results/segtrack_eval", exist_ok=True)
    out = "/path/to/ow_vis/results/segtrack_eval/results.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out}")
