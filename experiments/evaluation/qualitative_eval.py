import os, sys, json
import numpy as np
import torch
import cv2

PROJECT_ROOT = "/path/to/ow_vis"
SAM2_PATH = os.path.join(PROJECT_ROOT, "sam2")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SAM2_PATH)
sys.path.insert(0, SRC_PATH)

from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from pipeline import OWVISPipeline
from metrics import compute_j_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SAMPLE_VIDEO = os.path.join(PROJECT_ROOT, "datasets/youtube_vis/sample_video.mp4")
KNOWN_CATEGORIES = ["person", "car", "bicycle", "dog", "cat"]
SAM2_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "sam2.1_hiera_base_plus.pt")
SAM2_CFG = "sam2.1_hiera_base_plus"
N_FRAMES = 30

# Load frames
cap = cv2.VideoCapture(SAMPLE_VIDEO)
frames = []
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, total_frames // N_FRAMES)
fi = 0
while len(frames) < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    if fi % step == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fi += 1
cap.release()
frames = frames[:N_FRAMES]
print(f"Loaded {len(frames)} frames (from {total_frames} total)")

# Load crop embedder
crop_embedder = CropEmbedder(backbone_dim=768, embedding_dim=256)
ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "crop_embedder_best.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    crop_embedder.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded crop embedder: iteration={ckpt.get('iteration','?')}, loss={ckpt.get('loss',0):.4f}")
crop_embedder.to(device)

# Build pipeline with SAM2 enabled
memory_bank = UnknownInstanceMemoryBank(
    embedding_dim=256,
    max_instances=256,
    ema_momentum=0.9,
    match_threshold=0.65,
    prune_after_frames=60,
    promote_after_obs=8,
)
pipeline = OWVISPipeline(
    sam2_checkpoint_path=SAM2_CKPT,
    sam2_config=SAM2_CFG,
    memory_bank=memory_bank,
    crop_embedder=crop_embedder,
    unknown_score_threshold=0.25,
    device=device,
)
print("Pipeline initialized. Running with run_sam2=True...\n")

# Run full pipeline with SAM2
results = pipeline.process_video(
    video_path_or_frames=frames,
    known_categories=KNOWN_CATEGORIES,
    run_sam2=True,
    sam2_prompt_every_n_frames=10,
)

per_frame_dets = results["per_frame_detections"]
masks_per_frame = results["masks_per_frame"]  # frame_idx -> {obj_id: H x W bool mask}

print(f"Frames with masks: {len(masks_per_frame)}")

# ---- 1. Detection density ----
unknowns_per_frame = [len(d["unknown_boxes"]) for d in per_frame_dets]
avg_unknowns_per_frame = float(np.mean(unknowns_per_frame))
print(f"Avg unknowns per frame: {avg_unknowns_per_frame:.3f}")

# ---- 2. Instance lifetime ----
# Track first and last frame each unknown instance was seen
instance_first_frame = {}
instance_last_frame = {}
for frame_idx, det in enumerate(per_frame_dets):
    for inst_id in det["unknown_ids"]:
        if inst_id not in instance_first_frame:
            instance_first_frame[inst_id] = frame_idx
        instance_last_frame[inst_id] = frame_idx

lifetimes = []
for inst_id in instance_first_frame:
    lifetime = instance_last_frame[inst_id] - instance_first_frame[inst_id] + 1
    lifetimes.append(lifetime)

if lifetimes:
    mean_lifetime = float(np.mean(lifetimes))
    min_lifetime = int(np.min(lifetimes))
    max_lifetime = int(np.max(lifetimes))
else:
    mean_lifetime = 0.0
    min_lifetime = 0
    max_lifetime = 0

n_unique_instances = len(lifetimes)
print(f"Unique unknown instances: {n_unique_instances}")
print(f"Instance lifetime — mean: {mean_lifetime:.2f}, min: {min_lifetime}, max: {max_lifetime}")

# ---- 3. Mask IoU consistency ----
# For each unknown instance tracked in SAM2 masks, compute pairwise IoU of masks
# between consecutive frames. Higher = more stable tracking.
mask_iou_scores = []
sorted_frame_idxs = sorted(masks_per_frame.keys())

# Collect per-instance mask sequences
instance_masks_by_frame = {}  # inst_id -> {frame_idx: mask}
for frame_idx, frame_masks in masks_per_frame.items():
    for obj_id, mask in frame_masks.items():
        if obj_id < 0:
            continue  # skip known objects (negative IDs)
        if obj_id not in instance_masks_by_frame:
            instance_masks_by_frame[obj_id] = {}
        instance_masks_by_frame[obj_id][frame_idx] = mask

# For each instance, compute consecutive-frame mask IoU
for inst_id, frame_mask_dict in instance_masks_by_frame.items():
    frame_idxs = sorted(frame_mask_dict.keys())
    if len(frame_idxs) < 2:
        continue
    for i in range(len(frame_idxs) - 1):
        fi = frame_idxs[i]
        fj = frame_idxs[i + 1]
        # Only compare consecutive (adjacent) frames
        if fj - fi > 1:
            continue
        mask_a = frame_mask_dict[fi].astype(bool)
        mask_b = frame_mask_dict[fj].astype(bool)
        iou = compute_j_score(mask_a, mask_b)
        mask_iou_scores.append(iou)

if mask_iou_scores:
    avg_mask_iou = float(np.mean(mask_iou_scores))
    n_mask_pairs = len(mask_iou_scores)
else:
    # SAM2 may not provide frame-adjacent masks if prompted only every 10 frames.
    # Compute IoU across all available frame pairs for the instance instead.
    print("No adjacent-frame mask pairs found. Computing across all available frame pairs...")
    for inst_id, frame_mask_dict in instance_masks_by_frame.items():
        frame_idxs = sorted(frame_mask_dict.keys())
        if len(frame_idxs) < 2:
            continue
        for i in range(len(frame_idxs) - 1):
            fi = frame_idxs[i]
            fj = frame_idxs[i + 1]
            mask_a = frame_mask_dict[fi].astype(bool)
            mask_b = frame_mask_dict[fj].astype(bool)
            iou = compute_j_score(mask_a, mask_b)
            mask_iou_scores.append(iou)

    if mask_iou_scores:
        avg_mask_iou = float(np.mean(mask_iou_scores))
        n_mask_pairs = len(mask_iou_scores)
    else:
        avg_mask_iou = 0.0
        n_mask_pairs = 0

print(f"Mask IoU consistency: {avg_mask_iou:.4f} (over {n_mask_pairs} frame pairs)")

# Summary
qualitative_metrics = {
    "n_frames": len(frames),
    "n_unique_unknown_instances": n_unique_instances,
    "mask_iou_consistency": round(avg_mask_iou, 4),
    "n_mask_iou_pairs_evaluated": n_mask_pairs,
    "instance_lifetime_mean_frames": round(mean_lifetime, 3),
    "instance_lifetime_min_frames": min_lifetime,
    "instance_lifetime_max_frames": max_lifetime,
    "avg_unknowns_per_frame": round(avg_unknowns_per_frame, 3),
    "detection_density": round(avg_unknowns_per_frame, 3),
    "per_instance_lifetimes": {
        str(inst_id): {
            "first_frame": instance_first_frame[inst_id],
            "last_frame": instance_last_frame[inst_id],
            "lifetime_frames": instance_last_frame[inst_id] - instance_first_frame[inst_id] + 1,
        }
        for inst_id in instance_first_frame
    },
}

out_json = os.path.join(PROJECT_ROOT, "results", "qualitative_metrics.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(qualitative_metrics, f, indent=2)
print(f"\nSaved: {out_json}")

print("\n" + "="*55)
print("QUALITATIVE METRICS SUMMARY")
print("="*55)
print(f"Mask IoU Consistency     : {avg_mask_iou:.4f}")
print(f"Mean Instance Lifetime   : {mean_lifetime:.2f} frames")
print(f"Min Instance Lifetime    : {min_lifetime} frames")
print(f"Max Instance Lifetime    : {max_lifetime} frames")
print(f"Avg Unknowns per Frame   : {avg_unknowns_per_frame:.3f}")
print(f"Unique Unknown Instances : {n_unique_instances}")
print("="*55)
