#!/bin/bash
# OW-VIS Baseline Pipeline Run
# Runs the full OW-VIS pipeline on the sample video with a fixed set of known categories.
# Results saved to $HOME/ow_vis/results/baseline_run/
# Log saved to $HOME/ow_vis/logs/baseline_run.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$HOME/ow_vis"
SAM2_PATH="${PROJECT_ROOT}/sam2"
SRC_PATH="${PROJECT_ROOT}/src"
RESULTS_DIR="${PROJECT_ROOT}/results/baseline_run"
LOG_FILE="${PROJECT_ROOT}/logs/baseline_run.log"

# Use large if valid, else fall back to base_plus
if python3 -c "import torch; torch.load('${PROJECT_ROOT}/checkpoints/sam2.1_hiera_large.pt', map_location='cpu')" 2>/dev/null; then
    SAM2_CHECKPOINT="${PROJECT_ROOT}/checkpoints/sam2.1_hiera_large.pt"
    SAM2_CONFIG="sam2.1_hiera_large"
else
    echo "[WARN] sam2.1_hiera_large.pt is corrupt or missing; falling back to base_plus" >&2
    SAM2_CHECKPOINT="${PROJECT_ROOT}/checkpoints/sam2.1_hiera_base_plus.pt"
    SAM2_CONFIG="sam2.1_hiera_base_plus"
fi
SAMPLE_VIDEO="${PROJECT_ROOT}/datasets/youtube_vis/sample_video.mp4"

KNOWN_CATEGORIES="person car bicycle dog cat"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================" | tee -a "${LOG_FILE}"
echo "OW-VIS Baseline Run — $(date)" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "Results dir : ${RESULTS_DIR}" | tee -a "${LOG_FILE}"
echo "Sample video: ${SAMPLE_VIDEO}" | tee -a "${LOG_FILE}"
echo "Known cats  : ${KNOWN_CATEGORIES}" | tee -a "${LOG_FILE}"
echo "SAM2 ckpt   : ${SAM2_CHECKPOINT}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Check dependencies
if [ ! -f "${SAMPLE_VIDEO}" ]; then
    echo "ERROR: sample video not found at ${SAMPLE_VIDEO}" | tee -a "${LOG_FILE}"
    echo "Download it with:" | tee -a "${LOG_FILE}"
    echo "  wget https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.mp4 -O ${SAMPLE_VIDEO}" | tee -a "${LOG_FILE}"
    exit 1
fi

if [ ! -f "${SAM2_CHECKPOINT}" ]; then
    echo "ERROR: SAM2 checkpoint not found at ${SAM2_CHECKPOINT}" | tee -a "${LOG_FILE}"
    exit 1
fi

python3 - <<PYEOF 2>&1 | tee -a "${LOG_FILE}"
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, "${SAM2_PATH}")
sys.path.insert(0, "${SRC_PATH}")

from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from pipeline import OWVISPipeline

RESULTS_DIR = "${RESULTS_DIR}"
SAMPLE_VIDEO = "${SAMPLE_VIDEO}"
KNOWN_CATEGORIES = "${KNOWN_CATEGORIES}".split()
SAM2_CHECKPOINT = "${SAM2_CHECKPOINT}"
SAM2_CONFIG = "${SAM2_CONFIG}"

print(f"Known categories: {KNOWN_CATEGORIES}")
print(f"Device: ", end="")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Build pipeline components
memory_bank = UnknownInstanceMemoryBank(
    embedding_dim=256,
    max_instances=256,
    ema_momentum=0.9,
    match_threshold=0.65,
    prune_after_frames=60,
    promote_after_obs=8,
)
crop_embedder = CropEmbedder(backbone_dim=768, embedding_dim=256)

pipeline = OWVISPipeline(
    sam2_checkpoint_path=SAM2_CHECKPOINT,
    sam2_config=SAM2_CONFIG,
    memory_bank=memory_bank,
    crop_embedder=crop_embedder,
    unknown_score_threshold=0.25,
    device=device,
)
print("Pipeline initialized OK")

# Load video frames (sample up to 30 frames for baseline)
import cv2
cap = cv2.VideoCapture(SAMPLE_VIDEO)
frames = []
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, total_frames // 30)  # sample up to 30 frames
frame_idx = 0
while len(frames) < 30:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % step == 0:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_idx += 1
cap.release()
print(f"Loaded {len(frames)} frames (sampled from {total_frames} total @ {fps:.1f}fps)")

# Run pipeline (no SAM2 propagation for baseline speed — set run_sam2=False)
t0 = time.time()
results = pipeline.process_video(
    video_path_or_frames=frames,
    known_categories=KNOWN_CATEGORIES,
    run_sam2=False,   # detection-only baseline; SAM2 needs contiguous frames
)
elapsed = time.time() - t0
print(f"Pipeline finished in {elapsed:.1f}s")

# Summarise results
total_known = sum(len(d["known_boxes"]) for d in results["per_frame_detections"])
total_unknown = sum(len(d["unknown_boxes"]) for d in results["per_frame_detections"])
n_unique_instances = len(results["unknown_instances"])
promoted = [iid for iid, m in results["unknown_instances"].items() if m["promoted"]]

print(f"Total known detections:   {total_known}")
print(f"Total unknown detections: {total_unknown}")
print(f"Unique unknown instances: {n_unique_instances}")
print(f"Promoted instances:       {len(promoted)}")

# Save results as JSON
summary = {
    "video": SAMPLE_VIDEO,
    "n_frames": len(frames),
    "known_categories": KNOWN_CATEGORIES,
    "total_known_detections": total_known,
    "total_unknown_detections": total_unknown,
    "unique_unknown_instances": n_unique_instances,
    "promoted_instances": promoted,
    "elapsed_seconds": round(elapsed, 2),
    "per_frame": [
        {
            "frame_idx": i,
            "n_known": len(d["known_boxes"]),
            "n_unknown": len(d["unknown_boxes"]),
            "unknown_ids": d["unknown_ids"],
            "known_labels": d["known_labels"],
        }
        for i, d in enumerate(results["per_frame_detections"])
    ],
}

out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Results saved to {out_path}")

PYEOF

echo "" | tee -a "${LOG_FILE}"
echo "Baseline run completed — $(date)" | tee -a "${LOG_FILE}"
