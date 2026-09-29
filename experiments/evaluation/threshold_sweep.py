import os, sys, json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = "/path/to/ow_vis"
SAM2_PATH = os.path.join(PROJECT_ROOT, "sam2")
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SAM2_PATH)
sys.path.insert(0, SRC_PATH)

from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from pipeline import OWVISPipeline
from metrics import compute_id_consistency

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


def reset_bank(bank):
    bank.embeddings[:] = 0
    bank.instance_ids[:] = -1
    bank.observation_counts[:] = 0
    bank.last_seen_frame[:] = -1
    bank.is_active[:] = False
    bank._next_id = 0
    bank._current_frame = 0


# Sweep thresholds 0.10 to 0.50 in 0.05 steps (9 values)
thresholds = [round(0.10 + i * 0.05, 2) for i in range(9)]
print(f"Sweeping thresholds: {thresholds}\n")

sweep_results = []

for thresh in thresholds:
    print(f"--- Threshold: {thresh:.2f} ---", end=" ")

    # Fresh pipeline per threshold
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
        unknown_score_threshold=float(thresh),
        device=device,
    )

    results = pipeline.process_video(
        video_path_or_frames=frames,
        known_categories=KNOWN_CATEGORIES,
        run_sam2=False,
    )
    per_frame = results["per_frame_detections"]

    total_unknown = sum(len(d["unknown_boxes"]) for d in per_frame)
    all_ids = set()
    ids_per_frame = []
    for d in per_frame:
        all_ids.update(d["unknown_ids"])
        ids_per_frame.append(d["unknown_ids"])
    unique_ids = len(all_ids)
    consistency = compute_id_consistency(ids_per_frame)

    print(f"unknown_dets={total_unknown}, unique_ids={unique_ids}, id_consistency={consistency:.4f}")
    sweep_results.append({
        "threshold": float(thresh),
        "n_unknown_detections": int(total_unknown),
        "unique_instances": int(unique_ids),
        "id_consistency": round(float(consistency), 4),
    })

# Save JSON
out_json = os.path.join(PROJECT_ROOT, "results", "threshold_sweep.json")
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(sweep_results, f, indent=2)
print(f"\nSaved: {out_json}")

# Plot
thresh_vals = [r["threshold"] for r in sweep_results]
n_unknown = [r["n_unknown_detections"] for r in sweep_results]
n_unique = [r["unique_instances"] for r in sweep_results]

fig, ax1 = plt.subplots(figsize=(9, 5))
color1 = "tab:blue"
ax1.set_xlabel("Unknown Score Threshold", fontsize=12)
ax1.set_ylabel("N Unknown Detections", color=color1, fontsize=12)
ax1.plot(thresh_vals, n_unknown, "o-", color=color1, linewidth=2, markersize=7, label="N Unknown Detections")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_xticks(thresh_vals)

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Unique Instances", color=color2, fontsize=12)
ax2.plot(thresh_vals, n_unique, "s--", color=color2, linewidth=2, markersize=7, label="Unique Instances")
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Threshold Sweep: Unknown Score Threshold vs Detections / Instances", fontsize=13)
fig.tight_layout()

out_png = os.path.join(PROJECT_ROOT, "results", "threshold_sweep.png")
plt.savefig(out_png, dpi=150)
print(f"Saved plot: {out_png}")

# Print table
print("\nThreshold | N Detections | Unique IDs | ID Consistency")
print("-" * 55)
for r in sweep_results:
    print(f"  {r['threshold']:.2f}    |   {r['n_unknown_detections']:<10}   |   {r['unique_instances']:<8}   | {r['id_consistency']:.4f}")
