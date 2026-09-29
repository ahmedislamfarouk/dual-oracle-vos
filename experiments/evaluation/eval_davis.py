import sys, os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
from pipeline import OWVISPipeline
from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from metrics import compute_j_score, compute_f_score
import torch, cv2
import numpy as np
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"
TARGET_SIZE = (854, 480)  # DAVIS 480p

# Read validation sequences
val_file = os.path.join(DAVIS_ROOT, "ImageSets/2017/val.txt")
if os.path.exists(val_file):
    with open(val_file) as f:
        sequences = [l.strip() for l in f if l.strip()]
else:
    # Fall back to listing directories
    sequences = sorted(os.listdir(os.path.join(DAVIS_ROOT, "JPEGImages/480p")))
print("Evaluating %d sequences" % len(sequences))


def load_frames(seq):
    frame_paths = sorted(glob.glob(os.path.join(DAVIS_ROOT, "JPEGImages/480p", seq, "*.jpg")))
    frames = []
    for p in frame_paths:
        fr = cv2.imread(p)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(fr, TARGET_SIZE))
    return frames


def load_gt_masks(seq):
    mask_paths = sorted(glob.glob(os.path.join(DAVIS_ROOT, "Annotations/480p", seq, "*.png")))
    masks = []
    for p in mask_paths:
        m = np.array(Image.open(p))
        masks.append(cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST))
    return masks


all_j_scores = []
all_f_scores = []
sequence_results = {}

for seq_idx, seq in enumerate(sequences[:20]):  # eval first 20 sequences
    print("\n[%d/%d] %s" % (seq_idx + 1, min(20, len(sequences)), seq))

    frames = load_frames(seq)
    gt_masks = load_gt_masks(seq)
    if not frames or not gt_masks:
        print("  Skipping %s - no data" % seq)
        continue

    n_frames = min(len(frames), len(gt_masks))
    frames = frames[:n_frames]
    gt_masks = gt_masks[:n_frames]

    # Get unique GT instance IDs (excluding background=0)
    gt_instances = set()
    for m in gt_masks:
        gt_instances.update(np.unique(m[m > 0]).tolist())

    if not gt_instances:
        print("  No GT instances in %s" % seq)
        continue

    print("  %d frames, %d GT instances" % (n_frames, len(gt_instances)))

    # Run OW-VIS pipeline
    bank = UnknownInstanceMemoryBank()
    emb = CropEmbedder().to(DEVICE)
    try:
        ck = torch.load("/path/to/ow_vis/checkpoints/crop_embedder_best.pt", map_location=DEVICE)
        emb.load_state_dict(ck["model_state_dict"])
    except Exception:
        pass

    pipe = OWVISPipeline(CKPT, "sam2.1_hiera_base_plus", bank, emb,
                         unknown_score_threshold=0.5, use_owdetr=True, device=DEVICE)

    # Run with SAM2 propagation to get masks
    result = pipe.process_video(frames, known_categories=[], run_sam2=True)
    pred_masks_per_frame = result.get("masks_per_frame", {})
    per_frame_det = result.get("per_frame_detections", [])

    # Build per-instance prediction masks
    # pred_instance_masks: {instance_id: [frame_mask_0, frame_mask_1, ...]}
    pred_by_instance = {}
    for i in range(n_frames):
        det = per_frame_det[i] if i < len(per_frame_det) else {}
        frame_masks = pred_masks_per_frame.get(i, {})
        ids = det.get("unknown_ids", [])
        boxes = det.get("unknown_boxes", [])

        for uid, box in zip(ids, boxes):
            H, W = TARGET_SIZE[1], TARGET_SIZE[0]
            if uid in frame_masks and frame_masks[uid] is not None:
                mask = np.array(frame_masks[uid]).astype(np.uint8)
                if mask.shape != (H, W):
                    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            else:
                # Use bounding box as proxy mask if SAM2 mask unavailable
                mask = np.zeros((H, W), dtype=np.uint8)
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                mask[y1:y2, x1:x2] = 1

            if uid not in pred_by_instance:
                pred_by_instance[uid] = [np.zeros((H, W), dtype=np.uint8)] * n_frames
            pred_by_instance[uid][i] = mask

    print("  Predicted %d unknown instances" % len(pred_by_instance))

    # Match GT instances to predicted instances by best IoU
    seq_j, seq_f = [], []
    for gt_id in gt_instances:
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]

        best_j, best_f = 0.0, 0.0
        for pred_id, pred_seq in pred_by_instance.items():
            # Compute mean J score across frames
            frame_js = [compute_j_score(pred_seq[i], gt_seq[i]) for i in range(n_frames)]
            mean_j = float(np.mean(frame_js))
            if mean_j > best_j:
                best_j = mean_j
                frame_fs = [compute_f_score(pred_seq[i], gt_seq[i]) for i in range(n_frames)]
                best_f = float(np.mean(frame_fs))

        seq_j.append(best_j)
        seq_f.append(best_f)
        print("    GT_id=%s: J=%.3f F=%.3f" % (gt_id, best_j, best_f))

    mean_j = float(np.mean(seq_j)) if seq_j else 0.0
    mean_f = float(np.mean(seq_f)) if seq_f else 0.0
    jf_mean = (mean_j + mean_f) / 2
    sequence_results[seq] = {"J": round(mean_j, 4), "F": round(mean_f, 4), "JF": round(jf_mean, 4)}
    all_j_scores.extend(seq_j)
    all_f_scores.extend(seq_f)
    print("  Seq mean: J=%.3f F=%.3f J&F=%.3f" % (mean_j, mean_f, jf_mean))

# Overall results
overall_j = float(np.mean(all_j_scores)) if all_j_scores else 0.0
overall_f = float(np.mean(all_f_scores)) if all_f_scores else 0.0
overall_jf = (overall_j + overall_f) / 2

sep = "=" * 50
print("\n" + sep)
print("OVERALL DAVIS J&F:")
print("  J (Jaccard/IoU):     %.4f" % overall_j)
print("  F (F-measure):       %.4f" % overall_f)
print("  J&F mean:            %.4f" % overall_jf)
print("  Sequences evaluated: %d" % len(sequence_results))
print(sep)

os.makedirs("/path/to/ow_vis/results/davis_eval", exist_ok=True)
results = {
    "overall": {"J": round(overall_j, 4), "F": round(overall_f, 4), "JF": round(overall_jf, 4)},
    "n_sequences": len(sequence_results),
    "per_sequence": sequence_results,
}
with open("/path/to/ow_vis/results/davis_eval/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to results/davis_eval/results.json")
