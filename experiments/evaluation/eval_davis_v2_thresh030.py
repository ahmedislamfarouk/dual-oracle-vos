"""
OW-VIS DAVIS Evaluation v2: Lower threshold=0.3 with SAM2 Base+.
Tests whether more liberal detection threshold improves J&F.
"""
import sys, os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
import torch, cv2, numpy as np, time
from PIL import Image

import os as _os
_os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
THRESHOLD  = 0.3

print(f"Loading OpenWorldDetector (threshold={THRESHOLD})...", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=THRESHOLD, objectness_threshold=0.05)
print("Loading SAM2 Base+ video predictor...", flush=True)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

val_seqs = sorted(os.listdir(f"{DAVIS_ROOT}/JPEGImages/480p"))

all_j, all_f = [], []
seq_results = {}
t0_total = time.time()

for seq_idx, seq in enumerate(val_seqs[:25]):
    t0_seq = time.time()
    frame_paths = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mask_paths  = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    if not frame_paths or not mask_paths:
        print(f"  [{seq_idx+1}] {seq}: missing frames or masks, skipping", flush=True)
        continue

    n = min(len(frame_paths), len(mask_paths), 60)
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]

    gt_ids = set()
    for m in gt_masks:
        gt_ids.update(np.unique(m[m>0]).tolist())
    if not gt_ids:
        print(f"  [{seq_idx+1}] {seq}: no GT objects, skipping", flush=True)
        continue

    frame0_result = det.detect(frames[0])
    all_boxes_frame0 = []
    known_boxes = frame0_result.get("known_boxes", [])
    unknown_boxes = frame0_result.get("unknown_boxes", [])
    if len(known_boxes) > 0:
        for i, b in enumerate(known_boxes):
            box_list = b.tolist() if hasattr(b, 'tolist') else list(b)
            all_boxes_frame0.append((box_list, f"known_{i}"))
    if len(unknown_boxes) > 0:
        for i, b in enumerate(unknown_boxes):
            box_list = b.tolist() if hasattr(b, 'tolist') else list(b)
            all_boxes_frame0.append((box_list, f"unknown_{i}"))

    if not all_boxes_frame0:
        print(f"  [{seq_idx+1}] {seq}: no frame-0 detections, skipping", flush=True)
        continue

    tmpdir = tempfile.mkdtemp(prefix="davis_sam2_t030_")
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

    pred_masks = {inst_id: [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8) for _ in range(n)]
                  for _, inst_id in all_boxes_frame0}

    sam2_ok = False
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for box_data, inst_id in all_boxes_frame0:
                x1, y1, x2, y2 = [float(v) for v in box_data[:4]]
                x1 = max(0, min(x1, TARGET_SIZE[0]-1))
                y1 = max(0, min(y1, TARGET_SIZE[1]-1))
                x2 = max(x1+1, min(x2, TARGET_SIZE[0]))
                y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
                box_arr = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                obj_id = inst_id_to_obj_id[inst_id]
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=obj_id, box=box_arr)

            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n:
                    break
                for obj_id, mask_logits in zip(obj_ids, masks):
                    inst_id = obj_id_map.get(int(obj_id))
                    if inst_id is None:
                        continue
                    m = (mask_logits[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
                        m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    pred_masks[inst_id][frame_idx] = m
            sam2.reset_state(state)
            sam2_ok = True
    except Exception as e:
        print(f"  SAM2 error on {seq}: {e}", flush=True)
        for box_data, inst_id in all_boxes_frame0:
            x1, y1, x2, y2 = [int(float(v)) for v in box_data[:4]]
            x1 = max(0, min(x1, TARGET_SIZE[0]-1))
            y1 = max(0, min(y1, TARGET_SIZE[1]-1))
            x2 = max(x1+1, min(x2, TARGET_SIZE[0]))
            y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
            box_mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8)
            box_mask[y1:y2, x1:x2] = 1
            for i in range(n):
                pred_masks[inst_id][i] = box_mask

    shutil.rmtree(tmpdir, ignore_errors=True)

    seq_j, seq_f = [], []
    for gt_id in gt_ids:
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks[:n]]
        best_j = best_f = 0.0
        for inst_id, pred_seq in pred_masks.items():
            mj = float(np.mean([compute_j_score(pred_seq[i], gt_seq[i]) for i in range(n)]))
            if mj > best_j:
                best_j = mj
                best_f = float(np.mean([compute_f_score(pred_seq[i], gt_seq[i]) for i in range(n)]))
        seq_j.append(best_j)
        seq_f.append(best_f)

    mj = float(np.mean(seq_j)) if seq_j else 0.0
    mf = float(np.mean(seq_f)) if seq_f else 0.0
    jf = (mj + mf) / 2
    elapsed = time.time() - t0_seq
    seq_results[seq] = {
        "J": round(mj, 4), "F": round(mf, 4), "JF": round(jf, 4),
        "n_prompt_instances": len(all_boxes_frame0),
        "n_known": len(known_boxes), "n_unknown": len(unknown_boxes),
        "sam2_propagated": sam2_ok, "elapsed_s": round(elapsed, 1)
    }
    all_j.extend(seq_j)
    all_f.extend(seq_f)
    print(f"  [{seq_idx+1}] {seq}: J={mj:.3f} F={mf:.3f} J&F={jf:.3f}  prompts={len(all_boxes_frame0)} (known={len(known_boxes)}, unk={len(unknown_boxes)})  t={elapsed:.1f}s", flush=True)

oj = float(np.mean(all_j)) if all_j else 0.0
of_ = float(np.mean(all_f)) if all_f else 0.0
ojf = (oj + of_) / 2
total_elapsed = time.time() - t0_total
print(f"\n=== DAVIS RESULTS (Base+, threshold=0.3) ===", flush=True)
print(f"J={oj:.4f}  F={of_:.4f}  J&F={ojf:.4f}", flush=True)
print(f"Total time: {total_elapsed/60:.1f} min", flush=True)

os.makedirs("/path/to/ow_vis/results/davis_thresh030", exist_ok=True)
result_data = {
    "model": "sam2.1_hiera_base_plus",
    "threshold": THRESHOLD,
    "overall": {"J": round(oj, 4), "F": round(of_, 4), "JF": round(ojf, 4)},
    "n_sequences_evaluated": len(seq_results),
    "total_elapsed_min": round(total_elapsed/60, 1),
    "per_seq": seq_results
}
with open("/path/to/ow_vis/results/davis_thresh030/results.json", "w") as f:
    json.dump(result_data, f, indent=2)
print("Saved results/davis_thresh030/results.json", flush=True)
