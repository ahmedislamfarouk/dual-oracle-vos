"""
DAVIS 2017 J&F evaluation for OW-VIS pipeline.

Usage:
    python davis_eval.py [--davis-root /tmp/davis2017/DAVIS] [--n-seqs 5]
"""
import sys
import os
import json
import glob
import argparse
import time

sys.path.insert(0, '/path/to/ow_vis/sam2')
sys.path.insert(0, '/path/to/ow_vis/src')

import torch
import cv2
import numpy as np
from PIL import Image

from metrics import compute_j_score, compute_f_score
from pipeline import OWVISPipeline
from memory_bank import UnknownInstanceMemoryBank, CropEmbedder

DAVIS_ROOT_DEFAULT = '/tmp/davis2017/DAVIS'
RESIZE_WH = (854, 480)  # DAVIS 480p standard


# ─────────────────────── helpers ───────────────────────

def load_gt_masks(ann_dir: str):
    """Return list of (H,W) uint8 arrays; pixel value = instance_id (0=bg)."""
    masks = []
    for path in sorted(glob.glob(f'{ann_dir}/*.png')):
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            # Some DAVIS annotations are RGB palettes — convert
            mask = mask[:, :, 0]
        masks.append(mask)
    return masks


def read_val_sequences(davis_root: str):
    """Read the official val split list."""
    val_txt = os.path.join(davis_root, 'ImageSets', '2017', 'val.txt')
    if os.path.isfile(val_txt):
        with open(val_txt) as f:
            seqs = [l.strip() for l in f if l.strip()]
        return seqs
    # Fallback: discover from Annotations dir
    ann_480 = os.path.join(davis_root, 'Annotations', '480p')
    if os.path.isdir(ann_480):
        return sorted(os.listdir(ann_480))
    return []


# ─────────────────────── per-sequence eval ───────────────────────

def evaluate_sequence(seq_name: str, davis_root: str, pipeline: OWVISPipeline,
                      known_categories: list, resize_wh=(854, 480)):
    """Run OW-VIS on one DAVIS sequence and return J, F per GT instance."""
    frame_dir = os.path.join(davis_root, 'JPEGImages', '480p', seq_name)
    ann_dir   = os.path.join(davis_root, 'Annotations', '480p', seq_name)

    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    gt_masks    = load_gt_masks(ann_dir)

    if not frame_paths or not gt_masks:
        print(f'  [SKIP] {seq_name}: missing frames or annotations')
        return None

    n = min(len(frame_paths), len(gt_masks))
    frame_paths = frame_paths[:n]
    gt_masks    = gt_masks[:n]

    # Load & resize frames
    frames = []
    for p in frame_paths:
        f = cv2.imread(p)
        if f is None:
            continue
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if f.shape[1] != resize_wh[0] or f.shape[0] != resize_wh[1]:
            f = cv2.resize(f, resize_wh)
        frames.append(f)

    # Also resize GT masks to match
    gt_masks_r = []
    for m in gt_masks:
        mr = cv2.resize(m, resize_wh, interpolation=cv2.INTER_NEAREST)
        gt_masks_r.append(mr)

    gt_instance_ids = set()
    for m in gt_masks_r:
        gt_instance_ids.update(np.unique(m[m > 0]).tolist())

    print(f'  {seq_name}: {len(frames)} frames, {len(gt_instance_ids)} GT instances')

    # Reset pipeline memory bank for this sequence
    new_bank = UnknownInstanceMemoryBank()
    new_emb  = CropEmbedder().to(pipeline.device)
    pipeline.memory_bank = new_bank.to(pipeline.device)
    pipeline.crop_embedder = new_emb

    # Run pipeline with SAM2 propagation
    t0 = time.time()
    result = pipeline.process_video(
        frames,
        known_categories=known_categories,
        run_sam2=True,
        sam2_prompt_every_n_frames=5,
    )
    elapsed = time.time() - t0
    print(f'    pipeline done in {elapsed:.1f}s')

    # Collect masks: prefer SAM2 masks, fall back to box masks
    # masks_per_frame: {frame_idx -> {obj_id -> binary HxW mask}}
    masks_per_frame = result.get('masks_per_frame', {})
    per_frame_dets  = result['per_frame_detections']

    # Build per-instance per-frame binary mask arrays
    # inst_masks[uid][frame_idx] = binary mask or None
    inst_masks = {}  # {uid: {frame_idx: ndarray}}

    for fi in range(len(frames)):
        H, W = frames[fi].shape[:2]

        # SAM2 masks take priority
        if fi in masks_per_frame:
            for obj_id, mask in masks_per_frame[fi].items():
                if mask is None:
                    continue
                uid = int(obj_id)
                if uid not in inst_masks:
                    inst_masks[uid] = {}
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                inst_masks[uid][fi] = mask.astype(np.uint8)

        # Fall back to box-derived masks for detections not in SAM2 output
        det = per_frame_dets[fi]
        for box, uid in zip(det.get('unknown_boxes', []), det.get('unknown_ids', [])):
            if uid in inst_masks and fi in inst_masks[uid]:
                continue  # already have SAM2 mask
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            bmask = np.zeros((H, W), dtype=np.uint8)
            bmask[y1:y2, x1:x2] = 1
            if uid not in inst_masks:
                inst_masks[uid] = {}
            inst_masks[uid][fi] = bmask

    n_pred = len(inst_masks)
    print(f'    predicted instances: {n_pred}')

    if n_pred == 0:
        print(f'    [WARN] no predictions for {seq_name}')
        return {
            'sequence': seq_name,
            'n_frames': len(frames),
            'n_gt_instances': len(gt_instance_ids),
            'n_pred_instances': 0,
            'mean_J': 0.0,
            'mean_F': 0.0,
            'mean_JF': 0.0,
            'elapsed_s': round(elapsed, 2),
        }

    # Match each GT instance to best predicted instance
    j_scores, f_scores = [], []

    for gt_id in gt_instance_ids:
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks_r]
        best_j = 0.0
        best_f = 0.0

        for uid, frame_dict in inst_masks.items():
            frame_js, frame_fs = [], []
            for fi, gt_m in enumerate(gt_seq):
                pred_m = frame_dict.get(fi)
                if pred_m is not None:
                    frame_js.append(compute_j_score(pred_m, gt_m))
                    frame_fs.append(compute_f_score(pred_m, gt_m))
            if frame_js:
                mj = float(np.mean(frame_js))
                mf = float(np.mean(frame_fs))
                if mj > best_j:
                    best_j = mj
                    best_f = mf

        j_scores.append(best_j)
        f_scores.append(best_f)

    mean_j  = float(np.mean(j_scores)) if j_scores else 0.0
    mean_f  = float(np.mean(f_scores)) if f_scores else 0.0
    mean_jf = (mean_j + mean_f) / 2.0

    print(f'    J={mean_j:.4f}  F={mean_f:.4f}  J&F={mean_jf:.4f}')
    return {
        'sequence': seq_name,
        'n_frames': len(frames),
        'n_gt_instances': len(gt_instance_ids),
        'n_pred_instances': n_pred,
        'mean_J': round(mean_j, 4),
        'mean_F': round(mean_f, 4),
        'mean_JF': round(mean_jf, 4),
        'elapsed_s': round(elapsed, 2),
    }


# ─────────────────────── main ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis-root', default=DAVIS_ROOT_DEFAULT)
    parser.add_argument('--n-seqs', type=int, default=5,
                        help='Number of sequences to evaluate (0 = all)')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', default='/path/to/ow_vis/results/davis_eval')
    args = parser.parse_args()

    davis_root = args.davis_root
    if not os.path.isdir(davis_root):
        print(f'[ERROR] DAVIS root not found: {davis_root}')
        print('  -> Run: python davis_eval.py --davis-root /path/to/DAVIS')
        sys.exit(1)

    seqs = read_val_sequences(davis_root)
    if not seqs:
        print('[ERROR] No sequences found in DAVIS root')
        sys.exit(1)

    if args.n_seqs > 0:
        seqs = seqs[:args.n_seqs]

    print(f'Evaluating {len(seqs)} sequences from {davis_root}')

    # Load pipeline (once)
    CKPT   = '/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt'
    CONFIG = 'sam2.1_hiera_base_plus'
    bank   = UnknownInstanceMemoryBank()
    emb    = CropEmbedder().to(args.device)

    print('Loading OW-VIS pipeline...')
    pipeline = OWVISPipeline(
        sam2_checkpoint_path=CKPT,
        sam2_config=CONFIG,
        memory_bank=bank,
        crop_embedder=emb,
        unknown_score_threshold=args.threshold,
        device=torch.device(args.device),
    )
    print('Pipeline loaded.')

    known_categories = [
        'person', 'car', 'bicycle', 'motorcycle', 'truck', 'bus',
        'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'bird', 'boat', 'surfboard', 'snowboard',
        'skateboard', 'ball', 'frisbee',
    ]

    results = []
    for seq in seqs:
        print(f'\n--- Sequence: {seq} ---')
        try:
            r = evaluate_sequence(seq, davis_root, pipeline, known_categories)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f'  [ERROR] {seq}: {e}')
            import traceback
            traceback.print_exc()

    if results:
        all_j  = [r['mean_J']  for r in results]
        all_f  = [r['mean_F']  for r in results]
        all_jf = [r['mean_JF'] for r in results]
        summary = {
            'dataset': 'DAVIS-2017-val',
            'n_sequences': len(results),
            'threshold': args.threshold,
            'mean_J_over_seqs':  round(float(np.mean(all_j)), 4),
            'mean_F_over_seqs':  round(float(np.mean(all_f)), 4),
            'mean_JF_over_seqs': round(float(np.mean(all_jf)), 4),
            'per_sequence': results,
        }
    else:
        summary = {'dataset': 'DAVIS-2017-val', 'n_sequences': 0, 'error': 'no results'}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print(f'OVERALL  J={summary.get("mean_J_over_seqs", "N/A")}'
          f'  F={summary.get("mean_F_over_seqs", "N/A")}'
          f'  J&F={summary.get("mean_JF_over_seqs", "N/A")}')


if __name__ == '__main__':
    main()
