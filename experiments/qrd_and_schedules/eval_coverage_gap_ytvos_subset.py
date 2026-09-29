"""
OW-VIS YouTubeVOS 2019 val subset: ALL DETR boxes + Coverage-Gap Re-Detection.
Subset = 150 videos stratified across n_gt=2 (75) and n_gt=3 (75), the regime
identified in Section 5.2.1 as having the lowest baseline J&F (68.2%/57.4%) --
the target population for testing whether coverage-gap re-detection helps.
Same tier1 (obj>0.05) + gap-fill (0.02<=obj<0.05, IoU<0.3 vs tier1) logic as
eval_coverage_gap_val30.py.
GPU: 2
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
import torch, cv2, numpy as np
from PIL import Image

YTVOS_ROOT = "/path/to/ytvos2019/valid"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
GAP_FILL_THRESHOLD = 0.02
TIER1_THRESHOLD    = 0.05
IOU_OVERLAP_MAX     = 0.3
OUT_DIR    = "/path/to/ow_vis/results/coverage_gap_ytvos_subset"
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(YTVOS_ROOT, "meta.json")) as f:
    meta = json.load(f)
with open("/path/to/ow_vis/ytvos_subset_videos.json") as f:
    video_ids = json.load(f)
video_ids = sorted(video_ids)

print(f"YouTubeVOS 2019 val SUBSET: {len(video_ids)} videos (n_gt=2,3 stratified), coverage-gap re-detection", flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=GAP_FILL_THRESHOLD)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def eval_video(vid_id):
    img_dir  = os.path.join(YTVOS_ROOT, "JPEGImages", vid_id)
    ann_dir  = os.path.join(YTVOS_ROOT, "Annotations", vid_id)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths: return None
    ann_paths = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    if not ann_paths: return None
    ann_names = {os.path.splitext(os.path.basename(p))[0]: p for p in ann_paths}

    frames, frame_names = [], []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None: continue
        img_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        frames.append(img_rgb)
        frame_names.append(os.path.splitext(os.path.basename(p))[0])
    if not frames: return None
    h, w = frames[0].shape[:2]
    n = len(frames)

    gt_ids = set()
    gt_masks_by_frame = {}
    for fname, ann_path in ann_names.items():
        mask = cv2.resize(np.array(Image.open(ann_path)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        gt_masks_by_frame[fname] = mask
        gt_ids.update(int(x) for x in np.unique(mask[mask > 0]))
    if not gt_ids: return None

    res = det.detect(frames[0])
    known_boxes   = list(res.get("known_boxes", []))
    unknown_boxes = list(res.get("unknown_boxes", []))
    unknown_scores = list(res.get("unknown_scores", []))

    tier1 = list(known_boxes) + [b for b, s in zip(unknown_boxes, unknown_scores) if s >= TIER1_THRESHOLD]
    gap_candidates = [b for b, s in zip(unknown_boxes, unknown_scores) if GAP_FILL_THRESHOLD <= s < TIER1_THRESHOLD]
    gap_fill = []
    for cand in gap_candidates:
        max_iou = max([box_iou(cand, t1) for t1 in tier1], default=0.0)
        if max_iou < IOU_OVERLAP_MAX:
            gap_fill.append(cand)
    all_boxes = tier1 + gap_fill
    n_boxes = len(all_boxes)
    print(f"  {vid_id}: {len(tier1)} tier1 + {len(gap_fill)} gap-fill (of {len(gap_candidates)} cand) "
          f"= {n_boxes} boxes, {len(gt_ids)} GT objects", flush=True)
    if n_boxes == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0, "n_gt": len(gt_ids)}

    tmpdir = tempfile.mkdtemp(prefix=f"covgap_yt_{vid_id}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    fname_to_idx = {name: i for i, name in enumerate(frame_names)}

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
        print(f"  SAM2 error on {vid_id}: {e}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    j_scores, f_scores = [], []
    for gt_id in gt_ids:
        obj_frames = []
        for fname, mask in gt_masks_by_frame.items():
            if gt_id in np.unique(mask) and fname in fname_to_idx:
                fi = fname_to_idx[fname]
                obj_frames.append((fi, (mask == gt_id).astype(np.uint8)))
        if not obj_frames: continue
        best_j, best_f = 0.0, 0.0
        for tid, tmasks in pred_by_track.items():
            mj = float(np.mean([compute_j_score(tmasks[fi], gt_bin) for fi, gt_bin in obj_frames]))
            if mj > best_j:
                best_j = mj
                best_f = float(np.mean([compute_f_score(tmasks[fi], gt_bin) for fi, gt_bin in obj_frames]))
        j_scores.append(best_j); f_scores.append(best_f)

    if not j_scores: return None
    mj, mf = float(np.mean(j_scores)), float(np.mean(f_scores))
    return {"J": round(mj, 4), "F": round(mf, 4), "JF": round((mj+mf)/2, 4),
            "n_boxes": n_boxes, "n_tier1": len(tier1), "n_gapfill": len(gap_fill), "n_gt": len(gt_ids)}


results = {}
t0 = time.time()
for vi, vid_id in enumerate(video_ids):
    print(f"\n[{vi+1}/{len(video_ids)}] {vid_id}", flush=True)
    r = eval_video(vid_id)
    if r:
        results[vid_id] = r
        print(f"  J={r['J']:.4f} F={r['F']:.4f} JF={r['JF']:.4f}", flush=True)
    if results:
        Js=[v["J"] for v in results.values()]; Fs=[v["F"] for v in results.values()]
        out={"per_video":results,"mean_J":round(float(np.mean(Js)),4),
             "mean_F":round(float(np.mean(Fs)),4),
             "mean_JF":round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
             "n_videos":len(results)}
        with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(out, fp, indent=2)

total = time.time()-t0
Js=[v["J"] for v in results.values()]; Fs=[v["F"] for v in results.values()]
final={"per_video":results,"mean_J":round(float(np.mean(Js)),4),"mean_F":round(float(np.mean(Fs)),4),
       "mean_JF":round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
       "n_videos":len(results),"total_s":round(total,1),
       "dataset":"YouTubeVOS-2019-val-subset-ngt23", "tier1_threshold": TIER1_THRESHOLD,
       "gap_fill_threshold": GAP_FILL_THRESHOLD, "iou_overlap_max": IOU_OVERLAP_MAX}
with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(final, fp, indent=2)
print(f"\n=== COVERAGE-GAP YTVOS SUBSET FINAL: J={final['mean_J']:.4f} F={final['mean_F']:.4f} JF={final['mean_JF']:.4f} ({len(results)}/{len(video_ids)} videos) ===", flush=True)
