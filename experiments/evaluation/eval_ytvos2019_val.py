"""
OW-VIS YouTubeVOS 2019 val evaluation.
ALL DETR boxes (known + unknown, tau=0.50), SAM2 Large — same candidate pool as DAVIS/SegTrack headline (83.37%).
Expected structure:
  /path/to/ytvos2019/valid/
    JPEGImages/  (video dirs, each containing frame_*.jpg)
    Annotations/ (video dirs, each containing 00000.png etc. for annotated frames)
    meta.json    (contains object categories and frame info)
GPU: 1
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
TAU        = 0.5
OUT_DIR    = "/path/to/ow_vis/results/ytvos2019_val"
os.makedirs(OUT_DIR, exist_ok=True)

# Load metadata
meta_path = os.path.join(YTVOS_ROOT, "meta.json")
with open(meta_path) as f:
    meta = json.load(f)

video_ids = sorted(meta["videos"].keys())
print(f"YouTubeVOS 2019 val: {len(video_ids)} videos, all boxes (tau={TAU})", flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=TAU, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def eval_video(vid_id):
    """Evaluate a single video. Annotated frames only contain GT objects."""
    img_dir  = os.path.join(YTVOS_ROOT, "JPEGImages", vid_id)
    ann_dir  = os.path.join(YTVOS_ROOT, "Annotations", vid_id)

    # Get all frames (sorted by name)
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        return None

    # Only annotated frames have GT masks
    ann_paths = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    if not ann_paths:
        return None

    # Build frame name -> GT mask mapping
    ann_names = {os.path.splitext(os.path.basename(p))[0]: p for p in ann_paths}

    # Load all video frames at target resolution
    frames = []
    frame_names = []
    for p in img_paths:
        img = cv2.imread(p)
        if img is None: continue
        img_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), TARGET_SIZE)
        frames.append(img_rgb)
        frame_names.append(os.path.splitext(os.path.basename(p))[0])

    if not frames: return None
    h, w = frames[0].shape[:2]
    n = len(frames)

    # Collect GT object IDs from annotated frames
    gt_ids = set()
    gt_masks_by_frame = {}  # frame_name -> mask array
    for fname, ann_path in ann_names.items():
        mask = cv2.resize(np.array(Image.open(ann_path)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        gt_masks_by_frame[fname] = mask
        gt_ids.update(int(x) for x in np.unique(mask[mask > 0]))

    if not gt_ids: return None

    # ALL boxes: known + unknown (same candidate pool as baseline 83.37% on DAVIS/SegTrack)
    res = det.detect(frames[0])
    known_boxes   = list(res.get("known_boxes", []))
    unknown_boxes = list(res.get("unknown_boxes", []))
    all_boxes = known_boxes + unknown_boxes
    n_boxes = len(all_boxes)
    print(f"  {vid_id}: {len(known_boxes)} known + {len(unknown_boxes)} unknown = {n_boxes} boxes, "
          f"{len(gt_ids)} GT objects, {len(ann_names)} annotated frames", flush=True)

    if n_boxes == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0, "n_gt": len(gt_ids)}

    # Write frames to temp dir
    tmpdir = tempfile.mkdtemp(prefix=f"ytvos_{vid_id}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    # Build frame index mapping
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

    # Evaluate ONLY on annotated frames (YouTubeVOS protocol); Hungarian-style best-match per GT object
    j_scores, f_scores = [], []
    for gt_id in gt_ids:
        # Collect annotated frames where this GT object appears
        obj_frames = []
        for fname, mask in gt_masks_by_frame.items():
            if gt_id in np.unique(mask) and fname in fname_to_idx:
                fi = fname_to_idx[fname]
                obj_frames.append((fi, (mask == gt_id).astype(np.uint8)))
        if not obj_frames: continue

        # Best-match track for this GT object
        best_j, best_f = 0.0, 0.0
        for tid, tmasks in pred_by_track.items():
            mj = float(np.mean([compute_j_score(tmasks[fi], gt_bin) for fi, gt_bin in obj_frames]))
            if mj > best_j:
                best_j = mj
                best_f = float(np.mean([compute_f_score(tmasks[fi], gt_bin) for fi, gt_bin in obj_frames]))
        j_scores.append(best_j)
        f_scores.append(best_f)

    if not j_scores: return None
    mj, mf = float(np.mean(j_scores)), float(np.mean(f_scores))
    return {"J": round(mj, 4), "F": round(mf, 4), "JF": round((mj+mf)/2, 4),
            "n_boxes": n_boxes, "n_known": len(known_boxes), "n_unknown": len(unknown_boxes), "n_gt": len(gt_ids)}


results = {}
t0 = time.time()
for vi, vid_id in enumerate(video_ids):
    print(f"\n[{vi+1}/{len(video_ids)}] {vid_id}", flush=True)
    r = eval_video(vid_id)
    if r:
        results[vid_id] = r
        print(f"  J={r['J']:.4f} F={r['F']:.4f} JF={r['JF']:.4f}", flush=True)
    # Save incrementally
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
       "n_videos":len(results),"total_s":round(total,1),"tau":TAU,
       "dataset":"YouTubeVOS-2019-val"}
with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(final, fp, indent=2)
print(f"\n=== YTVOS FINAL: J={final['mean_J']:.4f} F={final['mean_F']:.4f} JF={final['mean_JF']:.4f} ({len(results)}/{len(video_ids)} videos) ===", flush=True)
