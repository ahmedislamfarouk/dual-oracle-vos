"""
OW-VIS DAVIS 2017 Val (official 30 seqs): SAM2 Automatic Mask Generator as proposals.
Instead of DETR boxes, uses SAM2's own dense mask proposals (much better init quality).
GPU: 2
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import torch, cv2, numpy as np
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
OUT_DIR    = "/path/to/ow_vis/results/sam2_amg_val30"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
print(f"Evaluating {len(VAL_SEQS)} official val seqs with SAM2 AMG proposals", flush=True)

# SAM2 image predictor for AMG (generates mask proposals from dense grid)
print("Loading SAM2 image model for AMG...", flush=True)
sam2_image = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE)
amg = SAM2AutomaticMaskGenerator(
    model=sam2_image,
    points_per_side=16,        # 16x16 = 256 grid points (moderate density)
    pred_iou_thresh=0.80,      # keep high-quality masks
    stability_score_thresh=0.90,
    box_nms_thresh=0.5,        # merge overlapping proposals
    min_mask_region_area=400,  # ignore tiny masks (<400 px)
)
print("Loading SAM2 Large video predictor...", flush=True)
sam2_video = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def eval_sequence(seq):
    frame_paths = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mask_paths  = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    n = min(len(frame_paths), len(mask_paths))
    if n == 0: return None
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE)
                for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                for p in mask_paths[:n]]
    h, w = frames[0].shape[:2]

    gt_ids = set()
    for m in gt_masks: gt_ids.update(int(x) for x in np.unique(m[m>0]))
    if not gt_ids: return None

    # Generate mask proposals from frame 0 using SAM2 AMG
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposals = amg.generate(frames[0])
    proposals.sort(key=lambda x: x["predicted_iou"], reverse=True)
    print(f"  {seq}: {len(proposals)} SAM2 AMG proposals", flush=True)
    if len(proposals) == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_proposals": 0}

    # Limit to top-60 proposals by predicted IoU to control SAM2 propagation cost
    proposals = proposals[:60]

    tmpdir = tempfile.mkdtemp(prefix=f"amg_{seq}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    n_props = len(proposals)
    pred_by_track = {i: [np.zeros((h,w), np.uint8) for _ in range(n)]
                     for i in range(n_props)}

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2_video.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, prop in enumerate(proposals):
                # Use the AMG segmentation mask as initialization
                seg_mask = prop["segmentation"].astype(np.uint8)  # H×W binary
                if seg_mask.shape != (h, w):
                    seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                # Add mask prompt to video predictor at frame 0
                sam2_video.add_new_mask(state, frame_idx=0, obj_id=i, mask=seg_mask > 0)
            for frame_idx, obj_ids, masks in sam2_video.propagate_in_video(state):
                if frame_idx >= n: break
                for oi, ml in zip(obj_ids, masks):
                    m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h,w): m = cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
                    pred_by_track[int(oi)][frame_idx] = m
            sam2_video.reset_state(state)
    except Exception as e:
        print(f"  SAM2 video error on {seq}: {e}", flush=True)
        # Fallback: use static masks from frame 0
        for i, prop in enumerate(proposals):
            seg = prop["segmentation"].astype(np.uint8)
            if seg.shape != (h, w): seg = cv2.resize(seg,(w,h),interpolation=cv2.INTER_NEAREST)
            for fi in range(n): pred_by_track[i][fi] = seg

    shutil.rmtree(tmpdir, ignore_errors=True)

    j_scores, f_scores = [], []
    for gt_id in gt_ids:
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]
        best_j, best_f = 0.0, 0.0
        for track_id, track_masks in pred_by_track.items():
            mj = float(np.mean([compute_j_score(track_masks[i], gt_seq[i]) for i in range(n)]))
            if mj > best_j:
                best_j = mj
                best_f = float(np.mean([compute_f_score(track_masks[i], gt_seq[i]) for i in range(n)]))
        j_scores.append(best_j); f_scores.append(best_f)

    mj, mf = float(np.mean(j_scores)), float(np.mean(f_scores))
    return {"J": round(mj,4), "F": round(mf,4), "JF": round((mj+mf)/2,4), "n_proposals": n_props}


results = {}
t0 = time.time()
for seq in VAL_SEQS:
    t1 = time.time()
    print(f"\n[{list(VAL_SEQS).index(seq)+1}/{len(VAL_SEQS)}] {seq}", flush=True)
    r = eval_sequence(seq)
    if r:
        results[seq] = r
        elapsed = time.time()-t1
        print(f"  J={r['J']:.4f} F={r['F']:.4f} JF={r['JF']:.4f} ({elapsed:.1f}s)", flush=True)
    Js = [v["J"] for v in results.values()]
    Fs = [v["F"] for v in results.values()]
    out = {"per_seq": results,
           "mean_J": round(float(np.mean(Js)),4),
           "mean_F": round(float(np.mean(Fs)),4),
           "mean_JF": round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
           "n_seqs": len(results)}
    with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(out, fp, indent=2)

total = time.time()-t0
Js = [v["J"] for v in results.values()]
Fs = [v["F"] for v in results.values()]
final = {"per_seq": results,
         "mean_J": round(float(np.mean(Js)),4),
         "mean_F": round(float(np.mean(Fs)),4),
         "mean_JF": round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
         "n_seqs": len(results), "total_s": round(total,1),
         "model": "sam2-large-amg", "points_per_side": 16, "max_proposals": 60}
with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(final, fp, indent=2)
print(f"\n=== SAM2-AMG FINAL: J={final['mean_J']:.4f} F={final['mean_F']:.4f} JF={final['mean_JF']:.4f} ({len(results)}/{len(VAL_SEQS)} seqs) ===", flush=True)
