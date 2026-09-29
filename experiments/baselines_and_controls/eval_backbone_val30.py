"""
OW-VIS DAVIS 2017 val: SAM2 backbone ablation (MEASURED accuracy AND throughput).
Usage: python eval_backbone_val30.py <small|bplus|large>
Runs the identical all-boxes pipeline with the chosen SAM2 backbone, reporting
greedy + bijective J&F and mean FPS (frames/sec over propagation). Replaces the
paper's *extrapolated* SAM2-Small throughput claim with a real measurement.
GPU: 1
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
from scipy.optimize import linear_sum_assignment
import torch, cv2, numpy as np
from PIL import Image

BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "large"
CKPTS = {
    "small": ("/path/to/ow_vis/checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "bplus": ("/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
SAM2_CKPT, SAM2_CFG = CKPTS[BACKBONE]
DAVIS_ROOT = "/tmp/davis2017/DAVIS"; DEVICE = "cuda"; TARGET_SIZE = (854, 480)
OUT_DIR = f"/path/to/ow_vis/results/backbone_{BACKBONE}_val30"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print(f"Backbone={BACKBONE} ({os.path.basename(SAM2_CKPT)}) on {len(VAL_SEQS)} val seqs", flush=True)


def eval_sequence(seq):
    frame_paths = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mask_paths  = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    n = min(len(frame_paths), len(mask_paths))
    if n == 0: return None
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]
    h, w = frames[0].shape[:2]
    gt_ids = sorted(set(int(x) for m in gt_masks for x in np.unique(m[m > 0])))
    if not gt_ids: return None

    res = det.detect(frames[0])
    all_boxes = list(res.get("unknown_boxes", [])) + list(res.get("known_boxes", []))
    n_boxes = len(all_boxes)
    if n_boxes == 0: return None

    tmpdir = tempfile.mkdtemp(prefix=f"bb_{seq}_")
    for i, fr in enumerate(frames): cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    pred_by_track = {i: [np.zeros((h, w), np.uint8) for _ in range(n)] for i in range(n_boxes)}
    torch.cuda.synchronize(); t_prop = time.time()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, box in enumerate(all_boxes):
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1)); x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=i, box=np.array([[x1,y1,x2,y2]], dtype=np.float32))
            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n: break
                for oi, ml in zip(obj_ids, masks):
                    m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    pred_by_track[int(oi)][frame_idx] = m
            sam2.reset_state(state)
    except Exception as e:
        print(f"  SAM2 error {seq}: {e}", flush=True)
    torch.cuda.synchronize(); prop_s = time.time() - t_prop
    shutil.rmtree(tmpdir, ignore_errors=True)
    fps = (n * n_boxes) / prop_s if prop_s > 0 else 0.0   # frame-object propagations / sec
    fps_frames = n / prop_s if prop_s > 0 else 0.0

    n_gt = len(gt_ids)
    Jmat = np.zeros((n_gt, n_boxes)); Fmat = np.zeros((n_gt, n_boxes))
    for gi, gt_id in enumerate(gt_ids):
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks]
        for ti in range(n_boxes):
            tm = pred_by_track[ti]
            Jmat[gi, ti] = float(np.mean([compute_j_score(tm[i], gt_seq[i]) for i in range(n)]))
            Fmat[gi, ti] = float(np.mean([compute_f_score(tm[i], gt_seq[i]) for i in range(n)]))
    gi2 = Jmat.argmax(axis=1)
    Jg = float(Jmat[np.arange(n_gt), gi2].mean()); Fg = float(Fmat[np.arange(n_gt), gi2].mean())
    rows, cols = linear_sum_assignment(-Jmat)
    Jb = np.zeros(n_gt); Fb = np.zeros(n_gt)
    for r, c in zip(rows, cols): Jb[r] = Jmat[r, c]; Fb[r] = Fmat[r, c]
    print(f"  {seq}: greedy JF={(Jg+Fg)/2:.3f} bij JF={(float(Jb.mean())+float(Fb.mean()))/2:.3f} "
          f"fps_frames={fps_frames:.2f} ({n} frames x {n_boxes} obj in {prop_s:.1f}s)", flush=True)
    return {"JF_greedy": round((Jg+Fg)/2,4), "JF_bij": round((float(Jb.mean())+float(Fb.mean()))/2,4),
            "fps_frames": round(fps_frames,3), "fps_frameobj": round(fps,2),
            "n_frames": n, "n_boxes": n_boxes, "prop_s": round(prop_s,2)}


results = {}
for si, seq in enumerate(VAL_SEQS):
    print(f"[{si+1}/{len(VAL_SEQS)}] {seq}", flush=True)
    r = eval_sequence(seq)
    if r: results[seq] = r
    def m(k): return round(float(np.mean([v[k] for v in results.values()])),4) if results else 0.0
    summary = {"backbone": BACKBONE, "per_seq": results,
               "mean_JF_greedy": m("JF_greedy"), "mean_JF_bij": m("JF_bij"),
               "mean_fps_frames": m("fps_frames"), "total_prop_s": round(sum(v["prop_s"] for v in results.values()),1),
               "n_seqs": len(results)}
    with open(f"{OUT_DIR}/results.json", "w") as fp: json.dump(summary, fp, indent=2)
print(f"\n=== BACKBONE {BACKBONE} FINAL: greedy J&F={summary['mean_JF_greedy']:.4f} "
      f"bij J&F={summary['mean_JF_bij']:.4f} mean_fps={summary['mean_fps_frames']:.2f} ===", flush=True)
