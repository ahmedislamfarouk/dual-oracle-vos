"""
LEVER 2 (novel version) — GT-free detector-feature extraction on MOSE frame-0.
For each video, run DETR on the first frame ONLY (no SAM2, no GT) and extract
features that could predict discovery recall R: objectness-score distribution,
candidate count, box crowding, box sizes. Output joins by video name with the
independently-measured R (from instr_mose) to test: can R be predicted GT-free?
"""
import sys, os, glob, json
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
os.chdir("/path/to/ow_vis")
try:
    import transformers.utils.import_utils as _iu; _iu._torchao_available = False
except Exception: pass
import torch, cv2, numpy as np
best, bfree = 0, -1
for d in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(d)
    if free > bfree: bfree, best = free, d
torch.cuda.set_device(best); DEVICE = f"cuda:{best}"
from owdetr_detector import OpenWorldDetector

ROOT = "/path/to/ow_vis/datasets/MOSE/train/JPEGImages"
NAMES = [l.strip() for l in open("/path/to/ow_vis/mose_names.txt") if l.strip()]
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
print(f"DETR ready. {len(NAMES)} MOSE videos.", flush=True)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


feats = {}
for i, vid in enumerate(NAMES):
    imgs = sorted(glob.glob(f"{ROOT}/{vid}/*.jpg"))
    if not imgs: continue
    im = cv2.cvtColor(cv2.imread(imgs[0]), cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]; A = float(H * W)
    r = det.detect(im)
    boxes = list(r.get("known_boxes", [])) + list(r.get("unknown_boxes", []))
    ks = list(r.get("known_scores", [])); us = list(r.get("unknown_scores", []))
    scores = [float(s) for s in ks] + [float(s) for s in us]
    n = len(boxes)
    if n == 0:
        feats[vid] = dict(n_box=0, obj_mean=0, obj_std=0, obj_max=0, obj_min=0, obj_med=0,
                          n_high=0, n_low=0, mean_area=0, min_area=0, std_area=0,
                          crowd_iou=0, n_overlap=0, top1=0, top5=0); continue
    sc = np.array(scores); areas = np.array([((b[2]-b[0])*(b[3]-b[1]))/A for b in boxes])
    ious = [box_iou(boxes[a][:4], boxes[b][:4]) for a in range(n) for b in range(a+1, n)]
    ious = np.array(ious) if ious else np.array([0.0])
    order = np.sort(sc)[::-1]
    feats[vid] = dict(
        n_box=n, obj_mean=float(sc.mean()), obj_std=float(sc.std()), obj_max=float(sc.max()),
        obj_min=float(sc.min()), obj_med=float(np.median(sc)),
        n_high=int((sc > 0.3).sum()), n_low=int((sc < 0.1).sum()),
        mean_area=float(areas.mean()), min_area=float(areas.min()), std_area=float(areas.std()),
        crowd_iou=float(ious.mean()), n_overlap=int((ious > 0.3).sum()),
        top1=float(order[0]), top5=float(order[:5].mean()))
    if (i+1) % 20 == 0: print(f"  [{i+1}/{len(NAMES)}] {vid}: n_box={n} obj_mean={sc.mean():.3f}", flush=True)

json.dump(feats, open("/path/to/ow_vis/results/mose_detector_features.json", "w"), indent=2)
print(f"SAVED {len(feats)} feature rows -> results/mose_detector_features.json", flush=True)
