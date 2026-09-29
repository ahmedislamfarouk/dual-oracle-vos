"""
OW-VIS DAVIS 2017 Val (official 30 seqs): DETR-DC5 (ResNet-50 with dilated convolutions).
DC5 has higher-resolution feature maps -> better small object detection.
Target: fix kite-surf (J=0.46) and paragliding-launch (J=0.31) failures.
GPU: 0
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
import torch, cv2, numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE     = "cuda"
TARGET_SIZE = (854, 480)
TAU        = 0.25   # same as paper's best tau
SUPPRESS_COCO_ABOVE = 0.7
OUT_DIR    = "/path/to/ow_vis/results/detr_dc5_val30"
os.makedirs(OUT_DIR, exist_ok=True)

# Load official val sequences
with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
print(f"Evaluating {len(VAL_SEQS)} official DAVIS 2017 val sequences with DETR-DC5", flush=True)

# Load DETR-DC5 (dilation in last ResNet stage -> better small object recall)
print("Loading DETR-DC5 (facebook/detr-resnet-50-dc5)...", flush=True)
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-dc5")
detector  = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50-dc5").to(DEVICE).eval()
print(f"DETR-DC5 loaded ({sum(p.numel() for p in detector.parameters())/1e6:.1f}M params)", flush=True)

print("Loading SAM2 Large...", flush=True)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def detect_dc5(frame_rgb):
    """Detect using DETR-DC5. Returns list of (x1,y1,x2,y2) unknown boxes."""
    h, w = frame_rgb.shape[:2]
    pil = Image.fromarray(frame_rgb)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = detector(**inputs)
    logits = out.logits[0]
    pred_boxes = out.pred_boxes[0]
    probs = logits.softmax(dim=-1)
    objectness = 1.0 - probs[:, -1]
    class_probs = probs[:, :-1]
    max_class_probs, _ = class_probs.max(dim=-1)
    cx, cy, bw, bh = pred_boxes.unbind(-1)
    x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
    x2 = (cx + bw/2) * w; y2 = (cy + bh/2) * h
    boxes_xyxy = torch.stack([x1,y1,x2,y2], dim=-1)
    # Keep: objectness > 0.05 AND max_class_prob <= TAU (same logic as paper)
    mask = (objectness > 0.05) & (max_class_probs <= TAU)
    selected = boxes_xyxy[mask].cpu().numpy()
    return selected


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

    boxes = detect_dc5(frames[0])
    n_boxes = len(boxes)
    print(f"  {seq}: {n_boxes} DC5 boxes at τ={TAU}", flush=True)
    if n_boxes == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0}

    tmpdir = tempfile.mkdtemp(prefix=f"dc5_{seq}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    pred_by_track = {i: [np.zeros((h,w), np.uint8) for _ in range(n)]
                     for i in range(n_boxes)}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, box in enumerate(boxes):
                x1,y1,x2,y2 = [float(v) for v in box[:4]]
                x1=max(0,min(x1,w-1)); y1=max(0,min(y1,h-1))
                x2=max(x1+1,min(x2,w)); y2=max(y1+1,min(y2,h))
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=i,
                    box=np.array([[x1,y1,x2,y2]], dtype=np.float32))
            for frame_idx, obj_ids, masks in sam2.propagate_in_video(state):
                if frame_idx >= n: break
                for oi, ml in zip(obj_ids, masks):
                    m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                    if m.shape != (h,w): m = cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
                    pred_by_track[int(oi)][frame_idx] = m
            sam2.reset_state(state)
    except Exception as e:
        print(f"  SAM2 error on {seq}: {e}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Per-instance best-match (standard DAVIS eval protocol)
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
    return {"J": round(mj,4), "F": round(mf,4), "JF": round((mj+mf)/2,4), "n_boxes": n_boxes}


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
    # Save incrementally
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
         "model": "detr-resnet-50-dc5", "tau": TAU}
with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(final, fp, indent=2)
print(f"\n=== DETR-DC5 FINAL: J={final['mean_J']:.4f} F={final['mean_F']:.4f} JF={final['mean_JF']:.4f} ({len(results)}/{len(VAL_SEQS)} seqs) ===", flush=True)
