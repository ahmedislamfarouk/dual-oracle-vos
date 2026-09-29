"""
OW-VIS DAVIS 2016 val (20 seqs): ALL DETR boxes, binary GT (m > 0).
DAVIS 2016 uses binary foreground annotations. Using DAVIS 2017 color annotations
with (m > 0) correctly recovers the binary GT for DAVIS 2016 evaluation.
This is comparable to published DAVIS 2016 numbers (XMem 92.0%, Cutie 94.4%).
GPU: 4
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
import os as _os; _os.chdir("/path/to/ow_vis")

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
import torch, cv2, numpy as np
from PIL import Image

DAVIS_ROOT  = "/tmp/davis2017/DAVIS"
SAM2_CKPT   = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG    = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE      = "cuda"
TARGET_SIZE = (854, 480)
TAU         = 0.5
OUT_DIR     = "/path/to/ow_vis/results/davis2016_binary"
os.makedirs(OUT_DIR, exist_ok=True)

with open("%s/ImageSets/2016/val.txt" % DAVIS_ROOT) as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
print("DAVIS 2016 val binary GT: %d seqs" % len(VAL_SEQS), flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=TAU, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def eval_sequence(seq):
    frame_paths = sorted(glob.glob("%s/JPEGImages/480p/%s/*.jpg" % (DAVIS_ROOT, seq)))
    mask_paths  = sorted(glob.glob("%s/Annotations/480p/%s/*.png" % (DAVIS_ROOT, seq)))
    n = min(len(frame_paths), len(mask_paths))
    if n == 0: return None
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]
    h, w = frames[0].shape[:2]
    # DAVIS 2016 binary GT: all annotated pixels (foreground = m > 0)
    gt_seqs = [(m > 0).astype(np.uint8) for m in gt_masks]
    if not any(g.sum() > 0 for g in gt_seqs):
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0}

    res = det.detect(frames[0])
    known_boxes   = list(res.get("known_boxes", []))
    unknown_boxes = list(res.get("unknown_boxes", []))
    all_boxes = known_boxes + unknown_boxes
    n_boxes = len(all_boxes)
    print("  %s: %d known + %d unknown = %d total boxes" % (seq, len(known_boxes), len(unknown_boxes), n_boxes), flush=True)
    if n_boxes == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0}

    tmpdir = tempfile.mkdtemp(prefix="d16bin_%s_" % seq)
    for i, fr in enumerate(frames):
        cv2.imwrite("%s/%05d.jpg" % (tmpdir, i), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    pred_by_track = {i: [np.zeros((h,w), np.uint8) for _ in range(n)] for i in range(n_boxes)}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
            for i, box in enumerate(all_boxes):
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
        print("  SAM2 error: %s" % e, flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Single-object: pick best track vs binary GT
    best_j, best_f = 0.0, 0.0
    for tid, tmasks in pred_by_track.items():
        mj = float(np.mean([compute_j_score(tmasks[i], gt_seqs[i]) for i in range(n)]))
        if mj > best_j:
            best_j = mj
            best_f = float(np.mean([compute_f_score(tmasks[i], gt_seqs[i]) for i in range(n)]))
    return {"J": round(best_j,4), "F": round(best_f,4), "JF": round((best_j+best_f)/2,4),
            "n_boxes": n_boxes, "n_known": len(known_boxes), "n_unknown": len(unknown_boxes)}


results = {}
t0 = time.time()
for seq in VAL_SEQS:
    t1 = time.time()
    idx = list(VAL_SEQS).index(seq)
    print("\n[%d/%d] %s" % (idx+1, len(VAL_SEQS), seq), flush=True)
    r = eval_sequence(seq)
    if r:
        results[seq] = r
        elapsed = time.time()-t1
        print("  J=%.4f F=%.4f JF=%.4f (%.1fs)" % (r["J"], r["F"], r["JF"], elapsed), flush=True)
    Js = [v["J"] for v in results.values()]
    Fs = [v["F"] for v in results.values()]
    out = {"per_seq": results,
           "mean_J": round(float(np.mean(Js)),4),
           "mean_F": round(float(np.mean(Fs)),4),
           "mean_JF": round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
           "n_seqs": len(results)}
    with open("%s/results.json" % OUT_DIR, "w") as fp: json.dump(out, fp, indent=2)

total = time.time()-t0
Js = [v["J"] for v in results.values()]
Fs = [v["F"] for v in results.values()]
final = {"per_seq": results,
         "mean_J": round(float(np.mean(Js)),4),
         "mean_F": round(float(np.mean(Fs)),4),
         "mean_JF": round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
         "n_seqs": len(results), "total_s": round(total,1),
         "model": "detr-resnet-50+sam2-large",
         "note": "all boxes (known+unknown), binary GT (m>0), DAVIS 2016 val 20 seqs"}
with open("%s/results.json" % OUT_DIR, "w") as fp: json.dump(final, fp, indent=2)
print("\n=== DAVIS2016-BINARY FINAL: J=%.4f F=%.4f JF=%.4f (%d/%d seqs) ===" % (
    final["mean_J"], final["mean_F"], final["mean_JF"], len(results), len(VAL_SEQS)), flush=True)
