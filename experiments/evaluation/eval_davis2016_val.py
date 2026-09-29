"""
OW-VIS DAVIS 2016 val (20 single-object seqs): τ=0.25, SAM2 Large.
DAVIS 2016 seqs are single-object; standard J&F on the 20 official val seqs.
GPU: 4
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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
TAU         = 0.25
OUT_DIR     = "/path/to/ow_vis/results/davis2016_val20"
os.makedirs(OUT_DIR, exist_ok=True)

with open(f"{DAVIS_ROOT}/ImageSets/2016/val.txt") as f:
    VAL_SEQS = [l.strip() for l in f if l.strip()]
print(f"DAVIS 2016 val: {len(VAL_SEQS)} sequences, τ={TAU}", flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=TAU, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)

def eval_sequence(seq):
    frame_paths = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mask_paths  = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    n = min(len(frame_paths), len(mask_paths))
    if n == 0: return None
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]
    h, w = frames[0].shape[:2]
    # DAVIS 2016: single object (all non-zero pixels = object)
    gt_seqs = [(m > 0).astype(np.uint8) for m in gt_masks]
    res = det.detect(frames[0])
    boxes = list(res.get("unknown_boxes", []))
    n_boxes = len(boxes)
    print(f"  {seq}: {n_boxes} boxes", flush=True)
    if n_boxes == 0:
        return {"J": 0.0, "F": 0.0, "JF": 0.0, "n_boxes": 0}
    tmpdir = tempfile.mkdtemp(prefix=f"d16_{seq}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpdir}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    pred_by_track = {i: [np.zeros((h,w), np.uint8) for _ in range(n)] for i in range(n_boxes)}
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
        print(f"  SAM2 error: {e}", flush=True)
    shutil.rmtree(tmpdir, ignore_errors=True)
    # Single GT object: pick best predicted track
    best_j, best_f = 0.0, 0.0
    for tid, tmasks in pred_by_track.items():
        mj = float(np.mean([compute_j_score(tmasks[i], gt_seqs[i]) for i in range(n)]))
        if mj > best_j:
            best_j = mj
            best_f = float(np.mean([compute_f_score(tmasks[i], gt_seqs[i]) for i in range(n)]))
    return {"J": round(best_j,4), "F": round(best_f,4), "JF": round((best_j+best_f)/2,4), "n_boxes": n_boxes}

results = {}
t0 = time.time()
for seq in VAL_SEQS:
    print(f"\n[{list(VAL_SEQS).index(seq)+1}/{len(VAL_SEQS)}] {seq}", flush=True)
    r = eval_sequence(seq)
    if r:
        results[seq] = r
        print(f"  J={r['J']:.4f} F={r['F']:.4f} JF={r['JF']:.4f}", flush=True)
    Js=[v["J"] for v in results.values()]; Fs=[v["F"] for v in results.values()]
    out={"per_seq":results,"mean_J":round(float(np.mean(Js)),4),"mean_F":round(float(np.mean(Fs)),4),
         "mean_JF":round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),"n_seqs":len(results)}
    with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(out, fp, indent=2)

Js=[v["J"] for v in results.values()]; Fs=[v["F"] for v in results.values()]
final={"per_seq":results,"mean_J":round(float(np.mean(Js)),4),"mean_F":round(float(np.mean(Fs)),4),
       "mean_JF":round(float(np.mean([(j+f)/2 for j,f in zip(Js,Fs)])),4),
       "n_seqs":len(results),"total_s":round(time.time()-t0,1),"tau":TAU,"dataset":"DAVIS-2016-val"}
with open(f"{OUT_DIR}/results.json","w") as fp: json.dump(final, fp, indent=2)
print(f"\n=== DAVIS-2016 FINAL: J={final['mean_J']:.4f} F={final['mean_F']:.4f} JF={final['mean_JF']:.4f} ({len(results)}/{len(VAL_SEQS)} seqs) ===", flush=True)
