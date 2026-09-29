import sys, os, json, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
from pipeline import OWVISPipeline
from memory_bank import UnknownInstanceMemoryBank, CropEmbedder
from metrics import compute_j_score, compute_f_score
import torch, cv2, numpy as np
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"
TARGET_SIZE = (854, 480)

# DAVIS sequences where GT object is NOT in COCO 80 classes = truly unknown to DETR
NON_COCO_SEQUENCES = ["camel", "gold-fish", "goat", "kite-surf", "india", "breakdance", "cows"]

print("Evaluating " + str(len(NON_COCO_SEQUENCES)) + " non-COCO sequences")
print("Lower threshold=0.25: more objects fall into unknown bucket")

all_j, all_f = [], []
seq_results = {}

for seq_idx, seq in enumerate(NON_COCO_SEQUENCES):
    frame_paths = sorted(glob.glob(DAVIS_ROOT + "/JPEGImages/480p/" + seq + "/*.jpg"))
    mask_paths  = sorted(glob.glob(DAVIS_ROOT + "/Annotations/480p/" + seq + "/*.png"))
    if not frame_paths or not mask_paths:
        print("  " + seq + ": not found, skipping")
        continue

    n = min(len(frame_paths), len(mask_paths), 50)
    frames   = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in frame_paths[:n]]
    gt_masks = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mask_paths[:n]]

    gt_ids = set()
    for m in gt_masks:
        gt_ids.update(np.unique(m[m > 0]).tolist())
    if not gt_ids:
        continue

    bank = UnknownInstanceMemoryBank()
    emb  = CropEmbedder().to(DEVICE)
    try:
        ck = torch.load("/path/to/ow_vis/checkpoints/crop_embedder_best.pt", map_location=DEVICE)
        emb.load_state_dict(ck["model_state_dict"])
    except:
        pass

    pipe = OWVISPipeline(CKPT, "sam2.1_hiera_base_plus", bank, emb,
                         unknown_score_threshold=0.25, use_owdetr=True, device=DEVICE)

    result = pipe.process_video(frames, known_categories=[], run_sam2=True)
    masks_pf = result.get("masks_per_frame", {})
    det_pf   = result.get("per_frame_detections", [])

    instances = {}
    for i in range(n):
        det    = det_pf[i] if i < len(det_pf) else {}
        frame_m = masks_pf.get(i, {})
        for uid in det.get("unknown_ids", []):
            if uid not in instances:
                instances[uid] = [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8)] * n
            if uid in frame_m and frame_m[uid] is not None:
                m = np.array(frame_m[uid]).astype(np.uint8)
                if m.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
                    m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                instances[uid][i] = m

    print("  [" + str(seq_idx+1) + "] " + seq + ": " + str(n) + " frames, " +
          str(len(instances)) + " tracked unknowns, " + str(len(gt_ids)) + " GT instances")

    seq_j, seq_f = [], []
    for gt_id in gt_ids:
        gt_seq = [(m == gt_id).astype(np.uint8) for m in gt_masks[:n]]
        best_j = 0.0
        best_f = 0.0
        for pred_id, pred_seq in instances.items():
            js = [compute_j_score(pred_seq[i], gt_seq[i]) for i in range(n)]
            mj = float(np.mean(js))
            if mj > best_j:
                best_j = mj
                best_f = float(np.mean([compute_f_score(pred_seq[i], gt_seq[i]) for i in range(n)]))
        seq_j.append(best_j)
        seq_f.append(best_f)

    mj = float(np.mean(seq_j)) if seq_j else 0.0
    mf = float(np.mean(seq_f)) if seq_f else 0.0
    jf = (mj + mf) / 2
    seq_results[seq] = {"J": round(mj, 4), "F": round(mf, 4), "JF": round(jf, 4)}
    all_j.extend(seq_j)
    all_f.extend(seq_f)
    print("       J=" + str(round(mj, 3)) + "  F=" + str(round(mf, 3)) + "  J&F=" + str(round(jf, 3)))

oj = float(np.mean(all_j)) if all_j else 0.0
of = float(np.mean(all_f)) if all_f else 0.0
print("\n=== NON-COCO SEQUENCES RESULTS ===")
print("J=" + str(round(oj, 4)) + "  F=" + str(round(of, 4)) + "  J&F=" + str(round((oj+of)/2, 4)))

os.makedirs("/path/to/ow_vis/results/davis_noncoco", exist_ok=True)
with open("/path/to/ow_vis/results/davis_noncoco/results.json", "w") as f:
    json.dump({"overall": {"J": round(oj,4), "F": round(of,4), "JF": round((oj+of)/2,4)},
               "per_seq": seq_results, "note": "Non-COCO sequences: objects unknown to DETR"}, f, indent=2)
print("Saved results.json")
