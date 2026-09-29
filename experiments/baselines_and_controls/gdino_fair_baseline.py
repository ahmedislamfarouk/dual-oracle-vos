"""
GDINO fair baseline: run with per-sequence GT category names.
Addresses reviewer concern that original run used generic "object." prompt.
"""
import sys, os, json, glob
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch, numpy as np, cv2
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
OUTPUT_DIR = "/path/to/ow_vis/results/gdino_fair_gt_categories"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_large.yaml"
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
TARGET_SIZE = (854, 480)

# GT category names for each DAVIS 2017 val sequence
DAVIS_CATS = {
    "bear": ["bear"], "bike-packing": ["person", "bicycle"],
    "blackswan": ["swan"], "bmx-trees": ["person", "bicycle"],
    "breakdance": ["person"], "camel": ["camel"],
    "car-roundabout": ["car"], "car-shadow": ["car"],
    "cows": ["cow"], "dance-twirl": ["person"],
    "dog": ["dog"], "dogs-jump": ["dog"],
    "drift-chicane": ["car"], "drift-straight": ["car"],
    "goat": ["goat"], "gold-fish": ["fish"],
    "horsejump-high": ["horse", "person"], "india": ["person"],
    "judo": ["person"], "kite-surf": ["person", "kite"],
    "lab-coat": ["person"], "libby": ["person"],
    "loading": ["person"], "mbike-trick": ["person", "motorcycle"],
    "motocross-jump": ["person", "motorcycle"],
    "paragliding-launch": ["person"], "parkour": ["person"],
    "pigs": ["pig"], "scooter-black": ["person", "motorcycle"],
    "shooting": ["person"], "soapbox": ["person"],
}

from metrics import compute_j_score, compute_f_score

def main():
    from sam2.build_sam import build_sam2_video_predictor
    from groundingdino.util.inference import load_model, load_image, predict

    gdino_model = load_model(
        "/path/to/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "/path/to/ow_vis/checkpoints/groundingdino_swint_ogc.pth"
    )
    sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    val_file = f"{DAVIS_ROOT}/ImageSets/2017/val.txt"
    sequences = [l.strip() for l in open(val_file) if l.strip()]
    print(f"Evaluating {len(sequences)} sequences with GT-category prompts")

    all_j, all_f, seq_results = [], [], {}

    for seq in sequences:
        cats = DAVIS_CATS.get(seq, ["object"])
        prompt = " . ".join(list(dict.fromkeys(cats))) + " ."
        print(f"\n{seq}: prompt={prompt}")

        frames = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
        gt_masks = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
        if not frames:
            print("  NO FRAMES"); continue

        # GDINO on frame 0
        img_src, img_t = load_image(frames[0])
        boxes, logits, phrases = predict(gdino_model, img_t, prompt, 0.25, 0.25)
        h, w = img_src.shape[:2]
        print(f"  {len(boxes)} detections: {phrases}")

        if len(boxes) == 0:
            seq_results[seq] = {"J": 0, "F": 0, "JF": 0, "n_det": 0, "prompt": prompt}
            all_j.append(0); all_f.append(0); continue

        boxes_px = []
        for b in boxes.tolist():
            cx,cy,bw,bh = b
            boxes_px.append([max(0,(cx-bw/2)*w), max(0,(cy-bh/2)*h),
                              min(w,(cx+bw/2)*w), min(h,(cy+bh/2)*h)])

        # SAM2 propagation
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            state = sam2.init_state(video_path=f"{DAVIS_ROOT}/JPEGImages/480p/{seq}")
            sam2.reset_state(state)
            for oid, box in enumerate(boxes_px):
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=oid,
                                            box=np.array(box))
            pred = {}
            for fidx, oids, masks in sam2.propagate_in_video(state):
                pred[fidx] = {int(oid): (masks[i][0].cpu().numpy() > 0)
                               for i, oid in enumerate(oids)}

        # Evaluate vs GT
        j_scores, f_scores = [], []
        for fidx, gt_path in enumerate(gt_masks):
            if fidx not in pred: continue
            gt = np.array(Image.open(gt_path))
            gt = cv2.resize(gt, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            gt_ids = [i for i in np.unique(gt) if i > 0]
            if not gt_ids: continue
            pred_masks = np.zeros((h, w), dtype=np.uint8) if len(pred[fidx]) == 0 else None
            combined = np.zeros_like(gt)
            for oid, mask in pred[fidx].items():
                m_r = cv2.resize(mask.astype(np.uint8), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                combined[m_r > 0] = oid + 1
            best_j = max(compute_j_score(combined == gid, combined > 0) for gid in gt_ids)
            best_f = max(compute_f_score(combined == gid, combined > 0) for gid in gt_ids)
            j_scores.append(best_j); f_scores.append(best_f)

        J = float(np.mean(j_scores)) if j_scores else 0
        F = float(np.mean(f_scores)) if f_scores else 0
        print(f"  J={J:.4f} F={F:.4f} JF={(J+F)/2:.4f}")
        seq_results[seq] = {"J": round(J,4), "F": round(F,4), "JF": round((J+F)/2,4),
                             "n_det": len(boxes), "prompt": prompt}
        all_j.append(J); all_f.append(F)

    mean_J = float(np.mean(all_j))
    mean_F = float(np.mean(all_f))
    mean_JF = (mean_J + mean_F) / 2
    out = {"method": "gdino_fair_gt_categories", "mean_J": round(mean_J,4),
           "mean_F": round(mean_F,4), "mean_JF": round(mean_JF,4),
           "n_seqs": len(all_j), "per_seq": seq_results}
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== FINAL: J={mean_J:.4f} F={mean_F:.4f} J&F={mean_JF:.4f} ===")
    print(f"Saved to {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()
