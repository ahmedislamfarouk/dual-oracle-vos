import sys, os, json, glob
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch, numpy as np, cv2
from PIL import Image

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
OUTPUT_DIR = "/path/to/ow_vis/results/gdino_fair_gt_categories"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
GDINO_CKPT = "/path/to/ow_vis/checkpoints/gdino/groundingdino_swint_ogc.pth"

DAVIS_CATS = {
    "bear": ["bear"], "bike-packing": ["person", "bicycle"],
    "blackswan": ["swan"], "bmx-trees": ["person"],
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

def compute_j(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter) / float(union) if union > 0 else (1.0 if inter == 0 else 0.0)

def compute_f(pred_mask, gt_mask):
    pred_b = pred_mask.astype(np.uint8)
    gt_b = gt_mask.astype(np.uint8)
    pred_e = cv2.Canny(pred_b * 255, 50, 150) > 0
    gt_e = cv2.Canny(gt_b * 255, 50, 150) > 0
    if pred_e.sum() == 0 and gt_e.sum() == 0:
        return 1.0
    if pred_e.sum() == 0 or gt_e.sum() == 0:
        return 0.0
    from scipy.ndimage import distance_transform_edt
    dist_gt = distance_transform_edt(~gt_e)
    dist_pred = distance_transform_edt(~pred_e)
    thr = 0.75 * np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2) / 100
    prec = (dist_gt[pred_e] <= thr).sum() / max(1, pred_e.sum())
    rec = (dist_pred[gt_e] <= thr).sum() / max(1, gt_e.sum())
    f = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f)

def main():
    from sam2.build_sam import build_sam2_video_predictor
    from groundingdino.util.inference import load_model, load_image, predict
    import groundingdino
    gdino_pkg = os.path.dirname(groundingdino.__file__)
    gdino_cfg = os.path.join(gdino_pkg, "config", "GroundingDINO_SwinT_OGC.py")

    gdino_model = load_model(gdino_cfg, GDINO_CKPT)
    sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    val_file = DAVIS_ROOT + "/ImageSets/2017/val.txt"
    sequences = [l.strip() for l in open(val_file) if l.strip()]
    print("Evaluating " + str(len(sequences)) + " sequences with GT-category prompts", flush=True)

    all_j, all_f, seq_results = [], [], {}

    for seq in sequences:
        cats = DAVIS_CATS.get(seq, ["object"])
        unique_cats = list(dict.fromkeys(cats))
        prompt = " . ".join(unique_cats) + " ."
        print("\n" + seq + ": prompt='" + prompt + "'", flush=True)

        frames = sorted(glob.glob(DAVIS_ROOT + "/JPEGImages/480p/" + seq + "/*.jpg"))
        gt_masks = sorted(glob.glob(DAVIS_ROOT + "/Annotations/480p/" + seq + "/*.png"))
        if not frames:
            print("  NO FRAMES"); continue

        img_src, img_t = load_image(frames[0])
        h, w = img_src.shape[:2]
        boxes, logits, phrases = predict(gdino_model, img_t, prompt, 0.25, 0.25)
        print("  " + str(len(boxes)) + " detections: " + str(phrases), flush=True)

        if len(boxes) == 0:
            seq_results[seq] = {"J": 0, "F": 0, "JF": 0, "n_det": 0, "prompt": prompt}
            all_j.append(0); all_f.append(0); continue

        boxes_px = []
        for b in boxes.tolist():
            cx, cy, bw, bh = b
            boxes_px.append([max(0,(cx-bw/2)*w), max(0,(cy-bh/2)*h),
                              min(w,(cx+bw/2)*w), min(h,(cy+bh/2)*h)])

        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
            state = sam2.init_state(video_path=DAVIS_ROOT + "/JPEGImages/480p/" + seq)
            sam2.reset_state(state)
            for oid, box in enumerate(boxes_px):
                sam2.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(box))
            pred = {}
            for fidx, oids, masks in sam2.propagate_in_video(state):
                pred[fidx] = {int(oid): (masks[i][0].cpu().numpy() > 0)
                               for i, oid in enumerate(oids)}

        j_scores, f_scores = [], []
        for fidx, gt_path in enumerate(gt_masks):
            if fidx not in pred: continue
            gt = np.array(Image.open(gt_path))  # native resolution
            gt_ids = [i for i in np.unique(gt) if i > 0]
            if not gt_ids: continue

            # Combine predicted masks (at SAM2's native resolution = same as frame)
            # SAM2 predicts at frame resolution
            pred_combined = np.zeros((h, w), dtype=np.uint8)
            for oid, mask in pred[fidx].items():
                # mask shape should be (h, w) already
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), cv2.INTER_NEAREST).astype(bool)
                pred_combined[mask] = oid + 1

            # Resize GT to frame size if needed
            if gt.shape != (h, w):
                gt = cv2.resize(gt, (w, h), cv2.INTER_NEAREST)

            pred_fg = pred_combined > 0
            for gid in gt_ids:
                gt_obj = gt == gid
                j_scores.append(compute_j(pred_fg, gt_obj))
                f_scores.append(compute_f(pred_fg, gt_obj))

        J = float(np.mean(j_scores)) if j_scores else 0
        F = float(np.mean(f_scores)) if f_scores else 0
        print("  J=" + str(round(J,4)) + " F=" + str(round(F,4)) + " JF=" + str(round((J+F)/2,4)), flush=True)
        seq_results[seq] = {"J": round(J,4), "F": round(F,4), "JF": round((J+F)/2,4),
                             "n_det": len(boxes), "prompt": prompt}
        all_j.append(J); all_f.append(F)

    mean_J = float(np.mean(all_j)) if all_j else 0
    mean_F = float(np.mean(all_f)) if all_f else 0
    mean_JF = (mean_J + mean_F) / 2
    out = {"method": "gdino_fair_gt_categories", "mean_J": round(mean_J,4),
           "mean_F": round(mean_F,4), "mean_JF": round(mean_JF,4),
           "n_seqs": len(all_j), "per_seq": seq_results}
    with open(OUTPUT_DIR + "/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== FINAL: J=" + str(round(mean_J,4)) + " F=" + str(round(mean_F,4)) + " J&F=" + str(round(mean_JF,4)) + " ===", flush=True)

if __name__ == "__main__":
    main()
