#!/usr/bin/env python3
"""Prototype: SAM2-AMG as a COMPLEMENTARY first-frame discoverer.

Motivation.  Prior results here show SAM2-AMG is worse than DETR when used as a
*replacement* proposer, and that the residual loss is concentrated in objects the
detector never proposes.  This prototype instead runs AMG only on the part of the
reference frame that the DETR-seeded tracks do not already cover, admitting a mask
only when at least UNCOVERED_MIN of its area falls outside the baseline union.  It
adds no detector call and trains nothing.

Non-evidence: a DAVIS-30 feasibility probe, not a confirmatory experiment.
Env: ROOT, DAVIS, OUT_DIR, CUDA_VISIBLE_DEVICES, UNCOVERED_MIN, AMG_PPS.
"""
import os, sys, json, glob, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path(os.environ.get("ROOT", "/path/to/ow_vis")).resolve()
DAVIS = Path(os.environ.get("DAVIS", "/tmp/davis2017/DAVIS"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/amg_gap_probe")
UNCOVERED_MIN = float(os.environ.get("UNCOVERED_MIN", "0.70"))
AMG_PPS = int(os.environ.get("AMG_PPS", "32"))
MAXF = int(os.environ.get("MAXF", "0"))          # 0 = all frames
NSEQ = int(os.environ.get("NSEQ", "0"))          # 0 = all sequences
if os.path.exists(os.path.join(OUT_DIR, "results.json")):
    raise SystemExit(f"refusing to overwrite existing {OUT_DIR}/results.json")
sys.path[:0] = [str(ROOT / "sam2"), str(ROOT / "src"),
                str(ROOT / "experiments/cpd2_20260824")]
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
import numpy as np, cv2, torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from metrics import compute_j_score, compute_f_score
import cpd1_driver as cpd1
os.chdir(ROOT)

CKPT = str(ROOT / "checkpoints/sam2.1_hiera_large.pt")
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
W, H = cpd1.TARGET_WH


class A:  # minimal args shim for cpd1.run_propagator
    sam2_config, sam2_checkpoint = CFG, CKPT
    device, offload_video_to_cpu = "cuda", True


def union_at(masks, fi):
    u = np.zeros((H, W), bool)
    for m in masks.values():
        u |= m[fi].astype(bool)
    return u


detr, _ = cpd1.load_detr("cuda")
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
amg_settings = dict(cpd1.AMG_SETTINGS); amg_settings["points_per_side"] = AMG_PPS
amg = SAM2AutomaticMaskGenerator(build_sam2(CFG, CKPT, device="cuda"), **amg_settings)

seqs = [l.strip() for l in open(DAVIS / "ImageSets/2017/val.txt") if l.strip()]
if NSEQ: seqs = seqs[:NSEQ]
per_seq, agg = {}, {"base": [], "aug": [], "n_new": [], "rec_base": [], "rec_aug": []}

for si, s in enumerate(seqs):
    t0 = time.time()
    imgs = sorted(glob.glob(str(DAVIS / "JPEGImages/480p" / s / "*.jpg")))
    anns = sorted(glob.glob(str(DAVIS / "Annotations/480p" / s / "*.png")))
    n = min(len(imgs), len(anns))
    if MAXF: n = min(n, MAXF)
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (W, H)) for p in imgs[:n]]
    gt0 = cv2.resize(np.array(Image.open(anns[0])), (W, H), interpolation=cv2.INTER_NEAREST)
    gt_ids = [int(i) for i in np.unique(gt0) if i != 0]
    if not gt_ids:
        continue
    # --- baseline: DETR frame-0, top-K, propagate
    native = detr(frames[0]); native.sort(key=lambda x: (-x["score"], x["native_id"]))
    base = [dict(it, track_id=f"{s}:base:{i:02d}", query_frame_index=0)
            for i, it in enumerate(native[:cpd1.TOPK])]
    bm, bc = cpd1.run_propagator(frames, base, A)
    # --- AMG restricted to the uncovered part of frame 0
    cov = union_at(bm, 0)
    raw = amg.generate(frames[0])
    new = []
    for k, item in enumerate(raw):
        m = item["segmentation"].astype(bool)
        a = m.sum()
        if a == 0:
            continue
        unc = 1.0 - (cov & m).sum() / a
        if unc < UNCOVERED_MIN:
            continue
        if a / (W * H) > cpd1.COMMIT_SETTINGS["area_max"]:
            continue
        x, y, w_, h_ = item["bbox"]
        new.append(dict(box=[float(x), float(y), float(x + w_), float(y + h_)],
                        score=float(item["predicted_iou"]),
                        native_id=f"amggap:{k}",
                        track_id=f"{s}:amggap:{k:02d}", query_frame_index=0))
    nm, nc = (cpd1.run_propagator(frames, new, A) if new else ({}, {}))
    # --- score both track sets, strict bijective on the same evaluator
    gts = {g: [(cv2.resize(np.array(Image.open(a)), (W, H), interpolation=cv2.INTER_NEAREST) == g)
               for a in anns[:n]] for g in gt_ids}

    def score(masks):
        ids = list(masks)
        if not ids:
            return 0.0, 0.0
        S = np.zeros((len(ids), len(gt_ids)))
        for i, tid in enumerate(ids):
            for j, g in enumerate(gt_ids):
                P = np.stack([masks[tid][f].astype(bool) for f in range(n)])
                G = np.stack(gts[g])
                inter = np.logical_and(P, G).sum((1, 2)).astype(float)
                union = np.logical_or(P, G).sum((1, 2)).astype(float)
                jm = float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean())
                fs = [compute_f_score(P[f], G[f]) for f in range(0, n, max(1, n // 20))]
                S[i, j] = (jm + float(np.mean(fs))) / 2
        r, c = linear_sum_assignment(-S)
        per = np.zeros(len(gt_ids))
        for a_, b_ in zip(r, c):
            per[b_] = S[a_, b_]
        return float(per.mean()), float((per > 0.5).mean())

    jb, rb = score(bm)
    allm = dict(bm); allm.update(nm)
    ja, ra = score(allm)
    per_seq[s] = dict(base=round(jb, 4), aug=round(ja, 4), delta=round(ja - jb, 4),
                      n_new=len(new), rec_base=round(rb, 4), rec_aug=round(ra, 4), n_gt=len(gt_ids))
    for k_, v_ in [("base", jb), ("aug", ja), ("n_new", len(new)), ("rec_base", rb), ("rec_aug", ra)]:
        agg[k_].append(v_)
    print(f"[{si+1:2d}/{len(seqs)}] {s:22s} base={jb:.4f} +amg={ja:.4f} "
          f"d={ja-jb:+.4f} new={len(new)} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(config=dict(uncovered_min=UNCOVERED_MIN, amg_points_per_side=AMG_PPS,
                           topk=cpd1.TOPK, resolution=[W, H], n_seqs=len(agg["base"])),
               summary={k: float(np.mean(v)) for k, v in agg.items()}, per_seq=per_seq)
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)
print("DONE", json.dumps(out["summary"]))
