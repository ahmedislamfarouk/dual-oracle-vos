#!/usr/bin/env python3
"""SAM2 oracle control on DAVIS-2017 val: GT first-frame masks -> SAM2 propagation.

Reproduces the archived `sam2_oracle_30seq` control and additionally records the
F score, which that run did not save.  Upper bound on SAM2 propagation capacity
given perfect first-frame proposals.  Env: ROOT, DAVIS, MAXF, CUDA_VISIBLE_DEVICES.
"""
import sys, os, json, glob, time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path(os.environ.get("ROOT", "/path/to/ow_vis")).resolve()
DAVIS = Path(os.environ.get("DAVIS", "/tmp/davis2017/DAVIS"))
MAXF = int(os.environ.get("MAXF", "60"))
OUT = os.environ.get("OUT", "/tmp/sam2_oracle_jf/results.json")
sys.path.insert(0, str(ROOT / "sam2")); sys.path.insert(0, str(ROOT / "src"))
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
from metrics import compute_j_score, compute_f_score
import torch, cv2, numpy as np
from PIL import Image
os.chdir(ROOT)
from sam2.build_sam import build_sam2_video_predictor

CKPT = str(ROOT / "checkpoints/sam2.1_hiera_large.pt")
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
W, H = 854, 480

seqs = [l.strip() for l in open(DAVIS / "ImageSets/2017/val.txt") if l.strip()]
pred = build_sam2_video_predictor(CFG, CKPT, device="cuda")
per_seq, Js, Fs = {}, [], []

for si, s in enumerate(seqs):
    t0 = time.time()
    imgs = sorted(glob.glob(str(DAVIS / "JPEGImages/480p" / s / "*.jpg")))[:MAXF]
    anns = sorted(glob.glob(str(DAVIS / "Annotations/480p" / s / "*.png")))[:MAXF]
    n = min(len(imgs), len(anns))
    imgs, anns = imgs[:n], anns[:n]
    gt0 = np.array(Image.open(anns[0]))
    obj_ids = [int(i) for i in np.unique(gt0) if i != 0]
    if not obj_ids:
        continue
    tmp = f"/tmp/_oracle_frames_{os.getpid()}"
    os.system(f"rm -rf {tmp}; mkdir -p {tmp}")
    for i, p in enumerate(imgs):
        im = cv2.resize(cv2.imread(p), (W, H))
        cv2.imwrite(f"{tmp}/{i:05d}.jpg", im)
    state = pred.init_state(video_path=tmp, offload_video_to_cpu=True)
    pred.reset_state(state)
    for oid in obj_ids:
        m = (np.array(Image.open(anns[0])) == oid).astype(np.uint8)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred.add_new_mask(state, frame_idx=0, obj_id=oid, mask=m)
    masks = {o: [np.zeros((H, W), np.uint8) for _ in range(n)] for o in obj_ids}
    for fi, ids, logits in pred.propagate_in_video(state):
        if fi >= n:
            continue
        for k, o in enumerate(ids):
            masks[o][fi] = (logits[k, 0].float().cpu().numpy() > 0).astype(np.uint8)
    sj, sf, cnt = 0.0, 0.0, 0
    for fi in range(n):
        gt = np.array(Image.open(anns[fi]))
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        for o in obj_ids:
            g = (gt == o)
            if not g.any():
                continue
            p = masks[o][fi].astype(bool)
            sj += compute_j_score(p, g); sf += compute_f_score(p, g); cnt += 1
    J, F = sj / max(cnt, 1), sf / max(cnt, 1)
    per_seq[s] = dict(J=round(J, 4), F=round(F, 4), JF=round((J + F) / 2, 4),
                      n_obj=len(obj_ids), n_frames=n)
    Js.append(J); Fs.append(F)
    os.system(f"rm -rf {tmp}")
    print(f"[{si+1}/{len(seqs)}] {s:22s} J={J:.4f} F={F:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(config=dict(ROOT=str(ROOT), DAVIS=str(DAVIS), MAXF=MAXF, ckpt=CKPT,
                           resolution=[W, H], n_seqs=len(Js)),
               oracle_mean_J=round(float(np.mean(Js)), 4),
               oracle_mean_F=round(float(np.mean(Fs)), 4),
               oracle_mean_JF=round(float((np.mean(Js) + np.mean(Fs)) / 2), 4),
               per_seq=per_seq)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
print("DONE", json.dumps({k: v for k, v in out.items() if k != "per_seq"}))
