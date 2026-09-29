#!/usr/bin/env python3
"""Score UVOSAM predicted masks under MC-VOS's own greedy and bijective protocols.

Uses the frozen canonical metrics (`compute_j_score`, `compute_f_score`) and the
same two matching rules the manuscript reports for MC-VOS on DAVIS-2017 val:
  greedy    - each GT object takes its best-matching predicted track
  bijective - one-to-one Hungarian assignment, so extra predictions are penalised
This makes the comparison protocol-matched rather than as-reported.
"""
import os, sys, glob, json, argparse
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument('--pred', required=True)
ap.add_argument('--davis', default='/tmp/davis2017/DAVIS')
ap.add_argument('--src', default='/path/to/ow_vis/src')
ap.add_argument('--maxf', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, a.src)
from metrics import compute_j_score, compute_f_score

names = [l.strip() for l in open(f'{a.davis}/ImageSets/2017/val.txt') if l.strip()]
per_seq, G, B = {}, [], []
for vid in names:
    gtf = sorted(glob.glob(f'{a.davis}/Annotations/480p/{vid}/*.png'))
    pdf = sorted(glob.glob(f'{a.pred}/{vid}/*.png'))
    n = min(len(gtf), len(pdf))
    if a.maxf: n = min(n, a.maxf)
    if n == 0: continue
    gt_ids = sorted(int(i) for i in np.unique(np.array(Image.open(gtf[0]))) if i != 0)
    pr_ids = set()
    for f in pdf[:n]: pr_ids |= {int(i) for i in np.unique(np.array(Image.open(f))) if i != 0}
    pr_ids = sorted(pr_ids)
    if not gt_ids: continue
    S = np.zeros((len(gt_ids), max(len(pr_ids), 1)))
    for fi in range(n):
        gtm = np.array(Image.open(gtf[fi]))
        prm = np.array(Image.open(pdf[fi]))
        if prm.shape != gtm.shape:
            prm = np.array(Image.fromarray(prm).resize((gtm.shape[1], gtm.shape[0]), Image.NEAREST))
        for gi, g in enumerate(gt_ids):
            gm = (gtm == g)
            if not gm.any(): continue
            for pi, p in enumerate(pr_ids):
                pm = (prm == p)
                S[gi, pi] += (compute_j_score(pm, gm) + compute_f_score(pm, gm)) / 2
    S /= max(n, 1)
    greedy = float(S.max(axis=1).mean()) if pr_ids else 0.0
    if pr_ids:
        r, c = linear_sum_assignment(-S)
        tot = sum(S[i, j] for i, j in zip(r, c))
        bij = float(tot / len(gt_ids))
    else:
        bij = 0.0
    per_seq[vid] = dict(greedy=round(greedy, 4), bij=round(bij, 4),
                        n_gt=len(gt_ids), n_pred=len(pr_ids), n_frames=n)
    G.append(greedy); B.append(bij)
    print(f'{vid:22s} greedy={greedy:.4f} bij={bij:.4f} gt={len(gt_ids)} pred={len(pr_ids)}', flush=True)
out = dict(method='UVOSAM (IDOL prompts + SAM ViT-H)', pred_dir=a.pred, n_seqs=len(G),
           mean_greedy_JF=round(float(np.mean(G)), 4), mean_bijective_JF=round(float(np.mean(B)), 4),
           mean_pred_tracks=round(float(np.mean([v['n_pred'] for v in per_seq.values()])), 2),
           per_seq=per_seq)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=2)
print('RESULT', json.dumps({k: v for k, v in out.items() if k != 'per_seq'}))
