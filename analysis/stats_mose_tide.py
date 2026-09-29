#!/usr/bin/env python3
"""Single source of truth for the headline capped-MOSE comparison (frame-0 vs TIDE).

Paired two-sided Wilcoxon (scipy default) and a paired bootstrap of the mean
difference, 20,000 video-level resamples, numpy default_rng(seed=0).
Supersedes raw_results/mose_temporal_discovery_stats.json, whose p-value and CI
came from a different, unrecorded calculation.
"""
import json, os, numpy as np
from scipy.stats import wilcoxon
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ps = json.load(open(os.path.join(H, "results", "mose_temporal_discovery_118.json")))["per_seq"]
v = [k for k in ps if isinstance(ps[k].get("base"), dict) and isinstance(ps[k].get("tide"), dict)]
b = np.array([ps[k]["base"]["commit_bij"] for k in v]) * 100
t = np.array([ps[k]["tide"]["commit_bij"] for k in v]) * 100
d = t - b
p = float(wilcoxon(b, t).pvalue)
rng = np.random.default_rng(0)
bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)])
out = dict(n=len(v), base=b.mean(), tide=t.mean(), diff=d.mean(), wilcoxon_p=p,
           ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
json.dump(out, open(os.path.join(H, "results", "mose_tide_stats_recomputed.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
