"""Paired tests of the official DAVIS-2017 unsupervised per-object scores. Run with cwd = raw_results."""
import csv, numpy as np
from scipy.stats import wilcoxon
def load(f): return {r['Sequence']:(float(r['J-Mean'])+float(r['F-Mean']))/2 for r in csv.DictReader(open(f))}
d={'mc':load('mcvos_official_davis_unsup_perseq.csv'),'uv':load('secondary/uvosam_public_official_davis_unsup_perseq.csv'),
   'orc':load('mcvos_official_davis_unsup_ORACLE20_perseq.csv')}
keys=sorted(d['mc']); rng=np.random.default_rng(0)
def boot(x): m=x[rng.integers(0,len(x),(20000,len(x)))].mean(1); return np.percentile(m,[2.5,97.5]).round(2)
for a,b in [('mc','uv'),('orc','mc')]:
    x=np.array([d[a][k]-d[b][k] for k in keys])*100
    print(f'{a}-{b} per object n={len(x)}: {x.mean():+.2f} CI95 {boot(x)} p={wilcoxon(x).pvalue:.2g} higher/lower {sum(x>0)}/{sum(x<0)} median {np.median(x):+.2f}')
    seqs=sorted({k.rsplit('_',1)[0] for k in keys})
    xs=np.array([np.mean([d[a][k]-d[b][k] for k in keys if k.rsplit('_',1)[0]==s]) for s in seqs])*100
    print(f'  per sequence n={len(xs)}: {xs.mean():+.2f} CI95 {boot(xs)} p={wilcoxon(xs).pvalue:.2g}')
