"""Equivalence (TOST-style) analysis for the single-seed oracle comparisons. Run with cwd = raw_results."""
import json, os, numpy as np
from scipy.stats import wilcoxon
def ps(p): d=json.load(open(p)); return d.get("per_seq",d)
def merge(*a):
    o={}
    for p in a:
        if os.path.exists(p): o.update(ps(p))
    return o
def cb(D,arm=None,key="commit_bij"):
    out={}
    for k,v in D.items():
        x=v[arm] if arm else v
        if x is not None and x.get(key) is not None: out[k]=x[key]
    return out
def boot(d,lo,hi):
    rng=np.random.default_rng(0); idx=rng.integers(0,len(d),(20000,len(d))); return np.percentile(d[idx].mean(1),[lo,hi])*100
def report(name,A,B,ids,zero=True):
    if not zero: ids=[k for k in ids if k in A and k in B]
    a=np.array([A.get(k,0.0) for k in ids]); b=np.array([B.get(k,0.0) for k in ids]); d=b-a
    c95=boot(d,2.5,97.5); c90=boot(d,5,95)
    m=max(abs(c90[0]),abs(c90[1]))
    print(f"{name}: n={len(ids)} A={a.mean()*100:.2f} B={b.mean()*100:.2f} d={d.mean()*100:+.2f} CI95=[{c95[0]:+.2f},{c95[1]:+.2f}] CI90=[{c90[0]:+.2f},{c90[1]:+.2f}] smallest_equiv_margin={m:.2f} p={wilcoxon(b,a).pvalue:.3g}")
E5=merge("x2/e9_cov05_s0/results.json","x2/e9_cov05_s1/results.json"); IDS=sorted(E5)
QRD=cb(E5,"tide")
O1=cb(merge("x3/c3_o1_s0/results.json","x3/c3_o1_s1/results.json"))
O4=cb(merge(*[f"x2/o4_sched_s{i}/results.json" for i in range(4)]))
for z in (True,False):
    print("zero-fill" if z else "drop-missing")
    report("MOSE O1 vs QRD (B-A = O1-QRD)",QRD,O1,IDS,z)
    report("MOSE O4 vs QRD",QRD,O4,IDS,z)
    report("MOSE O4 vs O1",O1,O4,IDS,z)
DT=merge("dv_tide_s0.json","dv_tide_s1.json"); DQ=cb(DT,"tide"); D1=cb(merge("dv_o1_s0.json","dv_o1_s1.json")); D4=cb(merge("dv_o4_s0.json","dv_o4_s1.json"))
ids=sorted(DT)
report("DAVIS O1 vs QRD",DQ,D1,ids); report("DAVIS O4 vs QRD",DQ,D4,ids)
