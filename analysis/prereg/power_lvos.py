import json, os, numpy as np, csv
from scipy.stats import norm
def ps(p): d=json.load(open(p)); return d.get("per_seq",d)
def merge(*a):
    o={}
    for p in a:
        if os.path.exists(p): o.update(ps(p))
    return o
def cb(D,arm=None,key="commit_bij"): return {k:(v[arm][key] if arm else v[key]) for k,v in D.items() if (v.get(arm) if arm else v) is not None and ((v[arm] if arm else v).get(key) is not None)}
T=merge("x3/x3_lvos_tide_s0/results.json","x3/x3_lvos_tide_s1/results.json"); Q=cb(T,"tide"); O3=cb(ps("x3/x3_lvos_o4_b3/results.json"))
ids=sorted(set(Q)|set(O3)); d=np.array([O3.get(k,0)-Q.get(k,0) for k in ids])*100
rng=np.random.default_rng(0); m=d[rng.integers(0,len(d),(20000,len(d)))].mean(1); se=m.std()
print("LVOS n",len(d),"mean",round(d.mean(),2),"CI95",np.percentile(m,[2.5,97.5]).round(2),"SE",round(se,2),"MDE80",round(2.8016*se,2),"power at 4.23:",round(norm.cdf(4.23/se-1.96),2))
# few-objects check for the official comparison
def load(f): return {r['Sequence']:(float(r['J-Mean'])+float(r['F-Mean']))/2 for r in csv.DictReader(open(f))}
mc=load('mcvos_official_davis_unsup_perseq.csv'); uv=load('secondary/uvosam_public_official_davis_unsup_perseq.csv')
diff=sorted(((mc[k]-uv[k])*100,k) for k in mc)
print("worst 8:",[(k,round(x,1)) for x,k in diff[:8]])
tot=sum(x for x,k in diff); print("sum of all diffs",round(tot,1),"sum of worst 8",round(sum(x for x,k in diff[:8]),1))
print("mc objects < 0.1:",sum(v<0.1 for v in mc.values()),"uv objects <0.1:",sum(v<0.1 for v in uv.values()))
print("median diff",round(np.median([x for x,k in diff]),2))
