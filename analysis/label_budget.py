"""Label-budget analysis: how many labelled videos are needed to estimate a benchmark's TIDE gain.
Two estimators from a random sample of k labelled videos:
  law:   beta * (1 - mean r0 on sample), beta=0.194 fitted through the origin on YouTube-VOS (full split)
  pilot: mean paired gain (tide - base) on the sample (needs both arms run on k videos)
Truth = full-population mean gain. 10,000 draws per k, seed 0."""
import json, numpy as np
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); R="results/"
rng=np.random.default_rng(0)
def pops():
    P={}
    for name,f in [("YouTube-VOS","fulltide_ytvos503.json"),("capped MOSE","fulltide_mose119.json"),("LVOS-300","lvos300.json")]:
        d=json.load(open(R+f)); v=[x for x in d.values() if all(x.get(k) is not None for k in ("r0","base","tide"))]
        P[name]=(np.array([x["r0"] for x in v]),np.array([x["tide"]-x["base"] for x in v]))
    T={}
    for s in (0,1): T.update(json.load(open(R+f"dv_tide_s{s}.json"))["per_seq"])
    ks=sorted(T); P["DAVIS-2017"]=(np.array([T[k]["base"]["rec"] for k in ks]),np.array([T[k]["tide"]["commit_bij"]-T[k]["base"]["commit_bij"] for k in ks]))
    return P
P=pops()
r0,g=P["YouTube-VOS"]; d=1-r0; beta=float((d*g).sum()/(d*d).sum())
out={"beta_ytvos":beta,"pops":{}}
print(f"beta (through origin, YouTube-VOS n={len(g)}) = {beta:.4f}")
for name,(r0,g) in P.items():
    n=len(g); truth=g.mean()*100; row={"n":n,"truth_pp":truth,"deficit":float(1-r0.mean()),"law_full_pp":beta*(1-r0.mean())*100,"k":{}}
    print(f"\n{name}: n={n} true gain {truth:+.2f}pp, deficit {1-r0.mean():.3f}, law on full pop {beta*(1-r0.mean())*100:+.2f}pp")
    for k in (5,10,20,30,50):
        if k>=n: continue
        idx=np.array([rng.choice(n,k,replace=False) for _ in range(10000)])
        law=beta*(1-r0[idx].mean(1))*100; pil=g[idx].mean(1)*100
        e_law=np.abs(law-truth); e_pil=np.abs(pil-truth)
        pos=(pil>0).mean()
        row["k"][k]=dict(law_medAE=float(np.median(e_law)),law_p90=float(np.percentile(e_law,90)),pilot_medAE=float(np.median(e_pil)),pilot_p90=float(np.percentile(e_pil,90)),pilot_sign_correct=float(pos))
        print(f"  k={k:3d}  law |err| median {np.median(e_law):.2f} p90 {np.percentile(e_law,90):.2f} | pilot |err| median {np.median(e_pil):.2f} p90 {np.percentile(e_pil,90):.2f} | pilot gain>0 in {pos*100:.0f}% of draws")
    out["pops"][name]=row
json.dump(out,open("results/label_budget.json","w"),indent=1)
