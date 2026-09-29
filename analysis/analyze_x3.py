"""x3 paired analysis. Videos an arm skipped ('no det') score 0 for that arm (the pipeline outputs nothing)."""
import json, os, numpy as np
from scipy.stats import wilcoxon
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
def ps(p): d=json.load(open(p)); return d.get("per_seq",d)
def merge(*a):
    o={}
    for p in a:
        if os.path.exists(p): o.update(ps(p))
    return o
def boot(d):
    rng=np.random.default_rng(0); idx=rng.integers(0,len(d),(20000,len(d))); return np.percentile(d[idx].mean(1),[2.5,97.5])*100
def cmp(A,B,ids):
    a=np.array([A.get(k,0.0) for k in ids]); b=np.array([B.get(k,0.0) for k in ids]); d=b-a
    try: p=wilcoxon(b,a).pvalue
    except Exception: p=float("nan")
    ci=boot(d); return dict(a=round(a.mean()*100,2),b=round(b.mean()*100,2),d=round(d.mean()*100,2),ci=[round(ci[0],2),round(ci[1],2)],p=float(p),n=len(ids),
                           wtl=[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d<-1e-9).sum())])
def cb(D,arm=None,key="commit_bij"): return {k:(v[arm][key] if arm else v[key]) for k,v in D.items() if (v.get(arm) if arm else v) is not None and ((v[arm] if arm else v).get(key) is not None)}
out={}
E5=merge("x2/e9_cov05_s0/results.json","x2/e9_cov05_s1/results.json"); IDS=sorted(E5)          # the paper's 118
arms={"base":cb(E5,"base"),"TIDE":cb(E5,"tide"),
      "O4_best1":cb(merge(*[f"x2/o4_sched_s{i}/results.json" for i in range(4)])),
      "O4_best3":cb(ps("x2/x2_o4k3/results.json"))}
for t,n in [("x3_mose_o4_q3","quart3"),("x3_mose_o4_q1","quart1"),("x3_mose_o4_r3a","rand3a"),("x3_mose_o4_r3b","rand3b"),("x3_mose_o4_mid1","mid1")]:
    p=f"x3/{t}/results.json"
    if os.path.exists(p): arms[n]=cb(ps(p))
out["mose_means"]={n:round(np.mean([A.get(k,0.0) for k in IDS])*100,2) for n,A in arms.items()}
out["mose_skipped"]={n:len([k for k in IDS if k not in A]) for n,A in arms.items()}
pairs=[("O4_best1","O4_best3"),("O4_best1","quart3"),("quart1","quart3"),("O4_best1","quart1"),("O4_best1","mid1"),("quart3","O4_best3"),("rand3a","O4_best3"),("TIDE","quart3"),("TIDE","rand3a"),("rand3a","rand3b"),("TIDE","O4_best1"),("TIDE","O4_best3")]
out["mose_pairs"]={f"{b} vs {a}":cmp(arms[a],arms[b],IDS) for a,b in pairs if a in arms and b in arms}
# 2x2 on the videos every arm seeded (robustness to skips)
seeded=[k for k in IDS if all(k in arms[n] for n in ("quart1","quart3","O4_best1","O4_best3"))]
out["mose_2x2_seeded_only"]={f"{b} vs {a}":cmp(arms[a],arms[b],seeded) for a,b in [("quart1","O4_best1"),("quart3","O4_best3"),("quart1","quart3"),("O4_best1","O4_best3")]}
# non-overlap re-score (excl = first claim wins a pixel; excl20 = top-20 by quality, DAVIS-2019 unsupervised style)
def tide_arms(files):
    T=merge(*files); return T,{f"{a}:{k}":cb(T,a,k) for a in ("base","tide") for k in ("commit_bij","commit_bij_excl","commit_bij_excl20")}
def excl_block(name,files,ids=None,extra=None):
    T,A=tide_arms(files)
    if not T: return
    if extra: A.update(extra)
    ids=ids or sorted(T)
    blk={"n":len(ids),"means":{n:round(np.mean([v.get(k,0.0) for k in ids])*100,2) for n,v in A.items()},
         "ncommit":{a:round(float(np.mean([T[k][a]["ncommit"] for k in T])),1) for a in ("base","tide")},
         "gt20":{a:int(sum(T[k][a]["ncommit"]>20 for k in T)) for a in ("base","tide")}}
    blk["pairs"]={f"tide-base [{k}]":cmp(A[f"base:{k}"],A[f"tide:{k}"],ids) for k in ("commit_bij","commit_bij_excl","commit_bij_excl20")}
    for n in (extra or {}):
        blk["pairs"][f"{n} vs QRD"]=cmp(A["tide:"+("commit_bij_excl" if "excl" in n else "commit_bij")],A[n],ids)
    out[name]=blk
excl_block("mose_excl",["x3/x3_mose_tide_s0/results.json","x3/x3_mose_tide_s1/results.json"],IDS)
excl_block("davis_excl",["x3/x3_dv_tide_s0/results.json","x3/x3_dv_tide_s1/results.json"])
excl_block("mose_tide07_excl",["x3/x3_mose_tide07_s0/results.json","x3/x3_mose_tide07_s1/results.json"],IDS)
# O4 under non-overlap on MOSE (b1/b3 re-runs carry commit_bij_excl)
for t,n in [("x3_mose_o4_b1","O4_best1"),("x3_mose_o4_b3","O4_best3")]:
    p=f"x3/{t}/results.json"
    if os.path.exists(p):
        R=ps(p); T=merge("x3/x3_mose_tide_s0/results.json","x3/x3_mose_tide_s1/results.json")
        out.setdefault("mose_o4_excl",{})[n]=dict(rerun_commit_bij=cmp(arms[n],cb(R),IDS),
            vs_QRD_excl=cmp(cb(T,"tide","commit_bij_excl"),cb(R,None,"commit_bij_excl"),IDS))
# held-out populations: base vs QRD (tide_x3) vs O4 (c4_x3); arms that skip a video score 0
for pop,tf,o4 in [("ytvos_late89",["x3/x3_yt_tide/results.json"],{"O4_best1":"x3_yt_o4_b1","O4_best3":"x3_yt_o4_b3","quart3":"x3_yt_o4_q3"}),
                  ("lvos",["x3/x3_lvos_tide_s0/results.json","x3/x3_lvos_tide_s1/results.json"],{"O4_best1":"x3_lvos_o4_b1","O4_best3":"x3_lvos_o4_b3"})]:
    T=merge(*tf); A={"base":cb(T,"base"),"QRD":cb(T,"tide")}
    for n,t in o4.items():
        p=f"x3/{t}/results.json"
        if os.path.exists(p): A[n]=cb(ps(p))
    if not T: continue
    ids=sorted(set().union(*[set(v) for v in A.values()]))
    out[pop]={"n":len(ids),"skipped":{n:len([k for k in ids if k not in v]) for n,v in A.items()},
              "means":{n:round(np.mean([v.get(k,0.0) for k in ids])*100,2) for n,v in A.items()},
              "pairs":{f"{b} vs {a}":cmp(A[a],A[b],ids) for a,b in [("base","QRD"),("base","O4_best1"),("QRD","O4_best1"),("QRD","O4_best3"),("O4_best1","O4_best3"),("QRD","quart3"),("quart3","O4_best3")] if a in A and b in A}}
    ex={f"{a}:{k}":cb(T,a,k) for a in ("base","tide") for k in ("commit_bij_excl","commit_bij_excl20")}
    out[pop]["excl_means"]={n:round(np.mean([v.get(k,0.0) for k in ids])*100,2) for n,v in ex.items()}
    out[pop]["excl_pairs"]={k:cmp(ex["base:"+k],ex["tide:"+k],ids) for k in ("commit_bij_excl","commit_bij_excl20")}
T5=merge("x3/x3_mose_tide_s0/results.json","x3/x3_mose_tide_s1/results.json"); T7=merge("x3/x3_mose_tide07_s0/results.json","x3/x3_mose_tide07_s1/results.json")
if T7: out["mose_g07_vs_g05"]={k:cmp(cb(T5,"tide",k),cb(T7,"tide",k),IDS) for k in ("commit_bij","commit_bij_excl","commit_bij_excl20")}
json.dump(out,open("x3/x3_analysis.json","w"),indent=1); print(json.dumps(out,indent=1))
