import json,numpy as np,os
from scipy.stats import wilcoxon
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "x2"))
def ps(r): return json.load(open(f"{r}/results.json"))["per_seq"]
def merge(*rs):
    d={}
    for r in rs: d.update(ps(r))
    return d
def c(a,b):
    d=b-a; rng=np.random.default_rng(0); idx=rng.integers(0,len(d),(20000,len(d))); ci=np.percentile(d[idx].mean(1),[2.5,97.5])*100
    try: p=wilcoxon(b,a).pvalue
    except Exception: p=float('nan')
    return dict(d=round(d.mean()*100,2),ci=[round(ci[0],2),round(ci[1],2)],p=float(p),wtl=[int((d>1e-9).sum()),int((abs(d)<=1e-9).sum()),int((d<-1e-9).sum())])
out={}
DV=merge("dv_tide_s0","dv_tide_s1"); ks=sorted(DV)
D5=np.array([DV[k]["tide"]["commit_bij"] for k in ks]); DB=np.array([DV[k]["base"]["commit_bij"] for k in ks])
out["davis_default"]=dict(jf=round(D5.mean()*100,2),commit=round(np.mean([DV[k]["tide"]["ncommit"] for k in ks]),1))
for g in ["03","04","06","07"]:
    if not os.path.exists(f"x2_dvcov{g}/results.json"): continue
    X=ps(f"x2_dvcov{g}"); assert sorted(X)==ks
    G=np.array([X[k]["tide"]["commit_bij"] for k in ks]); B=np.array([X[k]["base"]["commit_bij"] for k in ks])
    assert np.allclose(B,DB)
    out[f"davis_g{g}"]=dict(jf=round(G.mean()*100,2),vs_default=c(D5,G),vs_frame0=c(DB,G),commit=round(np.mean([X[k]["tide"]["ncommit"] for k in ks]),1))
K3r=ps("x2_o4k3"); K1r=merge(*[f"o4_sched_s{i}" for i in range(4)]); E5=merge("e9_cov05_s0","e9_cov05_s1"); E7=merge("e9_cov07_s0","e9_cov07_s1")
S=sorted(set(K3r)&set(K1r)&set(E5)&set(E7))
v=lambda D,k,a=None:(D[k][a] if a else D[k])["commit_bij"]
K3=np.array([v(K3r,k) for k in S]);K1=np.array([v(K1r,k) for k in S]);T=np.array([v(E5,k,"tide") for k in S]);G7=np.array([v(E7,k,"tide") for k in S]);B=np.array([v(E5,k,"base") for k in S])
out["mose"]=dict(n=len(S),o4x3=round(K3.mean()*100,2),vs_o4=c(K1,K3),vs_g07=c(G7,K3),vs_tide=c(T,K3),vs_base=c(B,K3),
  o4x3_commit=round(np.mean([K3r[k]['ncommit'] for k in S]),2),o4x3_prec=round(np.mean([K3r[k]['prec'] for k in S])*100,1))
json.dump(out,open("x2_analysis.json","w"),indent=1); print(json.dumps(out,indent=1))
