"""
LEVER 1 v2 — dense discovery + GT-free PERSISTENCE COMMITMENT (not merge).
Naive union-merge blobs distinct objects and destroys multi-object recall.
Instead: keep dense re-detection's recall, then PRUNE junk tracks with a GT-free
signal (persistence + size + safe near-duplicate dedup) that keeps individual
tracks. Win condition: recall stays near dense, tracks drop near base.
Arms: A base | B dense | C dense+commit-filter.
"""
import sys, os, glob, json, tempfile, shutil, time
os.environ.setdefault("PYTORCH_ALLOC_CONF","expandable_segments:True")
sys.path.insert(0,"/path/to/ow_vis/sam2"); sys.path.insert(0,"/path/to/ow_vis/src")
os.chdir("/path/to/ow_vis")
try:
    import transformers.utils.import_utils as _iu; _iu._torchao_available=False
except Exception: pass
import torch, cv2, numpy as np
from PIL import Image
best,bf=0,-1
for d in range(torch.cuda.device_count()):
    fr,_=torch.cuda.mem_get_info(d)
    if fr>bf: bf,best=fr,d
torch.cuda.set_device(best); DEVICE=f"cuda:{best}"
from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor
ROOT="/path/to/ow_vis/datasets/MOSE/train"
SAM2_CKPT="/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"; SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SIZE=(854,480); STRIDE=int(os.environ.get("STRIDE","8")); NV=int(os.environ.get("NV","40"))
UNCOV=0.70; PERSIST=float(os.environ.get("PERSIST","0.5")); AREAMIN=float(os.environ.get("AREAMIN","0.0015"))
DEDUP=float(os.environ.get("DEDUP","0.85"))
NAMES=[l.strip() for l in open("/path/to/ow_vis/mose_names.txt") if l.strip()][:NV]
det=OpenWorldDetector(device=DEVICE,unknown_threshold=0.5,objectness_threshold=0.05)
sam2=build_sam2_video_predictor(SAM2_CFG,SAM2_CKPT,device=DEVICE)
print(f"ready {len(NAMES)} vids stride={STRIDE} persist={PERSIST} area={AREAMIN} dedup={DEDUP}",flush=True)
def get_boxes(rgb):
    r=det.detect(rgb); return list(r.get("known_boxes",[]))+list(r.get("unknown_boxes",[]))
def propagate(tmp,boxes,n,h,w):
    pred={i:[np.zeros((h,w),np.uint8) for _ in range(n)] for i in range(len(boxes))}
    if not boxes: return pred
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        st=sam2.init_state(video_path=tmp,offload_video_to_cpu=True)
        for i,(b,f) in enumerate(boxes):
            x1,y1,x2,y2=[float(v) for v in b[:4]]
            x1=max(0,min(x1,w-1));y1=max(0,min(y1,h-1));x2=max(x1+1,min(x2,w));y2=max(y1+1,min(y2,h))
            sam2.add_new_points_or_box(st,frame_idx=int(f),obj_id=i,box=np.array([[x1,y1,x2,y2]],np.float32))
        start=min(int(f) for _,f in boxes)
        for fi,oids,masks in sam2.propagate_in_video(st,start_frame_idx=start):
            if fi>=n: break
            for oi,ml in zip(oids,masks):
                m=(ml[0].cpu().numpy()>0).astype(np.uint8)
                if m.shape!=(h,w): m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
                pred[int(oi)][fi]=m
        sam2.reset_state(st)
    return pred
def tiou(a,b):
    io=[]
    for x,y in zip(a,b):
        u=(x|y).sum()
        if u>0: io.append((x&y).sum()/u)
    return float(np.mean(io)) if io else 0.0
def commit_filter(tracks,n,h,w):
    """Keep persistent, non-tiny tracks; drop junk; safe dedup of near-identical."""
    kept={}
    for t,ms in tracks.items():
        ne=[i for i,m in enumerate(ms) if m.any()]
        if not ne: continue
        span=max(1,n-ne[0]); persist=len(ne)/span
        area=np.mean([m.sum() for m in ms if m.any()])/(h*w)
        if persist>=PERSIST and area>=AREAMIN: kept[t]=ms
    tids=list(kept); drop=set()
    for i in range(len(tids)):
        if tids[i] in drop: continue
        for j in range(i+1,len(tids)):
            if tids[j] in drop: continue
            if tiou(kept[tids[i]],kept[tids[j]])>DEDUP: drop.add(tids[j])
    return {t:kept[t] for t in kept if t not in drop}
def bij(gt_list,tracks):
    tids=list(tracks); G=len(gt_list)
    if not tids: return 0.0,0.0
    J=np.zeros((G,len(tids)))
    for gi,(_,of) in enumerate(gt_list):
        for ti,t in enumerate(tids): J[gi,ti]=np.mean([compute_j_score(tracks[t][fi],g) for fi,g in of])
    from scipy.optimize import linear_sum_assignment
    r,c=linear_sum_assignment(-J); aj=np.zeros(G); jf=np.zeros(G)
    for rr,cc in zip(r,c):
        aj[rr]=J[rr,cc]; f=np.mean([compute_f_score(tracks[tids[cc]][fi],g) for fi,g in gt_list[rr][1]]); jf[rr]=(J[rr,cc]+f)/2
    return float(np.mean(aj>=0.5)),float(np.mean(jf))
def evalv(vid):
    ip=sorted(glob.glob(f"{ROOT}/JPEGImages/{vid}/*.jpg")); ap=sorted(glob.glob(f"{ROOT}/Annotations/{vid}/*.png"))
    if len(ip)<4 or not ap: return None
    fr=[cv2.resize(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB),SIZE) for p in ip]
    fn=[os.path.splitext(os.path.basename(p))[0] for p in ip]; h,w=fr[0].shape[:2]; n=len(fr); fti={f:i for i,f in enumerate(fn)}
    gt_by={}
    for p in ap:
        f=os.path.splitext(os.path.basename(p))[0]; gt_by[f]=cv2.resize(np.array(Image.open(p)),SIZE,interpolation=cv2.INTER_NEAREST)
    gids=sorted({int(x) for m in gt_by.values() for x in np.unique(m[m>0])})
    gl=[]
    for g in gids:
        of=[(fti[f],(gt_by[f]==g).astype(np.uint8)) for f in gt_by if f in fti and g in np.unique(gt_by[f])]
        if of: gl.append((g,of))
    if not gl: return None
    tmp=tempfile.mkdtemp(prefix=f"v_{vid}_")
    for i,f in enumerate(fr): cv2.imwrite(f"{tmp}/{i:05d}.jpg",cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
    pA=propagate(tmp,[(b,0) for b in get_boxes(fr[0])],n,h,w); n0=len(pA)
    redet=sorted(set(f for f in range(STRIDE,n,STRIDE) if 0<f<n)); nb=[]
    for kf in redet:
        cov=np.zeros((h,w),np.uint8)
        for t in pA.values(): cov|=t[kf]
        for b in get_boxes(fr[kf]):
            x1,y1,x2,y2=[int(max(0,v)) for v in b[:4]]; x2=min(w,max(x1+1,x2)); y2=min(h,max(y1+1,y2)); ar=(x2-x1)*(y2-y1)
            if ar>0 and 1.0-int(cov[y1:y2,x1:x2].sum())/ar>=UNCOV: nb.append((b,kf))
    pN=propagate(tmp,nb,n,h,w) if nb else {}; shutil.rmtree(tmp,ignore_errors=True)
    pB=dict(pA)
    for k,v in pN.items(): pB[n0+k]=v
    pC=commit_filter(pB,n,h,w)
    rA,jA=bij(gl,pA); rB,jB=bij(gl,pB); rC,jC=bij(gl,pC)
    return dict(rA=rA,jA=jA,tA=len(pA),rB=rB,jB=jB,tB=len(pB),rC=rC,jC=jC,tC=len(pC))
res={}; t0=time.time()
for v in NAMES:
    try: r=evalv(v)
    except Exception as e: print(f"skip {v}: {e}",flush=True); continue
    if not r: continue
    res[v]=r
    print(f"[{len(res)}] {v} R:{r['rA']:.2f}/{r['rB']:.2f}/{r['rC']:.2f} JF:{r['jA']:.2f}/{r['jB']:.2f}/{r['jC']:.2f} trk:{r['tA']}/{r['tB']}/{r['tC']}",flush=True)
    A=lambda k:round(float(np.mean([x[k] for x in res.values()])),4)
    agg=dict(n=len(res),base=dict(R=A('rA'),JF=A('jA'),trk=A('tA')),dense=dict(R=A('rB'),JF=A('jB'),trk=A('tB')),commit=dict(R=A('rC'),JF=A('jC'),trk=A('tC')))
    json.dump({"agg":agg,"per":res},open("/path/to/ow_vis/results/consolidation_v2.json","w"),indent=2)
print("\n=== COMMIT-FILTER RESULT (MOSE) ===",flush=True)
for a in ["base","dense","commit"]:
    x=agg[a]; print(f"  {a:7s}: R={x['R']:.3f} J&F={x['JF']:.3f} tracks={x['trk']:.1f}",flush=True)
print(f"  n={agg['n']} {time.time()-t0:.0f}s",flush=True)
