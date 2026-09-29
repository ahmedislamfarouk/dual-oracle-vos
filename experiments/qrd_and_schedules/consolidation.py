"""
LEVER 1 core — flood-free temporal-discovery consolidation (MOSE).
Tests THE novelty claim: can identity consolidation keep dense re-detection's
recall gain while controlling the track flood?

Arms per video (strict bijective + greedy):
  A base    : frame-0 all-boxes.
  B dense   : A + dense re-detection every STRIDE frames, coverage-gated (the flood).
  C consol  : B, then merge tracks that are the SAME object via temporal-mask-IoU
              consolidation (the novel operator) -> fewer tracks, recall preserved.
Report recall / tracks / J&F for each -> does C break the recall-vs-flood tradeoff?
"""
import sys, os, glob, json, tempfile, shutil, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/path/to/ow_vis/sam2"); sys.path.insert(0, "/path/to/ow_vis/src")
os.chdir("/path/to/ow_vis")
try:
    import transformers.utils.import_utils as _iu; _iu._torchao_available = False
except Exception: pass
import torch, cv2, numpy as np
from PIL import Image
best, bfree = 0, -1
for d in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(d)
    if free > bfree: bfree, best = free, d
torch.cuda.set_device(best); DEVICE = f"cuda:{best}"
from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor

ROOT = "/path/to/ow_vis/datasets/MOSE/train"
SAM2_CKPT="/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"; SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SIZE=(854,480); STRIDE=int(os.environ.get("STRIDE","8")); NV=int(os.environ.get("NV","28"))
UNCOV=0.70; CONSOL_IOU=float(os.environ.get("CONSOL_IOU","0.30"))
NAMES=[l.strip() for l in open("/path/to/ow_vis/mose_names.txt") if l.strip()][:NV]
det=OpenWorldDetector(device=DEVICE,unknown_threshold=0.5,objectness_threshold=0.05)
sam2=build_sam2_video_predictor(SAM2_CFG,SAM2_CKPT,device=DEVICE)
print(f"ready. {len(NAMES)} MOSE vids, stride={STRIDE}, consol_iou={CONSOL_IOU}",flush=True)

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

def temporal_iou(a,b):
    ious=[]
    for fa,fb in zip(a,b):
        u=(fa|fb).sum()
        if u>0: ious.append((fa&fb).sum()/u)
    return float(np.mean(ious)) if ious else 0.0

def consolidate(tracks,thr):
    """Union-find merge of tracks with mean temporal-IoU > thr (same object)."""
    tids=list(tracks.keys()); par={t:t for t in tids}
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in range(len(tids)):
        for j in range(i+1,len(tids)):
            if temporal_iou(tracks[tids[i]],tracks[tids[j]])>thr:
                par[find(tids[i])]=find(tids[j])
    groups={}
    for t in tids: groups.setdefault(find(t),[]).append(t)
    out={}
    for k,g in groups.items():
        mm=[np.zeros_like(tracks[g[0]][0]) for _ in tracks[g[0]]]
        for t in g:
            for fi,m in enumerate(tracks[t]): mm[fi]=mm[fi]|m
        out[k]=mm
    return out

def bij_recall_jf(gt_list,tracks):
    tids=list(tracks.keys()); G=len(gt_list)
    if not tids: return 0.0,0.0
    J=np.zeros((G,len(tids)))
    for gi,(_,of) in enumerate(gt_list):
        for ti,t in enumerate(tids): J[gi,ti]=np.mean([compute_j_score(tracks[t][fi],g) for fi,g in of])
    from scipy.optimize import linear_sum_assignment
    r,c=linear_sum_assignment(-J); aj=np.zeros(G); jf=np.zeros(G)
    for rr,cc in zip(r,c):
        aj[rr]=J[rr,cc]; f=np.mean([compute_f_score(tracks[tids[cc]][fi],g) for fi,g in gt_list[rr][1]]); jf[rr]=(J[rr,cc]+f)/2
    recall=float(np.mean(aj>=0.5))
    return recall,float(np.mean(jf))

def evalv(vid):
    ip=sorted(glob.glob(f"{ROOT}/JPEGImages/{vid}/*.jpg")); ap=sorted(glob.glob(f"{ROOT}/Annotations/{vid}/*.png"))
    if len(ip)<4 or not ap: return None
    frames=[cv2.resize(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB),SIZE) for p in ip]
    fn=[os.path.splitext(os.path.basename(p))[0] for p in ip]; h,w=frames[0].shape[:2]; n=len(frames); fti={f:i for i,f in enumerate(fn)}
    gt_by={}
    for p in ap:
        f=os.path.splitext(os.path.basename(p))[0]; gt_by[f]=cv2.resize(np.array(Image.open(p)),SIZE,interpolation=cv2.INTER_NEAREST)
    gids=sorted({int(x) for m in gt_by.values() for x in np.unique(m[m>0])})
    gt_list=[]
    for g in gids:
        of=[(fti[f],(gt_by[f]==g).astype(np.uint8)) for f in gt_by if f in fti and g in np.unique(gt_by[f])]
        if of: gt_list.append((g,of))
    if not gt_list: return None
    tmp=tempfile.mkdtemp(prefix=f"c_{vid}_")
    for i,fr in enumerate(frames): cv2.imwrite(f"{tmp}/{i:05d}.jpg",cv2.cvtColor(fr,cv2.COLOR_RGB2BGR))
    predA=propagate(tmp,[(b,0) for b in get_boxes(frames[0])],n,h,w); n0=len(predA)
    # dense re-detection, coverage gated
    redet=sorted(set(f for f in range(STRIDE,n,STRIDE) if 0<f<n)); newb=[]
    for kf in redet:
        cov=np.zeros((h,w),np.uint8)
        for t in predA.values(): cov|=t[kf]
        for b in get_boxes(frames[kf]):
            x1,y1,x2,y2=[int(max(0,v)) for v in b[:4]]; x2=min(w,max(x1+1,x2)); y2=min(h,max(y1+1,y2))
            ar=(x2-x1)*(y2-y1)
            if ar>0 and 1.0-int(cov[y1:y2,x1:x2].sum())/ar>=UNCOV: newb.append((b,kf))
    predN=propagate(tmp,newb,n,h,w) if newb else {}
    shutil.rmtree(tmp,ignore_errors=True)
    predB=dict(predA)
    for k,v in predN.items(): predB[n0+k]=v
    predC=consolidate(predB,CONSOL_IOU)
    rA,jA=bij_recall_jf(gt_list,predA); rB,jB=bij_recall_jf(gt_list,predB); rC,jC=bij_recall_jf(gt_list,predC)
    return dict(rA=rA,jA=jA,tA=len(predA),rB=rB,jB=jB,tB=len(predB),rC=rC,jC=jC,tC=len(predC),ngt=len(gt_list))

res={}; t0=time.time()
for i,v in enumerate(NAMES):
    try: r=evalv(v)
    except Exception as e: print(f"  skip {v}: {e}",flush=True); continue
    if not r: continue
    res[v]=r
    print(f"[{len(res)}] {v} R:{r['rA']:.2f}/{r['rB']:.2f}/{r['rC']:.2f} JF:{r['jA']:.2f}/{r['jB']:.2f}/{r['jC']:.2f} trk:{r['tA']}/{r['tB']}/{r['tC']}",flush=True)
    A=lambda k:np.mean([x[k] for x in res.values()])
    agg=dict(n=len(res),
        base=dict(R=round(A('rA'),4),JF=round(A('jA'),4),trk=round(A('tA'),1)),
        dense=dict(R=round(A('rB'),4),JF=round(A('jB'),4),trk=round(A('tB'),1)),
        consol=dict(R=round(A('rC'),4),JF=round(A('jC'),4),trk=round(A('tC'),1)))
    json.dump({"agg":agg,"per":res},open("/path/to/ow_vis/results/consolidation_mose.json","w"),indent=2)
print("\n=== CONSOLIDATION RESULT (MOSE) ===",flush=True)
for arm in ["base","dense","consol"]:
    a=agg[arm]; print(f"  {arm:7s}: R={a['R']:.3f}  J&F={a['JF']:.3f}  tracks={a['trk']:.1f}",flush=True)
print(f"  n={agg['n']}  {time.time()-t0:.0f}s",flush=True)
