# Scores DEVA class-agnostic output on DAVIS-17 val under OW-VIS's EXACT protocol
# (greedy best-per-GT + bijective Hungarian + detection P/R). F-score computed ONLY at
# selected pairs (greedy/bijective columns) -> identical result, no wasted boundary transforms.
import sys, os, json, glob, time
os.chdir('/path/to/ow_vis'); sys.path.insert(0,'/path/to/ow_vis/src')
from metrics import compute_j_score, compute_f_score
from scipy.optimize import linear_sum_assignment
import cv2, numpy as np
from PIL import Image

DAVIS_ROOT='/tmp/davis2017/DAVIS'
DEVA_OUT=sys.argv[1] if len(sys.argv)>1 else '/path/to/deva_eval/out_davis'
TARGET_SIZE=(854,480); MATCH_J_THRESH=0.5
OUT_DIR='/path/to/ow_vis/results/deva_dualmetric_val30'; os.makedirs(OUT_DIR,exist_ok=True)
VAL_SEQS=[l.strip() for l in open(f'{DAVIS_ROOT}/ImageSets/2017/val.txt') if l.strip()]

def load_gt(p):
    return cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
def load_deva(p):
    a=cv2.resize(np.array(Image.open(p).convert('RGB')), TARGET_SIZE, interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return (a[:,:,0]<<16)|(a[:,:,1]<<8)|a[:,:,2]

def eval_sequence(seq):
    mask_paths=sorted(glob.glob(f'{DAVIS_ROOT}/Annotations/480p/{seq}/*.png'))
    deva_dir=next((c for c in [f'{DEVA_OUT}/{seq}/Annotations',f'{DEVA_OUT}/{seq}',f'{DEVA_OUT}/Annotations/{seq}'] if glob.glob(f'{c}/*.png')),None)
    if deva_dir is None: print(f'  {seq}: NO DEVA OUTPUT',flush=True); return None
    deva_paths=sorted(glob.glob(f'{deva_dir}/*.png')); n=min(len(mask_paths),len(deva_paths))
    if n==0: return None
    gt=[load_gt(p) for p in mask_paths[:n]]; dv=[load_deva(p) for p in deva_paths[:n]]
    gt_ids=sorted(set(int(x) for m in gt for x in np.unique(m[m>0])))
    dv_ids=sorted(set(int(x) for m in dv for x in np.unique(m[m>0])))
    n_gt=len(gt_ids)
    if n_gt==0: return None
    n_tr=len(dv_ids)
    if n_tr==0: return {'JF_greedy':0.,'J_greedy':0.,'F_greedy':0.,'JF_bij':0.,'J_bij':0.,'F_bij':0.,'TP':0,'FP':0,'FN':n_gt,'n_gt':n_gt,'n_track':0}
    gt_bin={g:[(gt[i]==g).astype(np.uint8) for i in range(n)] for g in gt_ids}
    dv_bin={t:[(dv[i]==t).astype(np.uint8) for i in range(n)] for t in dv_ids}
    Jmat=np.zeros((n_gt,n_tr))
    for gi,g in enumerate(gt_ids):
        for ti,t in enumerate(dv_ids):
            Jmat[gi,ti]=np.mean([compute_j_score(dv_bin[t][i],gt_bin[g][i]) for i in range(n)])
    Fcache={}
    def Fpair(gi,ti):
        if (gi,ti) not in Fcache:
            g=gt_ids[gi]; t=dv_ids[ti]
            Fcache[(gi,ti)]=float(np.mean([compute_f_score(dv_bin[t][i],gt_bin[g][i]) for i in range(n)]))
        return Fcache[(gi,ti)]
    ci=Jmat.argmax(axis=1)
    Jg=float(Jmat[np.arange(n_gt),ci].mean()); Fg=float(np.mean([Fpair(gi,ci[gi]) for gi in range(n_gt)]))
    rows,cols=linear_sum_assignment(-Jmat)
    Jb=np.zeros(n_gt); Fb=np.zeros(n_gt); pairs=[]
    for r,c in zip(rows,cols): Jb[r]=Jmat[r,c]; Fb[r]=Fpair(r,c); pairs.append((r,c,Jmat[r,c]))
    Jbm=float(Jb.mean()); Fbm=float(Fb.mean())
    TP=sum(1 for _,_,j in pairs if j>=MATCH_J_THRESH); FP=n_tr-TP; FN=n_gt-TP
    print(f'  {seq}: n_gt={n_gt} n_track={n_tr} | greedy JF={(Jg+Fg)/2:.3f} bij JF={(Jbm+Fbm)/2:.3f} | TP={TP} FP={FP} FN={FN}',flush=True)
    return {'J_greedy':round(Jg,4),'F_greedy':round(Fg,4),'JF_greedy':round((Jg+Fg)/2,4),'J_bij':round(Jbm,4),'F_bij':round(Fbm,4),'JF_bij':round((Jbm+Fbm)/2,4),'TP':TP,'FP':FP,'FN':FN,'n_gt':n_gt,'n_track':n_tr}

results={}
for si,seq in enumerate(VAL_SEQS):
    print(f'[{si+1}/{len(VAL_SEQS)}] {seq}',flush=True)
    r=eval_sequence(seq)
    if r: results[seq]=r
    agg=lambda k: round(float(np.mean([v[k] for v in results.values()])),4) if results else 0.
    tp=sum(v['TP'] for v in results.values()); fp=sum(v['FP'] for v in results.values()); fn=sum(v['FN'] for v in results.values())
    out={'per_seq':results,'mean_JF_greedy':agg('JF_greedy'),'mean_J_greedy':agg('J_greedy'),'mean_F_greedy':agg('F_greedy'),'mean_JF_bij':agg('JF_bij'),'mean_J_bij':agg('J_bij'),'mean_F_bij':agg('F_bij'),'det_precision':round(tp/(tp+fp),4) if (tp+fp) else 0.,'det_recall':round(tp/(tp+fn),4) if (tp+fn) else 0.,'TP':tp,'FP':fp,'FN':fn,'mean_tracks_per_video':round(float(np.mean([v['n_track'] for v in results.values()])),1) if results else 0.,'n_seqs':len(results)}
    json.dump(out,open(f'{OUT_DIR}/results.json','w'),indent=2)
print('=== DEVA DUAL-METRIC FINAL ===',flush=True)
print(f"  greedy J&F={out['mean_JF_greedy']:.4f}  bij J&F={out['mean_JF_bij']:.4f}",flush=True)
print(f"  det P={out['det_precision']:.4f} R={out['det_recall']:.4f} (TP={out['TP']} FP={out['FP']} FN={out['FN']}, mean {out['mean_tracks_per_video']} tracks/vid)",flush=True)
