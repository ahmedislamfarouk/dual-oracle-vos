import os,json,glob
import numpy as np
from PIL import Image
out={}
# DAVIS-2017 val
dv="/tmp/davis2017/DAVIS/Annotations/480p"
val="/tmp/davis2017/DAVIS/ImageSets/2017/val.txt"
seqs=[l.strip() for l in open(val) if l.strip()] if os.path.exists(val) else []
for s in seqs:
    p=sorted(glob.glob(os.path.join(dv,s,"*.png")))
    if not p: continue
    m=np.array(Image.open(p[0]))
    ids=[i for i in np.unique(m) if i!=0]
    if not ids: continue
    tot=m.shape[0]*m.shape[1]
    out["davis:"+s]=min(float((m==i).sum())/tot for i in ids)
# YouTube-VOS 2019 val
yv="/path/to/ytvos2019/valid/Annotations"
for s in sorted(os.listdir(yv)) if os.path.isdir(yv) else []:
    p=sorted(glob.glob(os.path.join(yv,s,"*.png")))
    if not p: continue
    m=np.array(Image.open(p[0]))
    ids=[i for i in np.unique(m) if i!=0]
    if not ids: continue
    tot=m.shape[0]*m.shape[1]
    out["ytvos:"+s]=min(float((m==i).sum())/tot for i in ids)
json.dump(out,open("/tmp/min_area.json","w"))
print("videos:",len(out))
