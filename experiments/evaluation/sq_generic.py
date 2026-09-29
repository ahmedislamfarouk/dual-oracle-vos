"""
Generic MC-VOS evaluation on any DAVIS-style dataset root (JPEGImages/<vid>/*.jpg +
Annotations/<vid>/*.png, color-palette masks). Used for MOSE and LVOS stress tests.

Same all-boxes DETR -> SAM2-Large pipeline + GT-free committed-track selection as
gtfree_select_eval.py, reporting greedy best-match J&F (upper bound), committed-track
bijective J&F (fair headline), precision/recall, and track counts.

Env: ROOT (dataset dir), CUDA_VISIBLE_DEVICES, MAXF (frames/video cap, default 60),
     MAXVIDS (video cap, 0=all), TAG (output subdir name).
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass
from owdetr_detector import OpenWorldDetector
from metrics import compute_f_score
import torch, cv2, numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

ROOT = os.environ["ROOT"]
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60")); MAXVIDS = int(os.environ.get("MAXVIDS", "0"))
TAG = os.environ.get("TAG", "generic")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5

def temporal_iou(a, b):
    inter = union = 0
    for x, y in zip(a, b):
        inter += np.logical_and(x, y).sum(); union += np.logical_or(x, y).sum()
    return inter/union if union > 0 else 0.0

print(f"Loading DETR + SAM2 Large... ROOT={ROOT}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

agg = {"greedy": [], "commit_bij": [], "prec": [], "rec": [], "ntrk": [], "ncommit": [], "ngt": []}
per_seq = {}

for si, vid in enumerate(vids):
    t0 = time.time()
    fps = sorted(glob.glob(os.path.join(JROOT, vid, "*.jpg")))
    astems = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(AROOT, vid, "*.png"))}
    # keep frames that have a matching annotation, cap to MAXF
    paired = [(p, astems[os.path.splitext(os.path.basename(p))[0]]) for p in fps
              if os.path.splitext(os.path.basename(p))[0] in astems]
    paired = paired[:MAXF]
    if len(paired) < 2:
        print(f"  [{si+1}] {vid}: <2 annotated frames, skip", flush=True); continue
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), TARGET_SIZE) for fp, _ in paired]
    gtm = [cv2.resize(np.array(Image.open(ap)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for _, ap in paired]
    n = len(frames)
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {vid}: no GT ids, skip", flush=True); continue

    d0 = det.detect(frames[0])
    boxes = []
    for i, b in enumerate(d0.get("known_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"k{i}"))
    for i, b in enumerate(d0.get("unknown_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"u{i}"))
    if not boxes:
        print(f"  [{si+1}] {vid}: no det, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="gen_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    omap, i2o = {}, {}
    for _, iid in boxes:
        oid = hash(iid) % 100000
        while oid in omap and omap[oid] != iid: oid = (oid+1) % 100000
        omap[oid] = iid; i2o[iid] = oid
    pred = {iid: [np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), np.uint8) for _ in range(n)] for _, iid in boxes}
    conf = {iid: [] for _, iid in boxes}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
            for bx, iid in boxes:
                x1, y1, x2, y2 = [float(v) for v in bx[:4]]
                x1 = max(0, min(x1, TARGET_SIZE[0]-1)); y1 = max(0, min(y1, TARGET_SIZE[1]-1))
                x2 = max(x1+1, min(x2, TARGET_SIZE[0])); y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
                sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i2o[iid],
                                           box=np.array([[x1, y1, x2, y2]], dtype=np.float32))
            for fi, oids, masks in sam2.propagate_in_video(st):
                if fi >= n: break
                for oid, ml in zip(oids, masks):
                    iid = omap.get(int(oid))
                    if iid is None: continue
                    logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                    if m.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
                        m = cv2.resize(m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                    pred[iid][fi] = m
                    pos = logit[logit > 0]; conf[iid].append(float(pos.mean()) if pos.size else 0.0)
            sam2.reset_state(st)
    except Exception as e:
        print(f"  SAM2 err {vid}: {e}", flush=True)
    shutil.rmtree(tmpd, ignore_errors=True)

    sig = {}
    for _, iid in boxes:
        ne = [int(m.sum()) for m in pred[iid] if m.sum() > 0]
        sig[iid] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                        quality=(len(ne)/n)*(float(np.mean(conf[iid])) if conf[iid] else 0.0))
    cand = [iid for _, iid in boxes if sig[iid]["persistence"] >= PERSIST_MIN and 0 < sig[iid]["mean_area"] < AREA_MAX]
    cand.sort(key=lambda i: sig[i]["quality"], reverse=True)
    committed = []
    for iid in cand:
        if all(temporal_iou(pred[iid], pred[k]) < DEDUP_IOU for k in committed): committed.append(iid)

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    track_ids = [iid for _, iid in boxes]
    def track_J(iid):
        pb = np.stack(pred[iid]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        return out
    _fc = {}
    def bF(iid, g):
        k = (iid, g)
        if k not in _fc:
            _fc[k] = float(np.mean([compute_f_score(pred[iid][i], gt_seqs[g][i]) for i in range(n)]))
        return _fc[k]
    Jall = np.array([track_J(iid) for iid in track_ids])
    g_jf = []
    for gi, g in enumerate(gt_ids):
        bt = int(np.argmax(Jall[:, gi])); g_jf.append((Jall[bt, gi] + bF(track_ids[bt], g)) / 2)
    greedy = float(np.mean(g_jf))
    P = committed
    if P:
        pidx = [track_ids.index(iid) for iid in P]; Jsub = Jall[pidx]
        ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sumhit = 0.0
        for r, c in zip(ri, ci):
            per_gt[c] = (Jsub[r, c] + bF(P[r], gt_ids[c])) / 2
            if Jsub[r, c] > 0.5: nhit += 1; sumhit += per_gt[c]
        commit_bij = float(np.mean(per_gt)); prec = nhit/len(P); rec = nhit/len(gt_ids)
        nmatch=int((per_gt>0).sum()); Rdisc=nmatch/len(gt_ids); Qcond=float(per_gt[per_gt>0].mean()) if nmatch else 0.0
        Rstrict=nhit/len(gt_ids); Qstrict=(sumhit/nhit) if nhit else 0.0
    else:
        commit_bij = prec = rec = 0.0
        nmatch=0; Rdisc=Qcond=Rstrict=Qstrict=0.0

    agg["greedy"].append(greedy); agg["commit_bij"].append(commit_bij); agg["prec"].append(prec)
    agg["rec"].append(rec); agg["ntrk"].append(len(boxes)); agg["ncommit"].append(len(P)); agg["ngt"].append(len(gt_ids))
    per_seq[vid] = dict(greedy=round(greedy, 4), commit_bij=round(commit_bij, 4), prec=round(prec, 3),
                        rec=round(rec, 3), nmatch=nmatch, Rdisc=round(Rdisc,4), Qcond=round(Qcond,4), Rstrict=round(Rstrict,4), Qstrict=round(Qstrict,4), ntrk=len(boxes), ncommit=len(P), ngt=len(gt_ids), nframe=n)
    print(f"  [{si+1:3d}] {vid:22s} greedy={greedy:.3f} commitBij={commit_bij:.3f} P={prec:.2f} R={rec:.2f} "
          f"trk={len(boxes)} commit={len(P)} gt={len(gt_ids)} nf={n} t={time.time()-t0:.0f}s", flush=True)

O = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
gr, cb = O("greedy"), O("commit_bij")
print(f"\n=== MC-VOS on {TAG} ({len(agg['greedy'])} videos, MAXF={MAXF}) ===", flush=True)
print(f"GREEDY best-match J&F : {gr*100:.2f}%   (upper bound)", flush=True)
print(f"COMMITTED-track J&F   : {cb*100:.2f}%   (fair headline; retention {cb/gr*100 if gr else 0:.1f}%)", flush=True)
print(f"committed precision   : {O('prec')*100:.1f}%   recall {O('rec')*100:.1f}%", flush=True)
print(f"tracks/video {O('ntrk'):.1f} -> committed {O('ncommit'):.1f}  (GT objs {O('ngt'):.1f})", flush=True)
out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS),
           summary=dict(n=len(agg["greedy"]), greedy=gr, commit_bij=cb, retention=(cb/gr if gr else 0),
                        precision=O("prec"), recall=O("rec"), mean_tracks=O("ntrk"),
                        mean_commit=O("ncommit"), mean_gt=O("ngt")), per_seq=per_seq)
od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)
print(f"Saved {od}/results.json", flush=True)
