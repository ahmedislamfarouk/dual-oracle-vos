"""
INSTRUMENTED MC-VOS eval on DAVIS-2017 official val-30 (val.txt, 480p).
Same instrumentation as sq_generic_instr.py: per-GT committed-bijective (J,F) for the
EXP1 tau sweep, and per-GT three-stage recall flags (proposal / propagation / commit)
for EXP2. Env: CUDA_VISIBLE_DEVICES, MAXF(60), NSEQ(30), TAG.
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
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

DAVIS_ROOT = "/tmp/davis2017/DAVIS"
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
W, H = TARGET_SIZE
MAXF = int(os.environ.get("MAXF", "60")); NSEQ = int(os.environ.get("NSEQ", "30"))
TAG = os.environ.get("TAG", "davis_instr")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5

def temporal_iou(a, b):
    inter = union = 0
    for x, y in zip(a, b):
        inter += np.logical_and(x, y).sum(); union += np.logical_or(x, y).sum()
    return inter/union if union > 0 else 0.0

def gt_box_from_mask(gtmask):
    ys, xs = np.where(gtmask)
    if xs.size == 0: return None
    return [float(xs.min()), float(ys.min()), float(xs.max())+1, float(ys.max())+1]

def box_box_iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1); ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1); inter = iw*ih
    ua = max(0.0, ax2-ax1)*max(0.0, ay2-ay1) + max(0.0, bx2-bx1)*max(0.0, by2-by1) - inter
    return inter/ua if ua > 0 else 0.0

print("Loading DETR + SAM2 Large...", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

with open(f"{DAVIS_ROOT}/ImageSets/2017/val.txt") as _f:
    val_seqs = [ln.strip() for ln in _f if ln.strip()][:NSEQ]

agg = {"greedy": [], "commit_bij": [], "prec": [], "rec": [], "ntrk": [], "ncommit": [], "ngt": []}
STG = dict(N=0, A=0, B=0, C=0)
per_seq = {}

for si, seq in enumerate(val_seqs):
    t0 = time.time()
    fps = sorted(glob.glob(f"{DAVIS_ROOT}/JPEGImages/480p/{seq}/*.jpg"))
    mps = sorted(glob.glob(f"{DAVIS_ROOT}/Annotations/480p/{seq}/*.png"))
    if not fps or not mps:
        print(f"  [{si+1}] {seq}: missing, skip", flush=True); continue
    n = min(len(fps), len(mps), MAXF)
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), TARGET_SIZE) for p in fps[:n]]
    gtm = [cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for p in mps[:n]]
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {seq}: no GT, skip", flush=True); continue

    d0 = det.detect(frames[0])
    boxes = []
    for i, b in enumerate(d0.get("known_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"k{i}"))
    for i, b in enumerate(d0.get("unknown_boxes", [])):
        boxes.append((list(b.tolist() if hasattr(b, "tolist") else b), f"u{i}"))
    if not boxes:
        print(f"  [{si+1}] {seq}: no det, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="gtf_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    omap, i2o = {}, {}
    for _, iid in boxes:
        oid = hash(iid) % 10000
        while oid in omap and omap[oid] != iid: oid = (oid+1) % 10000
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
        print(f"  SAM2 err {seq}: {e}", flush=True)
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
    committed_set = set(committed)

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    track_ids = [iid for _, iid in boxes]
    box_of = {iid: bx for bx, iid in boxes}
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

    per_obj = [dict(J=0.0, F=0.0) for _ in gt_ids]
    P = committed
    if P:
        pidx = [track_ids.index(iid) for iid in P]; Jsub = Jall[pidx]
        ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sumhit = 0.0
        for r, c in zip(ri, ci):
            jval = float(Jsub[r, c]); fval = bF(P[r], gt_ids[c])
            per_gt[c] = (jval + fval) / 2
            per_obj[c] = dict(J=round(jval, 4), F=round(fval, 4))
            if jval > 0.5: nhit += 1; sumhit += per_gt[c]
        commit_bij = float(np.mean(per_gt)); prec = nhit/len(P); rec = nhit/len(gt_ids)
        Rstrict=nhit/len(gt_ids); Qstrict=(sumhit/nhit) if nhit else 0.0
    else:
        commit_bij = prec = rec = 0.0; Rstrict=Qstrict=0.0

    stages = []; sA = sB = sC = 0
    for gi, g in enumerate(gt_ids):
        gt0 = (gtm[0] == g)
        gbox = gt_box_from_mask(gt0)
        if gbox is None:
            stages.append(dict(a=False, b=False, c=False)); continue
        matched = [iid for _, iid in boxes if box_box_iou(box_of[iid], gbox) > 0.5]
        a = len(matched) > 0
        good = [iid for iid in matched if Jall[track_ids.index(iid), gi] > 0.5]
        b = a and len(good) > 0
        c = b and any(iid in committed_set for iid in good)
        stages.append(dict(a=bool(a), b=bool(b), c=bool(c)))
        sA += int(a); sB += int(b); sC += int(c)
    STG["N"] += len(gt_ids); STG["A"] += sA; STG["B"] += sB; STG["C"] += sC

    agg["greedy"].append(greedy); agg["commit_bij"].append(commit_bij); agg["prec"].append(prec)
    agg["rec"].append(rec); agg["ntrk"].append(len(boxes)); agg["ncommit"].append(len(P)); agg["ngt"].append(len(gt_ids))
    per_seq[seq] = dict(greedy=round(greedy, 4), commit_bij=round(commit_bij, 4), prec=round(prec, 3),
                        rec=round(rec, 3), Rstrict=round(Rstrict,4), Qstrict=round(Qstrict,4),
                        ntrk=len(boxes), ncommit=len(P), ngt=len(gt_ids), nframe=n,
                        per_obj=per_obj, stages=stages, stageN=len(gt_ids), stageA=sA, stageB=sB, stageC=sC)
    print(f"  [{si+1:2d}] {seq:20s} greedy={greedy:.3f} commitBij={commit_bij:.3f} R={Rstrict:.2f} "
          f"stg N{len(gt_ids)}/A{sA}/B{sB}/C{sC} trk={len(boxes)} commit={len(P)} t={time.time()-t0:.0f}s", flush=True)

O = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
out = dict(config=dict(DAVIS_ROOT=DAVIS_ROOT, TAG=TAG, MAXF=MAXF, NSEQ=NSEQ),
           summary=dict(n=len(agg["greedy"]), greedy=O("greedy"), commit_bij=O("commit_bij"),
                        precision=O("prec"), recall=O("rec"), mean_tracks=O("ntrk"),
                        mean_commit=O("ncommit"), mean_gt=O("ngt"), stage_counts=dict(STG),
                        stage_rates=dict(R_proposal=(STG["A"]/STG["N"] if STG["N"] else 0),
                                         R_prop_given_proposal=(STG["B"]/STG["A"] if STG["A"] else 0),
                                         R_commit_given_prop=(STG["C"]/STG["B"] if STG["B"] else 0),
                                         R_committed=(STG["C"]/STG["N"] if STG["N"] else 0))),
           per_seq=per_seq)
od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)
print(f"\n=== INSTR DAVIS {TAG} ({len(agg['greedy'])} seqs) GREEDY {O('greedy')*100:.2f} COMMIT-BIJ {O('commit_bij')*100:.2f} ===", flush=True)
print(f"STAGES N={STG['N']} A={STG['A']} B={STG['B']} C={STG['C']}", flush=True)
if STG["N"]:
    print(f"  R_proposal={STG['A']/STG['N']:.4f} R_prop|prop={STG['B']/max(STG['A'],1):.4f} "
          f"R_commit|prop={STG['C']/max(STG['B'],1):.4f} product={STG['C']/STG['N']:.4f}", flush=True)
print(f"Saved results/{TAG}/results.json", flush=True)
