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
# Thread diet: the account has a hard RLIMIT_NPROC and a NEW SSH LOGIN inherits the
# lower default, so if the running jobs push the account over it nobody can log in.
# TensorFlow is never used here -- transformers imports it only because it is installed --
# and OpenCV otherwise starts one worker thread per core for trivial resizes.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
if os.environ.get("ALLOC_CONF", "") != "none":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
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
cv2.setNumThreads(1)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS","1")))
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

ROOT = os.environ["ROOT"]
_SZ = os.environ.get("SAM2_SIZE", "large").lower()
_MAP = {
  "tiny":   ("sam2.1_hiera_tiny.pt",      "configs/sam2.1/sam2.1_hiera_t.yaml"),
  "small":  ("sam2.1_hiera_small.pt",     "configs/sam2.1/sam2.1_hiera_s.yaml"),
  "base":   ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
  "large":  ("sam2.1_hiera_large.pt",     "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
_ck, _cf = _MAP[_SZ]
SAM2_CKPT = "/path/to/ow_vis/checkpoints/" + _ck
SAM2_CFG  = _cf
print(f"[backbone] SAM2 {_SZ}: {SAM2_CKPT}", flush=True)
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
TTA = int(os.environ.get("TTA", "0"))
OBJ_THR = float(os.environ.get("OBJ_THR", "0.05"))
print(f"[objectness floor] {OBJ_THR}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=OBJ_THR)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models loaded.", flush=True)

JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
_SH = int(os.environ.get("SHARD", "0")); _NS = int(os.environ.get("NSHARD", "1"))
if _NS > 1:
    vids = vids[_SH::_NS]
    print(f"[shard {_SH}/{_NS}] {len(vids)} videos", flush=True)
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

    ORACLE = int(os.environ.get("ORACLE", "0"))
    if ORACLE == 1:
        # O1: perfect detection AT FRAME 0. Every ground-truth object visible in the
        # reference frame is handed to the propagator as an exact bounding box. Objects
        # absent from frame 0 cannot be seeded at all, which is the point of the rung:
        # it bounds what any frame-0 detector, however good, could achieve.
        m0 = gtm[0]
        boxes = []
        for _g in sorted({int(v) for v in np.unique(m0[m0 > 0])}):
            ys, xs = np.where(m0 == _g)
            if xs.size < 10:
                continue
            boxes.append(([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                          f"o{_g}"))
        print(f"    [O1] {len(boxes)} GT boxes at frame 0, of {len(gt_ids)} GT objects in the video",
              flush=True)
    elif ORACLE == 4:
        # O4, the SCHEDULING oracle. The detector is the real one, unchanged; only the
        # CHOICE OF FRAME is oracular. For every ground-truth object we run the detector on
        # every frame, score each detection against that object's ground-truth box on that
        # frame, and seed the object at the frame where the real detector sees it best.
        # This bounds what any scheduling policy could achieve with detection held fixed --
        # the complement of O1, which bounds detection with timing held fixed.
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
            inter = iw*ih
            ua = max(0.0,(ax2-ax1))*max(0.0,(ay2-ay1)) + max(0.0,(bx2-bx1))*max(0.0,(by2-by1)) - inter
            return inter/ua if ua > 0 else 0.0
        det_by_frame = {}
        for _f in range(n):
            dv = det.detect(frames[_f])
            bl = []
            for key in ("known_boxes", "unknown_boxes"):
                for b in dv.get(key, []):
                    bl.append(list(b.tolist() if hasattr(b, "tolist") else b))
            det_by_frame[_f] = bl
        best = {}
        for _f in range(n):
            m = gtm[_f]
            for _g in np.unique(m[m > 0]):
                _g = int(_g)
                ys, xs = np.where(m == _g)
                if xs.size < 10: continue
                gt = (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))
                for b in det_by_frame[_f]:
                    v = _iou(tuple(float(x) for x in b[:4]), gt)
                    if v > best.get(_g, (0.0, None, None))[0]:
                        best[_g] = (v, _f, list(b))
        # MULTI-SEED variant: give each object its K best-overlap frames (spread at least
        # n/(2K) apart) instead of one, each as its own track; commitment then chooses.
        K = int(os.environ.get("OSEEDS", "1"))
        perf = {}
        for _f in range(n):
            m = gtm[_f]
            for _g in np.unique(m[m > 0]):
                _g = int(_g)
                ys, xs = np.where(m == _g)
                if xs.size < 10: continue
                gt = (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))
                for b in det_by_frame[_f]:
                    v = _iou(tuple(float(x) for x in b[:4]), gt)
                    if v > perf.get((_g, _f), (0.0, None))[0]:
                        perf[(_g, _f)] = (v, list(b))
        sep = max(1, n // (2 * K))
        boxes, seed_at, seed_mask = [], {}, {}
        for _g in sorted(best):
            cand = sorted([(v, f, b) for (g, f), (v, b) in perf.items() if g == _g and v > 0], key=lambda x: -x[0])
            chosen = []
            for v, f, b in cand:
                if all(abs(f - c[1]) >= sep for c in chosen):
                    chosen.append((v, f, b))
                if len(chosen) == K: break
            for j, (v, f, b) in enumerate(chosen):
                _iid = f"o{_g}_{j}"
                boxes.append((b, _iid)); seed_at[_iid] = int(f)
        _late = sum(1 for x in seed_at.values() if x > 0)
        print(f"    [O4] {len(boxes)} objects seeded at their best-detection frame "
              f"({_late} after frame 0), of {len(gt_ids)} GT objects", flush=True)
    elif ORACLE in (2, 3):
        # O2: perfect detection AT EACH OBJECT'S FIRST APPEARANCE. O3: the same, seeded with
        # the ground-truth MASK rather than its box, which also removes box-to-mask error.
        # Unlike O1 these rungs can reach objects absent from frame 0, so the difference
        # between them and O1 is the value of detecting at the right TIME rather than better.
        _first = {}
        for _fi, _m in enumerate(gtm):
            for _g in np.unique(_m[_m > 0]):
                _g = int(_g)
                if _g not in _first:
                    _first[_g] = _fi
        boxes, seed_at, seed_mask = [], {}, {}
        for _g, _fi in sorted(_first.items()):
            _mm = (gtm[_fi] == _g)
            ys, xs = np.where(_mm)
            if xs.size < 10:
                continue
            _iid = f"o{_g}"
            boxes.append(([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], _iid))
            seed_at[_iid] = _fi
            if ORACLE == 3:
                seed_mask[_iid] = _mm.astype(np.uint8)
        _late = sum(1 for v in seed_at.values() if v > 0)
        print(f"    [O{ORACLE}] {len(boxes)} GT objects seeded at first appearance "
              f"({_late} after frame 0), of {len(gt_ids)} GT objects", flush=True)
    elif TTA:
        # C1 control: FOUR detector calls, all on frame 0 (identity, h-flip, 0.75x, 1.25x).
        # Same call count as TIDE, zero temporal spread. Boxes mapped back to frame-0
        # coordinates and admitted under the same coverage rule TIDE uses at a re-detect frame.
        H0, W0 = frames[0].shape[:2]
        variants = [("id", frames[0]),
                    ("flip", np.ascontiguousarray(frames[0][:, ::-1])),
                    ("s075", cv2.resize(frames[0], (max(2,int(W0*0.75)), max(2,int(H0*0.75))))),
                    ("s125", cv2.resize(frames[0], (int(W0*1.25), int(H0*1.25))))]
        raw = []
        for tag, img in variants:
            dv = det.detect(img)
            for key in ("known_boxes", "unknown_boxes"):
                for b in dv.get(key, []):
                    b = list(b.tolist() if hasattr(b, "tolist") else b)
                    x1, y1, x2, y2 = b[:4]
                    if tag == "flip":
                        x1, x2 = W0 - x2, W0 - x1
                    elif tag == "s075":
                        x1, y1, x2, y2 = x1/0.75, y1/0.75, x2/0.75, y2/0.75
                    elif tag == "s125":
                        x1, y1, x2, y2 = x1/1.25, y1/1.25, x2/1.25, y2/1.25
                    raw.append(((tag == "id"), [float(x1), float(y1), float(x2), float(y2)] + list(b[4:])))
        # admit in order, rejecting a box already explained beyond COVER_MAX by admitted boxes
        # The identity call is the baseline's own frame-0 detection: admit every one of
        # its boxes unconditionally, exactly as the TTA=0 branch does, so the two arms
        # start from an identical proposal set. The coverage gate then applies ONLY to
        # the three EXTRA calls, mirroring TIDE, where a new proposal is gated against
        # the tracks that already exist. Gating the identity boxes against each other
        # would filter proposals the baseline keeps and confound the comparison.
        CM = float(os.environ.get("COVER_MAX", "0.5"))
        cover = np.zeros((H0, W0), np.uint8); boxes = []
        def _clip(b):
            x1, y1, x2, y2 = [int(max(0, v)) for v in b[:4]]
            return x1, y1, min(W0, max(x1+1, x2)), min(H0, max(y1+1, y2))
        n_id = 0
        for i, (is_id, b) in enumerate(raw):          # pass 1: identity call, ungated
            if not is_id: continue
            x1, y1, x2, y2 = _clip(b)
            if (x2-x1)*(y2-y1) <= 0: continue
            cover[y1:y2, x1:x2] = 1
            boxes.append((b, f"t{i}")); n_id += 1
        n_extra = 0
        for i, (is_id, b) in enumerate(raw):          # pass 2: extra calls, coverage-gated
            if is_id: continue
            x1, y1, x2, y2 = _clip(b)
            area = (x2-x1)*(y2-y1)
            if area <= 0: continue
            if int(cover[y1:y2, x1:x2].sum())/area >= CM: continue
            cover[y1:y2, x1:x2] = 1
            boxes.append((b, f"t{i}")); n_extra += 1
        print(f"    [TTA] {len(raw)} raw from 4 calls -> {len(boxes)} admitted ({n_id} identity, {n_extra} extra)", flush=True)
    else:
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
            if ORACLE in (2, 3, 4):
                # Group objects by their anchor frame and propagate outwards in BOTH
                # directions from it, exactly as Algorithm 1 does for a re-detected track.
                _by_anchor = {}
                for _bx, _iid in boxes:
                    _by_anchor.setdefault(int(seed_at[_iid]), []).append((_bx, _iid))
                for _a, _grp in sorted(_by_anchor.items()):
                    sam2.reset_state(st)
                    for _bx, _iid in _grp:
                        if ORACLE == 3:
                            sam2.add_new_mask(st, frame_idx=_a, obj_id=i2o[_iid],
                                              mask=seed_mask[_iid])
                        else:
                            x1, y1, x2, y2 = [float(v) for v in _bx[:4]]
                            x1 = max(0, min(x1, TARGET_SIZE[0]-1)); y1 = max(0, min(y1, TARGET_SIZE[1]-1))
                            x2 = max(x1+1, min(x2, TARGET_SIZE[0])); y2 = max(y1+1, min(y2, TARGET_SIZE[1]))
                            sam2.add_new_points_or_box(st, frame_idx=_a, obj_id=i2o[_iid],
                                                       box=np.array([[x1, y1, x2, y2]], dtype=np.float32))
                    for _rev in (False, True):
                        for fi, oids, masks in sam2.propagate_in_video(st, start_frame_idx=_a, reverse=_rev):
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
            else:
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
        ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0
        for r, c in zip(ri, ci):
            per_gt[c] = (Jsub[r, c] + bF(P[r], gt_ids[c])) / 2
            if Jsub[r, c] > 0.5: nhit += 1
        commit_bij = float(np.mean(per_gt)); prec = nhit/len(P); rec = nhit/len(gt_ids)
    else:
        commit_bij = prec = rec = 0.0

    agg["greedy"].append(greedy); agg["commit_bij"].append(commit_bij); agg["prec"].append(prec)
    agg["rec"].append(rec); agg["ntrk"].append(len(boxes)); agg["ncommit"].append(len(P)); agg["ngt"].append(len(gt_ids))
    per_seq[vid] = dict(greedy=round(greedy, 4), commit_bij=round(commit_bij, 4), prec=round(prec, 3),
                        rec=round(rec, 3), ntrk=len(boxes), ncommit=len(P), ngt=len(gt_ids), nframe=n)
    print(f"  [{si+1:3d}] {vid:22s} greedy={greedy:.3f} commitBij={commit_bij:.3f} P={prec:.2f} R={rec:.2f} "
          f"trk={len(boxes)} commit={len(P)} gt={len(gt_ids)} nf={n} t={time.time()-t0:.0f}s", flush=True)

O = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
gr, cb = O("greedy"), O("commit_bij")
print(f"\n=== MC-VOS on {TAG} ({len(agg['greedy'])} videos, MAXF={MAXF}) ===", flush=True)
print(f"GREEDY best-match J&F : {gr*100:.2f}%   (upper bound)", flush=True)
print(f"COMMITTED-track J&F   : {cb*100:.2f}%   (fair headline; retention {cb/gr*100 if gr else 0:.1f}%)", flush=True)
print(f"committed precision   : {O('prec')*100:.1f}%   recall {O('rec')*100:.1f}%", flush=True)
print(f"tracks/video {O('ntrk'):.1f} -> committed {O('ncommit'):.1f}  (GT objs {O('ngt'):.1f})", flush=True)
out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS,
                       TTA=TTA, ORACLE=int(os.environ.get("ORACLE","0")), OSEEDS=int(os.environ.get("OSEEDS","1")), OBJ_THR=OBJ_THR, COVER_MAX=float(os.environ.get("COVER_MAX","0.5")),
                       SAM2_SIZE=_SZ, SHARD=_SH, NSHARD=_NS),
           summary=dict(n=len(agg["greedy"]), greedy=gr, commit_bij=cb, retention=(cb/gr if gr else 0),
                        precision=O("prec"), recall=O("rec"), mean_tracks=O("ntrk"),
                        mean_commit=O("ncommit"), mean_gt=O("ngt")), per_seq=per_seq)
od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)
print(f"Saved {od}/results.json", flush=True)
