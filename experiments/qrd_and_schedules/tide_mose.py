"""
TIDE — Temporal Identity-consolidated Discovery Ensemble.

Goal: raise first-frame DISCOVERY recall R (fraction of GT objects committed as
true positives, J>0.5) on crowded/occluded scenes (MOSE) WITHOUT flooding tracks.

Pipeline (built on sq_generic.py's DETR->SAM2-Large + committed-track selection):
  1. Baseline arm: frame-0 DETR boxes -> SAM2 forward propagate (== sq_generic).
  2. Discovery arm: at sparse frames {n/4, n/2, 3n/4} re-run DETR; keep a box only
     if COVERAGE-GATED (< COVER_MAX of its area covered by union of existing track
     masks at that frame). Each surviving box is propagated BIDIRECTIONALLY from its
     anchor (forward to n-1 AND backward to 0) -> one full-video track. Backward is
     essential: MOSE objects exist at frame 0 but are undetected there due to crowd.
  3. Consolidation (novel operator): merge a discovery track into an existing track
     if same object, by temporal-mask-IoU (and optionally DINOv2 masked-prototype
     cosine). Keep the higher-quality (persistence*conf) track. Kills the flood.
  4. Commit (persistence + area + temporal-IoU dedup as in sq_generic) and score
     greedy + committed-bijective J&F with R/Q instrumentation, tagging each track
     as frame-0 ('f0') or discovery ('disc') so we can report off-frame-0 Q.

Emits, in ONE run, three arms for the flood-killer ablation:
   base  = frame-0 only (reproduces sq_generic)
   tide0 = base + discovery, NO consolidation (raw flood)
   tide  = base + discovery + consolidation

Env: ROOT, CUDA_VISIBLE_DEVICES, MAXF (60), MAXVIDS (0=all), TAG,
     COVER_MAX (0.5), MERGE_IOU (0.5), USE_DINO (0/1), DINO_IOU (0.2), DINO_COS (0.6)
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
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
TAG = os.environ.get("TAG", "tide")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5
COVER_MAX = float(os.environ.get("COVER_MAX", "0.5"))   # keep box if covered frac < this
MERGE_IOU = float(os.environ.get("MERGE_IOU", "0.5"))   # temporal-IoU merge threshold
USE_DINO  = os.environ.get("USE_DINO", "0") == "1"
DINO_IOU  = float(os.environ.get("DINO_IOU", "0.2"))
DINO_COS  = float(os.environ.get("DINO_COS", "0.6"))
W, H = TARGET_SIZE

def temporal_iou(a, b):
    inter = union = 0
    for x, y in zip(a, b):
        inter += np.logical_and(x, y).sum(); union += np.logical_or(x, y).sum()
    return inter/union if union > 0 else 0.0

print(f"Loading DETR + SAM2 Large... ROOT={ROOT} USE_DINO={USE_DINO} COVER_MAX={COVER_MAX} MERGE_IOU={MERGE_IOU}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
dino = None
if USE_DINO:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).eval().to(DEVICE)
    for p in dino.parameters(): p.requires_grad_(False)
print("Models loaded.", flush=True)


def get_boxes(frame_rgb):
    d = det.detect(frame_rgb)
    out = []
    for b in list(d.get("known_boxes", [])) + list(d.get("unknown_boxes", [])):
        out.append(list(b.tolist() if hasattr(b, "tolist") else b))
    return out


def _clip_box(bx):
    x1, y1, x2, y2 = [float(v) for v in bx[:4]]
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(x1+1, min(x2, W)); y2 = max(y1+1, min(y2, H))
    return x1, y1, x2, y2


def propagate_forward(tmpd, boxes):
    """boxes: list of clipped boxes seeded at frame 0. Returns {i:[mask]*n}, {i:[conf]}."""
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes))}
    conf = {i: [] for i in range(len(boxes))}
    if not boxes: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i,
                                       box=np.array([[x1, y1, x2, y2]], np.float32))
        for fi, oids, masks in sam2.propagate_in_video(st):
            if fi >= n: break
            for oid, ml in zip(oids, masks):
                logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                pred[int(oid)][fi] = m
                pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf


def propagate_bidir(tmpd, boxes_at, anchor):
    """boxes_at: list of clipped boxes all seeded at frame `anchor`. Propagate
    forward (anchor..n-1) AND backward (anchor..0). Returns {i:[mask]*n},{i:[conf]}."""
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes_at))}
    conf = {i: [] for i in range(len(boxes_at))}
    if not boxes_at: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes_at):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i,
                                       box=np.array([[x1, y1, x2, y2]], np.float32))
        for reverse in (False, True):
            for fi, oids, masks in sam2.propagate_in_video(st, start_frame_idx=int(anchor), reverse=reverse):
                if fi < 0 or fi >= n: continue
                for oid, ml in zip(oids, masks):
                    logit = ml[0].float().cpu().numpy(); m = (logit > 0).astype(np.uint8)
                    if m.shape != (H, W): m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    pred[int(oid)][fi] = m
                    pos = logit[logit > 0]; conf[int(oid)].append(float(pos.mean()) if pos.size else 0.0)
        sam2.reset_state(st)
    return pred, conf


# ---- DINOv2 masked prototype ----
_dino_cache = {}
def dino_patch(frame_idx, frame_rgb):
    if frame_idx in _dino_cache: return _dino_cache[frame_idx]
    Hd = (H // 14) * 14; Wd = (W // 14) * 14
    im = cv2.resize(frame_rgb, (Wd, Hd)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.inference_mode():
        f = dino.forward_features(t)["x_norm_patchtokens"][0]  # [Np, C]
    gh, gw = Hd // 14, Wd // 14
    f = f.reshape(gh, gw, -1)  # grid of features
    _dino_cache[frame_idx] = (f, gh, gw)
    return _dino_cache[frame_idx]


def dino_prototype(masks, frames):
    """masked-pooled DINOv2 feature averaged over up to 4 highest-area frames."""
    areas = [(int(m.sum()), i) for i, m in enumerate(masks) if m.sum() > 0]
    if not areas: return None
    areas.sort(reverse=True)
    protos = []
    for _, i in areas[:4]:
        f, gh, gw = dino_patch(i, frames[i])
        mk = cv2.resize(masks[i], (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
        if mk.sum() == 0: continue
        fm = f[torch.from_numpy(mk).to(f.device)]
        protos.append(fm.mean(0))
    if not protos: return None
    v = torch.stack(protos).mean(0)
    return torch.nn.functional.normalize(v, dim=0)


JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

ARMS = ["base", "tide0", "tide"]
agg = {a: {k: [] for k in ["greedy", "commit_bij", "prec", "rec", "ntrk", "ncommit",
                            "Rstrict", "Qstrict"]} for a in ARMS}
# off-frame-0 discovery accounting (tide arm)
disc_Q = []      # Q of committed-bijective TP hits attributed to discovery tracks
f0_Q = []        # Q of TP hits attributed to frame-0 tracks
disc_hits = 0; f0_hits = 0
per_seq = {}


def score_arm(pred, conf, origin, gtm, gt_ids, n):
    """pred: {tid:[mask]*n}, conf:{tid:[..]}, origin:{tid:'f0'|'disc'}.
    Returns metrics dict + committed list + per-committed origin/Q for the tide arm."""
    tids = list(pred.keys())
    sig = {}
    for t in tids:
        ne = [int(m.sum()) for m in pred[t] if m.sum() > 0]
        sig[t] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                      quality=(len(ne)/n)*(float(np.mean(conf[t])) if conf.get(t) else 0.0))
    cand = [t for t in tids if sig[t]["persistence"] >= PERSIST_MIN and 0 < sig[t]["mean_area"] < AREA_MAX]
    cand.sort(key=lambda t: sig[t]["quality"], reverse=True)
    committed = []
    for t in cand:
        if all(temporal_iou(pred[t], pred[k]) < DEDUP_IOU for k in committed): committed.append(t)

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    def track_J(t):
        pb = np.stack(pred[t]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        return out
    _fc = {}
    def bF(t, g):
        k = (t, g)
        if k not in _fc:
            _fc[k] = float(np.mean([compute_f_score(pred[t][i], gt_seqs[g][i]) for i in range(n)]))
        return _fc[k]
    Jall = np.array([track_J(t) for t in tids]) if tids else np.zeros((0, len(gt_ids)))
    g_jf = []
    for gi, g in enumerate(gt_ids):
        if Jall.shape[0] == 0: g_jf.append(0.0); continue
        bt = int(np.argmax(Jall[:, gi])); g_jf.append((Jall[bt, gi] + bF(tids[bt], g)) / 2)
    greedy = float(np.mean(g_jf)) if g_jf else 0.0

    info = dict(greedy=greedy, commit_bij=0.0, prec=0.0, rec=0.0, ntrk=len(tids),
                ncommit=len(committed), Rstrict=0.0, Qstrict=0.0,
                hits_origin=[])  # list of (origin, Q) for TP hits
    if committed:
        pidx = [tids.index(t) for t in committed]; Jsub = Jall[pidx]
        ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(len(gt_ids)); nhit = 0; sumhit = 0.0
        for r, c in zip(ri, ci):
            q = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2
            per_gt[c] = q
            if Jsub[r, c] > 0.5:
                nhit += 1; sumhit += q
                info["hits_origin"].append((origin[committed[r]], float(q)))
        info["commit_bij"] = float(np.mean(per_gt)); info["prec"] = nhit/len(committed)
        info["rec"] = nhit/len(gt_ids); info["Rstrict"] = nhit/len(gt_ids)
        info["Qstrict"] = (sumhit/nhit) if nhit else 0.0
    return info


for si, vid in enumerate(vids):
    t0 = time.time()
    fps = sorted(glob.glob(os.path.join(JROOT, vid, "*.jpg")))
    astems = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(AROOT, vid, "*.png"))}
    paired = [(p, astems[os.path.splitext(os.path.basename(p))[0]]) for p in fps
              if os.path.splitext(os.path.basename(p))[0] in astems]
    paired = paired[:MAXF]
    if len(paired) < 2:
        print(f"  [{si+1}] {vid}: <2 frames, skip", flush=True); continue
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB), TARGET_SIZE) for fp, _ in paired]
    gtm = [cv2.resize(np.array(Image.open(ap)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST) for _, ap in paired]
    n = len(frames); _NF = n
    gt_ids = sorted({int(v) for m in gtm for v in np.unique(m[m > 0])})
    if not gt_ids:
        print(f"  [{si+1}] {vid}: no GT, skip", flush=True); continue

    boxes0 = [_clip_box(b) for b in get_boxes(frames[0])]
    if not boxes0:
        print(f"  [{si+1}] {vid}: no det f0, skip", flush=True); continue

    tmpd = tempfile.mkdtemp(prefix="tide_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    _dino_cache.clear()

    # ---- Arm base: frame-0 forward ----
    predF, confF = propagate_forward(tmpd, boxes0)

    # ---- Discovery arm: sparse re-detection, coverage-gated, bidirectional ----
    redet = sorted(set(f for f in (n//4, n//2, (3*n)//4) if 0 < f < n))
    disc_pred, disc_conf, kept = {}, {}, 0
    for kf in redet:
        cover = np.zeros((H, W), np.uint8)
        for t in predF.values(): cover |= t[kf]
        for t in disc_pred.values(): cover |= t[kf]
        newb = []
        for b in get_boxes(frames[kf]):
            x1, y1, x2, y2 = [int(round(v)) for v in _clip_box(b)]
            x2 = min(W, max(x1+1, x2)); y2 = min(H, max(y1+1, y2))
            area = (x2-x1)*(y2-y1)
            if area <= 0: continue
            covfrac = int(cover[y1:y2, x1:x2].sum())/area
            if covfrac < COVER_MAX: newb.append(_clip_box(b))
        if not newb: continue
        bp, bc = propagate_bidir(tmpd, newb, kf)
        for j in bp:
            disc_pred[kept] = bp[j]; disc_conf[kept] = bc[j]; kept += 1
    shutil.rmtree(tmpd, ignore_errors=True)

    # persistence*conf quality for merge tie-breaking
    def qual(masks, cf):
        ne = [1 for m in masks if m.sum() > 0]
        return (len(ne)/n) * (float(np.mean(cf)) if cf else 0.0)

    # ---- pool assembly ----
    # base arm pool
    base_pred = {f"f0_{i}": predF[i] for i in predF}
    base_conf = {f"f0_{i}": confF[i] for i in predF}
    base_orig = {k: "f0" for k in base_pred}

    # tide0 = base + all discovery (no consolidation)
    tide0_pred = dict(base_pred); tide0_conf = dict(base_conf); tide0_orig = dict(base_orig)
    for j in disc_pred:
        tide0_pred[f"d_{j}"] = disc_pred[j]; tide0_conf[f"d_{j}"] = disc_conf[j]; tide0_orig[f"d_{j}"] = "disc"

    # tide = base + discovery WITH consolidation (merge dup discovery into existing)
    tide_pred = dict(base_pred); tide_conf = dict(base_conf); tide_orig = dict(base_orig)
    # precompute dino prototypes for base tracks lazily
    proto = {}
    def get_proto(key, masks):
        if not USE_DINO: return None
        if key not in proto: proto[key] = dino_prototype(masks, frames)
        return proto[key]
    n_merged = 0
    for j in disc_pred:
        dmask = disc_pred[j]; dq = qual(dmask, disc_conf[j])
        best_key, best_iou = None, 0.0
        for k in list(tide_pred.keys()):
            iou = temporal_iou(dmask, tide_pred[k])
            if iou > best_iou: best_iou, best_key = iou, k
        merge = False
        if best_key is not None:
            if best_iou >= MERGE_IOU:
                merge = True
            elif USE_DINO and best_iou >= DINO_IOU:
                pd = get_proto(f"d_{j}", dmask); pk = get_proto(best_key, tide_pred[best_key])
                if pd is not None and pk is not None and float((pd*pk).sum()) >= DINO_COS:
                    merge = True
        if merge:
            n_merged += 1
            kq = qual(tide_pred[best_key], tide_conf[best_key])
            if dq > kq:  # keep discovery track (higher quality), drop existing
                tide_pred[best_key] = dmask; tide_conf[best_key] = disc_conf[j]
                # origin becomes disc only if it was f0 and disc is genuinely better -> keep as disc
                tide_orig[best_key] = tide_orig[best_key] if tide_orig[best_key] == "disc" else "disc"
            # else keep existing, drop discovery
        else:
            tide_pred[f"d_{j}"] = dmask; tide_conf[f"d_{j}"] = disc_conf[j]; tide_orig[f"d_{j}"] = "disc"

    res = {}
    res["base"]  = score_arm(base_pred,  base_conf,  base_orig,  gtm, gt_ids, n)
    res["tide0"] = score_arm(tide0_pred, tide0_conf, tide0_orig, gtm, gt_ids, n)
    res["tide"]  = score_arm(tide_pred,  tide_conf,  tide_orig,  gtm, gt_ids, n)

    for a in ARMS:
        r = res[a]
        for k in agg[a]:
            agg[a][k].append(r[k])
    for (orig, q) in res["tide"]["hits_origin"]:
        if orig == "disc": disc_Q.append(q); disc_hits += 1
        else: f0_Q.append(q); f0_hits += 1

    per_seq[vid] = {a: {k: round(res[a][k], 4) for k in ["greedy", "commit_bij", "prec", "rec",
                    "ntrk", "ncommit", "Rstrict", "Qstrict"]} for a in ARMS}
    per_seq[vid]["n_disc"] = kept; per_seq[vid]["n_merged"] = n_merged
    per_seq[vid]["ngt"] = len(gt_ids); per_seq[vid]["nframe"] = n
    b, t0a, ti = res["base"], res["tide0"], res["tide"]
    print(f"  [{si+1:3d}] {vid:20s} R {b['Rstrict']:.2f}->{ti['Rstrict']:.2f} "
          f"JF {b['commit_bij']:.3f}->{ti['commit_bij']:.3f} | "
          f"trk {b['ntrk']}->{t0a['ntrk']}->{ti['ntrk']} commit {b['ncommit']}->{ti['ncommit']} "
          f"disc={kept} merged={n_merged} gt={len(gt_ids)} t={time.time()-t0:.0f}s", flush=True)

    if (si+1) % 5 == 0 or si == len(vids)-1:
        def M(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
        summ = {a: {k: M(a, k) for k in agg[a]} for a in ARMS}
        summ["off_frame0"] = dict(disc_hits=disc_hits, f0_hits=f0_hits,
                                  disc_Q=float(np.mean(disc_Q)) if disc_Q else 0.0,
                                  f0_Q=float(np.mean(f0_Q)) if f0_Q else 0.0)
        out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS,
                   COVER_MAX=COVER_MAX, MERGE_IOU=MERGE_IOU, USE_DINO=USE_DINO,
                   DINO_IOU=DINO_IOU, DINO_COS=DINO_COS, n=len(agg["base"]["greedy"])),
                   summary=summ, per_seq=per_seq)
        od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)

def M(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
print(f"\n=== TIDE on {TAG} ({len(agg['base']['greedy'])} vids, MAXF={MAXF}) ===", flush=True)
for a in ARMS:
    print(f"[{a:5s}] Rstrict={M(a,'Rstrict')*100:5.2f} commitJF={M(a,'commit_bij')*100:5.2f} "
          f"greedy={M(a,'greedy')*100:5.2f} prec={M(a,'prec')*100:4.1f} rec={M(a,'rec')*100:4.1f} "
          f"trk={M(a,'ntrk'):5.1f} commit={M(a,'ncommit'):4.1f} Qstrict={M(a,'Qstrict')*100:5.2f}", flush=True)
print(f"[off-frame0] discovery TP hits={disc_hits} (Q={np.mean(disc_Q)*100 if disc_Q else 0:.2f}) | "
      f"frame0 TP hits={f0_hits} (Q={np.mean(f0_Q)*100 if f0_Q else 0:.2f})", flush=True)
print("Saved results.", flush=True)
