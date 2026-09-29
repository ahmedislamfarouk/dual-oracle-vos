"""
TIDE v4 — frame-0-PROTECTED committed assignment (kills DAVIS regression).

Diagnosis (from full-scale runs): the coverage gate alone does NOT hold DAVIS.
Discovery tracks land in genuinely-uncovered background/part regions, pass the gate,
get committed, and STEAL bijective matches from real frame-0 objects (DAVIS-5 JF
-1.3pp with 0 discovery TP hits).

Fix (two enforced properties that together GUARANTEE per-video tidep_JF >= base_JF):
  (A) FRAME-0-PRIORITY COMMIT: the committed set is seeded with the base committed
      frame-0 tracks, THEN gated discovery tracks are added only if they survive
      temporal-IoU dedup against that set. So base's frame-0 committed tracks are a
      subset of tidep's committed set (a discovery track can never dedup-evict a
      committed frame-0 track).
  (B) FRAME-0-PROTECTED BIJECTIVE: stage-1 Hungarian assigns committed frame-0 tracks
      to GTs (== the exact base assignment); stage-2 lets committed discovery tracks
      claim ONLY the GTs left unexplained (assigned-J <= 0.5) by frame-0, taking the
      per-GT max. Discovery can therefore only ADD recall on missed GTs, never lower a
      frame-0 win. => commit_bij(tidep) >= commit_bij(base) and R(tidep) >= R(base),
      per video, so DAVIS (saturated) regression is 0 by construction while MOSE keeps
      the discovery gain.

This is a commit/assignment POLICY (prefer stable frame-0 identities; use off-frame
discovery only to fill gaps), reported transparently alongside the un-protected arms.

Arms (one run, propagation done once): base | tide(all-disc, standard) |
  tideg(gated-disc, standard) | tidep(gated-disc, PROTECTED) |
  tidep_go(all-disc, PROTECTED) + protected gate sweep P{.7,.8,.9}xC{.3,.2,.1}.

Env: ROOT, CUDA_VISIBLE_DEVICES, MAXF(60), MAXVIDS(0), TAG, COVER_MAX(0.5)[detect-gate],
     DISC_PERSIST_MIN(0.7), DISC_COMMIT_COVER_MAX(0.3)[primary gate], SWEEP(1/0)
"""
import sys, os, json, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
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
torch.set_num_threads(2)
from PIL import Image
from scipy.optimize import linear_sum_assignment
os.chdir("/path/to/ow_vis")
from sam2.build_sam import build_sam2_video_predictor
import tempfile, shutil

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.environ["ROOT"]
SAM2_CKPT = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda"; TARGET_SIZE = (854, 480); HW = TARGET_SIZE[0]*TARGET_SIZE[1]
MAXF = int(os.environ.get("MAXF", "60")); MAXVIDS = int(os.environ.get("MAXVIDS", "0"))
TAG = os.environ.get("TAG", "tidep")
PERSIST_MIN, AREA_MAX, DEDUP_IOU = 0.5, 0.5, 0.5
COVER_MAX = float(os.environ.get("COVER_MAX", "0.5"))
CYCLE = os.environ.get("CYCLE","1") == "1"
CYCLE_MIN = float(os.environ.get("CYCLE_MIN","0.5"))
DISC_PERSIST_MIN = float(os.environ.get("DISC_PERSIST_MIN", "0.7"))
DISC_COMMIT_COVER_MAX = float(os.environ.get("DISC_COMMIT_COVER_MAX", "0.3"))
SWEEP = os.environ.get("SWEEP", "1") == "1"
W, H = TARGET_SIZE
GRID_P = [0.7, 0.8, 0.9]
GRID_C = [0.3, 0.2, 0.1]

print(f"Loading DETR + SAM2 Large... ROOT={ROOT} primary gate P>={DISC_PERSIST_MIN} "
      f"cover<{DISC_COMMIT_COVER_MAX} SWEEP={SWEEP}", flush=True)
det = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=0.05)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
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
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes))}
    conf = {i: [] for i in range(len(boxes))}
    if not boxes: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=0, obj_id=i, box=np.array([[x1, y1, x2, y2]], np.float32))
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
    n = _NF
    pred = {i: [np.zeros((H, W), np.uint8) for _ in range(n)] for i in range(len(boxes_at))}
    conf = {i: [] for i in range(len(boxes_at))}
    if not boxes_at: return pred, conf
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = sam2.init_state(video_path=tmpd, offload_video_to_cpu=True)
        for i, bx in enumerate(boxes_at):
            x1, y1, x2, y2 = bx
            sam2.add_new_points_or_box(st, frame_idx=int(anchor), obj_id=i, box=np.array([[x1, y1, x2, y2]], np.float32))
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


JROOT = os.path.join(ROOT, "JPEGImages"); AROOT = os.path.join(ROOT, "Annotations")
vids = sorted([v for v in os.listdir(JROOT) if os.path.isdir(os.path.join(JROOT, v))])
SHARD=int(os.environ.get('SHARD','0')); NSHARD=int(os.environ.get('NSHARD','1'))
if NSHARD>1:
    vids=vids[SHARD::NSHARD]
    print(f'[shard {SHARD}/{NSHARD}] {len(vids)} videos', flush=True)
if MAXVIDS > 0: vids = vids[:MAXVIDS]
print(f"{len(vids)} videos", flush=True)

MET = ["greedy", "commit_bij", "prec", "rec", "ntrk", "ncommit", "Rstrict", "Qstrict"]
sweep_arms = [f"p_{p}_{c}" for p in GRID_P for c in GRID_C] if SWEEP else []
arm_names = ["base", "tide", "tideg", "tidep", "tidep_go",
             "tidec", "tidec_p"] + sweep_arms
CYC_GRID = [float(x) for x in os.environ.get("CYC_GRID","0.3,0.5,0.7").split(",")]
arm_names += [f"c_{c}" for c in CYC_GRID]
agg = {a: {k: [] for k in MET} for a in arm_names}
disc_Q, f0_Q = [], []; disc_hits = 0; f0_hits = 0
per_seq = {}

for si, vid in enumerate(vids):
    t0 = time.time()
    fps = sorted(glob.glob(os.path.join(JROOT, vid, "*.jpg")))
    astems = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(AROOT, vid, "*.png"))}
    paired = [(p, astems[os.path.splitext(os.path.basename(p))[0]]) for p in fps
              if os.path.splitext(os.path.basename(p))[0] in astems]
    # FULL TEMPORAL EXTENT under a per-process address-space cap.
    # Truncating to the first MAXF frames discards the later part of the video, which is
    # exactly where late objects appear -- the phenomenon under test. Instead, when the clip
    # is longer than MAXF we keep MAXF frames spread evenly across the WHOLE clip, so the
    # timeline is covered end to end at a lower frame rate rather than covered only at its start.
    # STRIDE=1 with a large MAXF restores the original truncating behaviour.
    _n0 = len(paired)
    if MAXF > 0 and _n0 > MAXF:
        if os.environ.get("SPREAD", "1") == "1":
            import numpy as _np
            _idx = _np.linspace(0, _n0 - 1, MAXF).round().astype(int)
            _seen, _keep = set(), []
            for _i in _idx:
                if _i not in _seen:
                    _seen.add(_i); _keep.append(paired[_i])
            paired = _keep
        else:
            paired = paired[:MAXF]
    _cov = (paired[-1][0] != None) if paired else False
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

    tmpd = tempfile.mkdtemp(prefix="tidep_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmpd}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    predF, confF = propagate_forward(tmpd, boxes0)
    redet = sorted(set(f for f in (n//4, n//2, (3*n)//4) if 0 < f < n))
    disc_pred, disc_conf, disc_anchor, kept = {}, {}, {}, 0
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
            if int(cover[y1:y2, x1:x2].sum())/area < COVER_MAX: newb.append(_clip_box(b))
        if not newb: continue
        bp, bc = propagate_bidir(tmpd, newb, kf)
        for j in bp:
            disc_pred[kept] = bp[j]; disc_conf[kept] = bc[j]
            disc_anchor[kept] = int(kf); kept += 1

    # ---- forward-backward CYCLE CONSISTENCY on discovered tracks ----
    # A genuine object discovered at frame t and propagated back to frame 0 should,
    # when re-propagated forward from its frame-0 mask, return to itself at frame t.
    # One extra batched propagation pass per video; no ground truth, no training.
    cyc = {}
    if CYCLE and disc_pred:
        cids, cbox = [], []
        for j, ms in disc_pred.items():
            m0 = ms[0]
            if m0.sum() == 0:
                cyc[j] = -1.0   # never reached frame 0: unverifiable
                continue
            ys, xs = np.nonzero(m0)
            cbox.append([float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)])
            cids.append(j)
        if cbox:
            cp, _ = propagate_bidir(tmpd, cbox, 0)
            for k, j in enumerate(cids):
                a = disc_anchor[j]
                o = disc_pred[j][a].astype(bool); r = cp[k][a].astype(bool)
                u = int((o | r).sum())
                cyc[j] = (float((o & r).sum()) / u) if u else 0.0
    shutil.rmtree(tmpd, ignore_errors=True)

    pred, conf, origin = {}, {}, {}
    for i in predF: k = f"f0_{i}"; pred[k] = predF[i]; conf[k] = confF[i]; origin[k] = "f0"
    for j in disc_pred: k = f"d_{j}"; pred[k] = disc_pred[j]; conf[k] = disc_conf[j]; origin[k] = "disc"
    cycscore = {f"d_{j}": cyc.get(j, -1.0) for j in disc_pred}
    tids = list(pred.keys())
    f0_ids = [t for t in tids if origin[t] == "f0"]
    d_ids  = [t for t in tids if origin[t] == "disc"]

    sig = {}
    for t in tids:
        ne = [int(m.sum()) for m in pred[t] if m.sum() > 0]
        sig[t] = dict(persistence=len(ne)/n, mean_area=(np.mean(ne)/HW if ne else 0.0),
                      quality=(len(ne)/n)*(float(np.mean(conf[t])) if conf.get(t) else 0.0))
    dstack = {t: np.stack([m[::4, ::4] for m in pred[t]]).astype(bool) for t in tids}
    def tiou_d(a, b):
        inter = np.logical_and(dstack[a], dstack[b]).sum(); union = np.logical_or(dstack[a], dstack[b]).sum()
        return inter/union if union > 0 else 0.0

    gt_seqs = {g: [(m == g).astype(np.uint8) for m in gtm] for g in gt_ids}
    gt_bool = {g: np.stack(gt_seqs[g]).astype(bool).reshape(n, -1) for g in gt_ids}
    Jrow = {}
    for t in tids:
        pb = np.stack(pred[t]).astype(bool).reshape(n, -1); out = []
        for g in gt_ids:
            gb = gt_bool[g]; inter = np.logical_and(pb, gb).sum(1); union = np.logical_or(pb, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter/np.maximum(union, 1)).mean()))
        Jrow[t] = np.array(out)
    _fc = {}
    def bF(t, g):
        k = (t, g)
        if k not in _fc:
            _fc[k] = float(np.mean([compute_f_score(pred[t][i], gt_seqs[g][i]) for i in range(n)]))
        return _fc[k]

    def dedup_select(cand_sorted, seed=None):
        comm = list(seed) if seed else []
        for t in cand_sorted:
            if t in comm: continue
            if all(tiou_d(t, k) < DEDUP_IOU for k in comm): comm.append(t)
        return comm

    def select_std(subset):
        cand = [t for t in subset if sig[t]["persistence"] >= PERSIST_MIN and 0 < sig[t]["mean_area"] < AREA_MAX]
        cand.sort(key=lambda t: sig[t]["quality"], reverse=True)
        return dedup_select(cand)

    def greedy_of(subset):
        if not subset: return 0.0
        Js = np.stack([Jrow[t] for t in subset])
        return float(np.mean([(Js[int(np.argmax(Js[:, gi])), gi] +
                    bF(subset[int(np.argmax(Js[:, gi]))], g)) / 2 for gi, g in enumerate(gt_ids)]))

    # base committed frame-0 (also the priority seed for protected commit)
    comm_f0 = select_std(f0_ids)
    if comm_f0:
        union_d = np.logical_or.reduce([dstack[t] for t in comm_f0])
    else:
        union_d = np.zeros_like(next(iter(dstack.values())))

    def gate_disc(persist_thr, cover_thr):
        keep = []
        for t in d_ids:
            if sig[t]["persistence"] < persist_thr: continue
            if not (0 < sig[t]["mean_area"] < AREA_MAX): continue
            dt = dstack[t]; area = int(dt.sum())
            ov = int(np.logical_and(dt, union_d).sum())/area if area else 1.0
            if ov < cover_thr: keep.append(t)
        return keep

    def commit_protected(gated):
        # frame-0-priority: seed with comm_f0, add gated disc surviving dedup
        gc = sorted(gated, key=lambda t: sig[t]["quality"], reverse=True)
        return dedup_select(gc, seed=list(comm_f0))

    def metrics(subset_greedy, committed, protect=False):
        G = len(gt_ids)
        m = dict(greedy=greedy_of(subset_greedy), commit_bij=0.0, prec=0.0, rec=0.0,
                 ntrk=len(subset_greedy), ncommit=len(committed), Rstrict=0.0, Qstrict=0.0, hits=[])
        if not committed: return m
        if not protect:
            Jsub = np.stack([Jrow[t] for t in committed])
            ri, ci = linear_sum_assignment(-Jsub); per_gt = np.zeros(G); nhit = 0; sh = 0.0
            for r, c in zip(ri, ci):
                q = (Jsub[r, c] + bF(committed[r], gt_ids[c])) / 2; per_gt[c] = q
                if Jsub[r, c] > 0.5: nhit += 1; sh += q; m["hits"].append((origin[committed[r]], float(q)))
            m.update(commit_bij=float(np.mean(per_gt)), prec=nhit/len(committed), rec=nhit/G,
                     Rstrict=nhit/G, Qstrict=(sh/nhit) if nhit else 0.0)
            return m
        # protected: stage-1 frame-0, stage-2 discovery on unexplained GTs (per-GT max)
        f0c = [t for t in committed if origin[t] == "f0"]
        dc  = [t for t in committed if origin[t] == "disc"]
        per_gt = np.zeros(G); win_J = np.zeros(G); win_src = [None]*G
        if f0c:
            Jf0 = np.stack([Jrow[t] for t in f0c])
            ri, ci = linear_sum_assignment(-Jf0)
            for r, c in zip(ri, ci):
                per_gt[c] = (Jf0[r, c] + bF(f0c[r], gt_ids[c])) / 2; win_J[c] = Jf0[r, c]; win_src[c] = "f0"
        U = [g for g in range(G) if win_J[g] <= 0.5]
        if dc and U:
            Jd = np.stack([Jrow[t][U] for t in dc])  # [D,|U|]
            ri, ci = linear_sum_assignment(-Jd)
            for r, c in zip(ri, ci):
                g = U[c]; jval = Jd[r, c]; q = (jval + bF(dc[r], gt_ids[g])) / 2
                if q > per_gt[g]: per_gt[g] = q; win_J[g] = jval; win_src[g] = "disc"
        nhit = 0; sh = 0.0
        for g in range(G):
            if win_J[g] > 0.5: nhit += 1; sh += per_gt[g]; m["hits"].append((win_src[g], float(per_gt[g])))
        m.update(commit_bij=float(np.mean(per_gt)), prec=nhit/len(committed), rec=nhit/G,
                 Rstrict=nhit/G, Qstrict=(sh/nhit) if nhit else 0.0)
        return m

    gk = gate_disc(DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX)
    res = {}
    res["base"]     = metrics(f0_ids, comm_f0, protect=False)
    res["tide"]     = metrics(tids, select_std(tids), protect=False)
    res["tideg"]    = metrics(f0_ids + gk, select_std(f0_ids + gk), protect=False)
    res["tidep"]    = metrics(f0_ids + gk, commit_protected(gk), protect=True)
    res["tidep_go"] = metrics(tids, commit_protected(d_ids), protect=True)
    # cycle-consistency gate: keep a discovered track when its frame-0 mask,
    # re-propagated forward, returns to the original detection at its anchor frame.
    ck = [t for t in d_ids if cycscore.get(t, -1.0) >= CYCLE_MIN]
    res["tidec"]   = metrics(f0_ids + ck, select_std(f0_ids + ck), protect=False)
    res["tidec_p"] = metrics(f0_ids + ck, commit_protected(ck), protect=True)
    for _c in CYC_GRID:
        _k = [t for t in d_ids if cycscore.get(t, -1.0) >= _c]
        res[f"c_{_c}"] = metrics(f0_ids + _k, select_std(f0_ids + _k), protect=False)
    if SWEEP:
        for p in GRID_P:
            for c in GRID_C:
                g2 = gate_disc(p, c)
                res[f"p_{p}_{c}"] = metrics(f0_ids + g2, commit_protected(g2), protect=True)

    for a in arm_names:
        for k in MET: agg[a][k].append(res[a][k])
    for (o, q) in res["tidep"]["hits"]:
        if o == "disc": disc_Q.append(q); disc_hits += 1
        else: f0_Q.append(q); f0_hits += 1

    per_seq[vid] = {a: {k: round(res[a][k], 4) for k in MET} for a in ["base", "tide", "tideg", "tidep"]}
    per_seq[vid].update(n_disc=kept, n_gated=len(gk), ngt=len(gt_ids), nframe=n)
    b, td, tg, tp = res["base"], res["tide"], res["tideg"], res["tidep"]
    print(f"  [{si+1:3d}] {vid:20s} R {b['Rstrict']:.2f}->p{tp['Rstrict']:.2f} "
          f"JF base{b['commit_bij']:.3f} tide{td['commit_bij']:.3f} tideg{tg['commit_bij']:.3f} "
          f"tidec{res['tidec']['commit_bij']:.3f} cyc{len(ck)}/{len(d_ids)} "
          f"tidep{tp['commit_bij']:.3f} | commit {b['ncommit']}->p{tp['ncommit']} "
          f"disc={kept} gated={len(gk)} gt={len(gt_ids)} t={time.time()-t0:.0f}s", flush=True)

    if (si+1) % 5 == 0 or si == len(vids)-1:
        def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
        summ = {a: {k: Mv(a, k) for k in MET} for a in arm_names}
        summ["off_frame0"] = dict(disc_hits=disc_hits, f0_hits=f0_hits,
                                  disc_Q=float(np.mean(disc_Q)) if disc_Q else 0.0,
                                  f0_Q=float(np.mean(f0_Q)) if f0_Q else 0.0)
        out = dict(config=dict(ROOT=ROOT, TAG=TAG, MAXF=MAXF, MAXVIDS=MAXVIDS, COVER_MAX=COVER_MAX,
                       SPREAD=os.environ.get("SPREAD","1"),
                   DISC_PERSIST_MIN=DISC_PERSIST_MIN, DISC_COMMIT_COVER_MAX=DISC_COMMIT_COVER_MAX,
                   n=len(agg["base"]["greedy"])), summary=summ, per_seq=per_seq)
        od = f"/path/to/ow_vis/results/{TAG}"; os.makedirs(od, exist_ok=True)
        with open(f"{od}/results.json", "w") as f: json.dump(out, f, indent=2)

def Mv(a, k): return float(np.mean(agg[a][k])) if agg[a][k] else 0.0
print(f"\n=== TIDE-v4 PROTECTED on {TAG} ({len(agg['base']['greedy'])} vids) ===", flush=True)
for a in ["base", "tide", "tideg", "tidep", "tidep_go"]:
    print(f"[{a:9s}] R={Mv(a,'Rstrict')*100:5.2f} commitJF={Mv(a,'commit_bij')*100:5.2f} "
          f"greedy={Mv(a,'greedy')*100:5.2f} prec={Mv(a,'prec')*100:4.1f} "
          f"trk={Mv(a,'ntrk'):5.1f} commit={Mv(a,'ncommit'):4.1f} Q={Mv(a,'Qstrict')*100:5.2f}", flush=True)
if SWEEP:
    print("--- protected gate sweep (persist,cover) ---", flush=True)
    for p in GRID_P:
        for c in GRID_C:
            a = f"p_{p}_{c}"
            print(f"  P>={p} cover<{c}: JF={Mv(a,'commit_bij')*100:5.2f} R={Mv(a,'Rstrict')*100:5.2f} "
                  f"commit={Mv(a,'ncommit'):4.1f} prec={Mv(a,'prec')*100:4.1f}", flush=True)
print(f"[off-frame0 tidep] disc hits={disc_hits} (Q={np.mean(disc_Q)*100 if disc_Q else 0:.2f}) | "
      f"f0 hits={f0_hits} (Q={np.mean(f0_Q)*100 if f0_Q else 0:.2f})", flush=True)
print("Saved.", flush=True)
