"""
Periodic coverage-gap re-detection on YouTube-VOS 2019 val.

Motivated by the SB-4 late-appearing-object analysis: 10.7% of GT objects first
appear AFTER the reference frame and are structurally unsegmentable by single-shot
frame-0 detection (~5.4pp of the 15.17pp drop). This experiment IMPLEMENTS and
MEASURES the "periodic per-frame re-detection pass" the paper had left as future
work, and isolates its effect with a strictly-paired A/B comparison in one run:

  Arm A (baseline)  = frame-0 all-boxes (objectness>0.05), the 68.16% config.
  Arm B (periodic)  = Arm A PLUS re-detection at quartile frames {n/4, n/2, 3n/4};
                      at each re-detection frame only boxes whose region is NOT
                      already covered by an existing track's mask (mask-coverage
                      gating) are kept and seeded as new forward-propagated tracks.

Under greedy best-match (the standard protocol) Arm B >= Arm A always, so the
delta is exactly the recovered structural late-appearance tax. We also count, for
the late GT objects specifically, how many go from missed (best J<0.1) to
recovered (best J>0.5).

Run via:  python3 /path/to/ow_vis/launcher.py /tmp/eval_periodic_redetect_ytvos.py
Env: SUBSET = late | ontime | all   (default late)
     MAXVIDS = int cap (default 0 = no cap)
"""
import sys, os, json, glob, tempfile, shutil, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/path/to/ow_vis/sam2")
sys.path.insert(0, "/path/to/ow_vis/src")
os.chdir("/path/to/ow_vis")

# transformers crashes parsing torchao's version (metadata None); disable it
try:
    import transformers.utils.import_utils as _iu
    _iu._torchao_available = False
except Exception:
    pass

import torch, cv2, numpy as np
from PIL import Image

# pick the freest GPU (nvidia-smi CLI broken; torch works)
best, bestfree = 0, -1
for d in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(d)
    if free > bestfree:
        bestfree, best = free, d
torch.cuda.set_device(best)
DEVICE = f"cuda:{best}"
print(f"Using {DEVICE}, free={bestfree/1e9:.1f}GB", flush=True)

from owdetr_detector import OpenWorldDetector
from metrics import compute_j_score, compute_f_score
from sam2.build_sam import build_sam2_video_predictor

YTVOS_ROOT = "/path/to/ytvos2019/valid"
SAM2_CKPT  = "/path/to/ow_vis/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
TARGET_SIZE = (854, 480)
TIER1 = 0.05
UNCOVERED_MIN = 0.70   # a re-detected box is "new" if >=70% of its area is uncovered
OUT_DIR = "/path/to/ow_vis/results/periodic_redetect_ytvos"
os.makedirs(OUT_DIR, exist_ok=True)

SUBSET = os.environ.get("SUBSET", "late")
MAXVIDS = int(os.environ.get("MAXVIDS", "0"))

with open(os.path.join(YTVOS_ROOT, "meta.json")) as f:
    meta = json.load(f)["videos"]


def video_first(info):
    return min(int(o["frames"][0]) for o in info["objects"].values())


def n_late(info):
    vf = video_first(info)
    return sum(1 for o in info["objects"].values() if int(o["frames"][0]) > vf)


late_vids   = sorted([v for v, info in meta.items() if n_late(info) >= 1])
ontime_vids = sorted([v for v, info in meta.items() if n_late(info) == 0])
if SUBSET == "late":
    video_ids = late_vids
elif SUBSET == "ontime":
    import random; random.seed(42); video_ids = sorted(random.sample(ontime_vids, min(89, len(ontime_vids))))
else:
    video_ids = sorted(meta.keys())
if MAXVIDS > 0:
    video_ids = video_ids[:MAXVIDS]
print(f"SUBSET={SUBSET}: {len(video_ids)} videos "
      f"(late pool={len(late_vids)}, ontime pool={len(ontime_vids)})", flush=True)

det  = OpenWorldDetector(device=DEVICE, unknown_threshold=0.5, objectness_threshold=TIER1)
sam2 = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=DEVICE)
print("Models ready.", flush=True)


def get_boxes(frame_rgb):
    res = det.detect(frame_rgb)
    return list(res.get("known_boxes", [])) + list(res.get("unknown_boxes", []))


def propagate(tmpdir, boxes_at_frame, n, h, w):
    """boxes_at_frame: list of (box, frame_idx). Returns list of per-track mask lists."""
    pred = {i: [np.zeros((h, w), np.uint8) for _ in range(n)] for i in range(len(boxes_at_frame))}
    if not boxes_at_frame:
        return pred
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = sam2.init_state(video_path=tmpdir, offload_video_to_cpu=True)
        for i, (box, fidx) in enumerate(boxes_at_frame):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w)); y2 = max(y1 + 1, min(y2, h))
            sam2.add_new_points_or_box(state, frame_idx=int(fidx), obj_id=i,
                                       box=np.array([[x1, y1, x2, y2]], np.float32))
        # propagate forward from the earliest seeded frame
        start = min(int(f) for _, f in boxes_at_frame)
        for fi, oids, masks in sam2.propagate_in_video(state, start_frame_idx=start):
            if fi >= n: break
            for oi, ml in zip(oids, masks):
                m = (ml[0].cpu().numpy() > 0).astype(np.uint8)
                if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                pred[int(oi)][fi] = m
        sam2.reset_state(state)
    return pred


def j_matrix(gt_list, pred_tracks):
    """Returns J (G x T) IoU matrix (cheap) and the track-id list. Boundary-F is
    computed lazily only for selected pairs (it is ~100x costlier than IoU)."""
    G, tids = len(gt_list), list(pred_tracks.keys())
    J = np.zeros((G, len(tids)))
    for gi, (_, of) in enumerate(gt_list):
        for ti, tid in enumerate(tids):
            tm = pred_tracks[tid]
            J[gi, ti] = np.mean([compute_j_score(tm[fi], g) for fi, g in of])
    return J, tids


def _f_for(gt_list, pred_tracks, tids, gi, ti):
    of = gt_list[gi][1]; tm = pred_tracks[tids[ti]]
    return float(np.mean([compute_f_score(tm[fi], g) for fi, g in of]))


def greedy_scores(gt_list, pred_tracks, J, tids):
    """Per-GT best track (track may match several GTs). F only for chosen pairs."""
    G = len(gt_list)
    if J.shape[1] == 0:
        return np.zeros(G), np.zeros(G)
    bi = J.argmax(axis=1)
    bestJ = J[np.arange(G), bi]
    jf = np.array([(bestJ[gi] + _f_for(gt_list, pred_tracks, tids, gi, bi[gi])) / 2.0
                   for gi in range(G)])
    return jf, bestJ


def bijective_scores(gt_list, pred_tracks, J, tids):
    """One-to-one Hungarian GT<->track maximising IoU; unassigned GTs score 0.
    F computed only for assigned pairs. Returns per-GT (J+F)/2 and assigned-J."""
    G, T = J.shape
    jf = np.zeros(G); aj = np.zeros(G)
    if T == 0:
        return jf, aj
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-J)
    except Exception:  # greedy one-to-one fallback
        rows, cols, ur, uc = [], [], set(), set()
        for r, c in np.dstack(np.unravel_index(np.argsort(-J, axis=None), J.shape))[0]:
            if r in ur or c in uc: continue
            ur.add(r); uc.add(c); rows.append(r); cols.append(c)
        rows, cols = np.array(rows), np.array(cols)
    for r, c in zip(rows, cols):
        aj[r] = J[r, c]
        jf[r] = (J[r, c] + _f_for(gt_list, pred_tracks, tids, int(r), int(c))) / 2.0
    return jf, aj


def eval_video(vid):
    img_paths = sorted(glob.glob(os.path.join(YTVOS_ROOT, "JPEGImages", vid, "*.jpg")))
    ann_paths = sorted(glob.glob(os.path.join(YTVOS_ROOT, "Annotations", vid, "*.png")))
    if not img_paths or not ann_paths: return None
    frames, fnames = [], []
    for p in img_paths:
        im = cv2.imread(p)
        if im is None: continue
        frames.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), TARGET_SIZE))
        fnames.append(os.path.splitext(os.path.basename(p))[0])
    if not frames: return None
    h, w = frames[0].shape[:2]; n = len(frames)
    fti = {f: i for i, f in enumerate(fnames)}

    gt_by_frame, gt_ids = {}, set()
    for p in ann_paths:
        fn = os.path.splitext(os.path.basename(p))[0]
        m = cv2.resize(np.array(Image.open(p)), TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        gt_by_frame[fn] = m; gt_ids.update(int(x) for x in np.unique(m[m > 0]))
    gt_ids = sorted(gt_ids)
    if not gt_ids: return None

    # per-GT first-appearance frame index (in our sampled-frame index space)
    info = meta[vid]; vf = video_first(info)
    gt_first_idx = {}
    for fn, mk in gt_by_frame.items():
        for g in np.unique(mk[mk > 0]):
            g = int(g)
            if g not in gt_first_idx and fn in fti:
                gt_first_idx[g] = min(gt_first_idx.get(g, 10**9), fti[fn])
            elif fn in fti:
                gt_first_idx[g] = min(gt_first_idx[g], fti[fn])
    late_gt = {int(oid) for oid, o in info["objects"].items() if int(o["frames"][0]) > vf}

    tmp = tempfile.mkdtemp(prefix=f"per_{vid}_")
    for i, fr in enumerate(frames):
        cv2.imwrite(f"{tmp}/{i:05d}.jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    # ---- Arm A: frame-0 all-boxes ----
    boxes0 = get_boxes(frames[0])
    predA = propagate(tmp, [(b, 0) for b in boxes0], n, h, w)
    n0 = len(boxes0)

    # ---- Periodic re-detection: keep only uncovered boxes at scheduled frames ----
    # Schedule: REDET_STRIDE=0 (default) -> quartile periodic {n/4,n/2,3n/4};
    # REDET_STRIDE=k>0 -> DENSE re-detection at every k-th frame {k,2k,3k,...}
    # (k=1 == per-frame detection, the online-VIS standard) for the MAJOR-4 baseline.
    _stride = int(os.environ.get("REDET_STRIDE", "0"))
    if _stride > 0:
        redet_frames = sorted(set(f for f in range(_stride, n, _stride) if 0 < f < n))
    else:
        redet_frames = sorted(set(f for f in (n // 4, n // 2, (3 * n) // 4) if 0 < f < n))
    new_boxes = []   # (box, frame_idx)
    for kf in redet_frames:
        # union of existing-track coverage at frame kf
        cover = np.zeros((h, w), np.uint8)
        for t in predA.values():
            cover |= t[kf]
        for b in get_boxes(frames[kf]):
            x1, y1, x2, y2 = [int(max(0, v)) for v in b[:4]]
            x2 = min(w, max(x1 + 1, x2)); y2 = min(h, max(y1 + 1, y2))
            area = (x2 - x1) * (y2 - y1)
            if area <= 0: continue
            covered = int(cover[y1:y2, x1:x2].sum())
            if 1.0 - covered / area >= UNCOVERED_MIN:
                new_boxes.append((b, kf))

    predB_new = propagate(tmp, new_boxes, n, h, w) if new_boxes else {}
    shutil.rmtree(tmp, ignore_errors=True)

    # combined Arm B tracks = predA + predB_new (renumbered)
    predB = dict(predA)
    off = n0
    for k, v in predB_new.items():
        predB[off + k] = v

    # ---- build per-GT frame lists, then score under BOTH protocols ----
    gt_list, gt_order = [], []
    for g in gt_ids:
        of = [(fti[fn], (gt_by_frame[fn] == g).astype(np.uint8))
              for fn in gt_by_frame if g in np.unique(gt_by_frame[fn]) and fn in fti]
        if of:
            gt_list.append((g, of)); gt_order.append(g)
    if not gt_list: return None

    JA, tidsA = j_matrix(gt_list, predA)
    JB, tidsB = j_matrix(gt_list, predB)
    g_jfA, g_bjA = greedy_scores(gt_list, predA, JA, tidsA)
    g_jfB, g_bjB = greedy_scores(gt_list, predB, JB, tidsB)
    b_jfA, b_ajA = bijective_scores(gt_list, predA, JA, tidsA)
    b_jfB, b_ajB = bijective_scores(gt_list, predB, JB, tidsB)

    # late-object recovery (protocol-robust): bijective assigned-J 0.10 -> 0.50
    late_total = late_recovered = 0
    for gi, g in enumerate(gt_order):
        if g in late_gt:
            late_total += 1
            if b_ajA[gi] < 0.10 and b_ajB[gi] > 0.50:
                late_recovered += 1

    return {
        # greedy protocol (candidate-flooding-inflated; reported as upper bound)
        "greedy_JF_A": round(float(np.mean(g_jfA)), 4),
        "greedy_JF_B": round(float(np.mean(g_jfB)), 4),
        # bijective protocol (HONEST headline: junk tracks go unassigned)
        "bij_JF_A": round(float(np.mean(b_jfA)), 4),
        "bij_JF_B": round(float(np.mean(b_jfB)), 4),
        "bij_delta": round(float(np.mean(b_jfB) - np.mean(b_jfA)), 4),
        "n_gt": len(gt_order),
        "n_track_A": int(JA.shape[1]), "n_track_B": int(JB.shape[1]),  # precision cost
        "n_box_f0": n0, "n_redet_new": len(new_boxes),
        "late_total": late_total, "late_recovered": late_recovered}


results = {}
t0 = time.time()
for i, vid in enumerate(video_ids):
    r = eval_video(vid)
    if r:
        results[vid] = r
        print(f"[{i+1}/{len(video_ids)}] {vid}  bij {r['bij_JF_A']:.4f}->{r['bij_JF_B']:.4f} "
              f"(d={r['bij_delta']:+.4f}) | greedy {r['greedy_JF_A']:.4f}->{r['greedy_JF_B']:.4f} | "
              f"trk {r['n_track_A']}->{r['n_track_B']} new={r['n_redet_new']} "
              f"late_rec={r['late_recovered']}/{r['late_total']}", flush=True)
    if results:
        bA = [v["bij_JF_A"] for v in results.values()]; bB = [v["bij_JF_B"] for v in results.values()]
        gA = [v["greedy_JF_A"] for v in results.values()]; gB = [v["greedy_JF_B"] for v in results.values()]
        agg = {"per_video": results, "n_videos": len(results), "subset": SUBSET,
               "bij_mean_A": round(float(np.mean(bA)), 4), "bij_mean_B": round(float(np.mean(bB)), 4),
               "bij_delta": round(float(np.mean(bB) - np.mean(bA)), 4),
               "greedy_mean_A": round(float(np.mean(gA)), 4), "greedy_mean_B": round(float(np.mean(gB)), 4),
               "greedy_delta": round(float(np.mean(gB) - np.mean(gA)), 4),
               "mean_tracks_A": round(float(np.mean([v["n_track_A"] for v in results.values()])), 2),
               "mean_tracks_B": round(float(np.mean([v["n_track_B"] for v in results.values()])), 2),
               "late_recovered": sum(v["late_recovered"] for v in results.values()),
               "late_total": sum(v["late_total"] for v in results.values())}
        with open(f"{OUT_DIR}/results_{SUBSET}.json", "w") as fp:
            json.dump(agg, fp, indent=2)

bA = [v["bij_JF_A"] for v in results.values()]; bB = [v["bij_JF_B"] for v in results.values()]
gA = [v["greedy_JF_A"] for v in results.values()]; gB = [v["greedy_JF_B"] for v in results.values()]
try:
    from scipy import stats
    w, p = stats.wilcoxon(bB, bA, alternative="greater") if any(b > a for b, a in zip(bB, bA)) else (0, 1.0)
    sig = f"Wilcoxon bij B>A: W={w:.1f} p={p:.3e}"
except Exception as e:
    sig = f"(scipy unavailable: {e})"
print(f"\n=== PERIODIC RE-DETECT [{SUBSET}] {len(results)} videos ===\n"
      f"  BIJECTIVE (honest): {np.mean(bA):.4f} -> {np.mean(bB):.4f}  "
      f"(+{100*(np.mean(bB)-np.mean(bA)):.2f}pp)\n"
      f"  greedy   (inflated): {np.mean(gA):.4f} -> {np.mean(gB):.4f}  "
      f"(+{100*(np.mean(gB)-np.mean(gA)):.2f}pp)\n"
      f"  tracks/video: {np.mean([v['n_track_A'] for v in results.values()]):.1f} -> "
      f"{np.mean([v['n_track_B'] for v in results.values()]):.1f} (precision cost)\n"
      f"  late objects recovered (bij J<0.1->>0.5): "
      f"{sum(v['late_recovered'] for v in results.values())}/{sum(v['late_total'] for v in results.values())}\n"
      f"  {sig} | {time.time()-t0:.0f}s", flush=True)
