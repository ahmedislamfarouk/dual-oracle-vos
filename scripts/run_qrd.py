#!/usr/bin/env python3
"""Frame-0 baseline and QRD on a dataset, paired per video.

Arms written for every video:
    frame0     detector on frame 0 only
    qrd        frame 0 plus quartile re-detection (Algorithm 1)
    qrd_gated  QRD with the stricter precision gates on discovered tracks (reported as an ablation)

Example (capped MOSE, as in the paper):
    python scripts/run_qrd.py --root data/mose_subset/train --max-frames 60 --out results/runs/qrd_mose.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dovos.runtime import common_parser, configure, write_json

GATE_PERSISTENCE = 0.7     # precision gates: stricter persistence for discovered tracks ...
GATE_COVERAGE = 0.3        # ... and at most this fraction already covered by committed frame-0 tracks


def main():
    p = common_parser(__doc__.split("\n")[0])
    p.add_argument("--cover-max", type=float, default=0.5, help="admission threshold gamma_cov")
    p.add_argument("--redetect", default="quartile", help="'quartile' or 'stride:K'")
    p.add_argument("--tiou-stride", type=int, default=4,
                   help="spatial subsampling for temporal-IoU de-duplication (4 in the paper's QRD runs)")
    a = p.parse_args()
    configure(a.threads)

    import numpy as np
    from dovos.commit import TemporalIoU, commit, track_stats
    from dovos.data import list_videos, load_video
    from dovos.detector import build_detector
    from dovos.evaluate import score_arm, summarize
    from dovos.metrics import VideoScorer
    from dovos.propagation import Propagator
    from dovos.qrd import run_qrd

    detect = build_detector(a.detector, a.device, a.objectness)
    propagator = Propagator(a.checkpoint_dir, a.sam2_size, a.device)
    videos = list_videos(a.root, a.video_list, a.shard, a.num_shards, a.max_videos)
    per_video, skipped = {}, {}
    for i, name in enumerate(videos, 1):
        t0 = time.time()
        video = load_video(a.root, name, a.max_frames, a.spread)
        if video is None or not video.object_ids:
            skipped[name] = "no annotated objects"
            continue
        out = run_qrd(video.frames, detect, propagator, a.cover_max, a.redetect)
        if out is None:
            skipped[name] = "no detection at frame 0"
            continue
        tracks, conf, origin, calls = out
        stats = {t: track_stats(tracks[t], conf[t]) for t in tracks}
        tiou = TemporalIoU(tracks, a.tiou_stride)
        scorer = VideoScorer(video.masks)
        jrows = {t: scorer.j_row(tracks[t]) for t in tracks}
        f0 = [t for t in tracks if origin[t] == "f0"]
        disc = [t for t in tracks if origin[t] == "disc"]

        committed_f0 = commit(f0, stats, tiou)
        committed_all = commit(list(tracks), stats, tiou)
        f0_union = tiou.union_mask(committed_f0)
        gated = []
        for t in disc:
            s = stats[t]
            if s.persistence < GATE_PERSISTENCE or not (0 < s.mean_area < 0.5):
                continue
            area = int(tiou.stacked(t).sum())
            covered = 0 if f0_union is None else int(np.logical_and(tiou.stacked(t), f0_union).sum())
            if (covered / area if area else 1.0) < GATE_COVERAGE:
                gated.append(t)

        per_video[name] = {
            "frame0": score_arm(scorer, tracks, jrows, f0, committed_f0, nonoverlap=True),
            "qrd": score_arm(scorer, tracks, jrows, list(tracks), committed_all, nonoverlap=True),
            "qrd_gated": score_arm(scorer, tracks, jrows, f0 + gated, commit(f0 + gated, stats, tiou)),
            "n_discovered": len(disc), "n_objects": len(video.object_ids), "n_frames": video.n,
            "detector_calls": calls, "seconds": round(time.time() - t0, 1),
        }
        r = per_video[name]
        print(f"[{i}/{len(videos)}] {name}: frame0 {r['frame0']['jf']:.3f} qrd {r['qrd']['jf']:.3f} "
              f"({len(disc)} discovered)", flush=True)
        write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped,
                               summary=summarize(per_video, ["frame0", "qrd", "qrd_gated"])))
    write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped,
                           summary=summarize(per_video, ["frame0", "qrd", "qrd_gated"]), complete=True))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
