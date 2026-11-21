#!/usr/bin/env python3
"""One seeding rule (baseline, oracle or control) through the frozen pipeline.

    --seeding frame0     detector on frame 0 (the baseline)
    --seeding o1         ground-truth boxes at frame 0 (detection oracle)
    --seeding o2 | o3    ground-truth box | mask at each object's first appearance
    --seeding o4         the real detector's box at each object's best frame(s) (timing oracle);
                         --seeds K and --frames {best,quartile,mid,random} give the frame x seed variants
    --seeding tta        four detector calls on frame 0 (detector-budget control)

Example (three-seed timing oracle on capped MOSE):
    python scripts/run_seeding.py --root data/mose_subset/train --max-frames 60 \
        --seeding o4 --seeds 3 --frames best --out results/runs/o4_best3_mose.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dovos.runtime import common_parser, configure, write_json


def main():
    p = common_parser(__doc__.split("\n")[0])
    p.add_argument("--seeding", required=True, choices=["frame0", "o1", "o2", "o3", "o4", "tta"])
    p.add_argument("--seeds", type=int, default=1, help="seeds per object for o4")
    p.add_argument("--frames", default="best", choices=["best", "quartile", "mid", "random"])
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--cover-max", type=float, default=0.5, help="coverage gate for the tta control")
    p.add_argument("--tiou-stride", type=int, default=1, help="spatial subsampling for de-duplication")
    a = p.parse_args()
    configure(a.threads)

    from dovos import oracles
    from dovos.commit import TemporalIoU, commit, track_stats
    from dovos.data import list_videos, load_video
    from dovos.detector import build_detector
    from dovos.evaluate import score_arm, summarize
    from dovos.metrics import VideoScorer
    from dovos.propagation import FrameDir, Propagator

    detect = build_detector(a.detector, a.device, a.objectness)
    propagator = Propagator(a.checkpoint_dir, a.sam2_size, a.device)
    forward_only = a.seeding in ("frame0", "o1", "tta")    # everything else propagates both ways from its anchor
    videos = list_videos(a.root, a.video_list, a.shard, a.num_shards, a.max_videos)
    per_video, skipped = {}, {}
    for i, name in enumerate(videos, 1):
        t0 = time.time()
        video = load_video(a.root, name, a.max_frames, a.spread)
        if video is None or not video.object_ids:
            skipped[name] = "no annotated objects"
            continue
        if a.seeding == "frame0":
            seeds = oracles.frame0(video.frames, detect)
        elif a.seeding == "o1":
            seeds = oracles.o1(video.masks)
        elif a.seeding in ("o2", "o3"):
            seeds = oracles.first_appearance(video.masks, use_mask=a.seeding == "o3")
        elif a.seeding == "o4":
            seeds = oracles.o4(video.frames, video.masks, detect, name, a.seeds, a.frames, a.random_seed)
        else:
            seeds = oracles.tta(video.frames, detect, a.cover_max)
        if not seeds:
            skipped[name] = "no seed"
            continue

        tracks, conf = {}, {}
        h, w = video.masks[0].shape
        with FrameDir(video.frames) as fd:
            for anchor in sorted({s.anchor for s in seeds}):
                group = [s for s in seeds if s.anchor == anchor]
                t, c = propagator.run(fd, (h, w), anchor,
                                      boxes={s.id: s.box for s in group if s.mask is None},
                                      masks={s.id: s.mask for s in group if s.mask is not None},
                                      bidirectional=not forward_only)
                tracks.update(t); conf.update(c)
        order = [s.id for s in seeds]                      # seed order fixes tie-breaking in commitment
        tracks = {k: tracks[k] for k in order}
        stats = {k: track_stats(tracks[k], conf[k]) for k in order}
        committed = commit(order, stats, TemporalIoU(tracks, a.tiou_stride))
        scorer = VideoScorer(video.masks)
        jrows = {k: scorer.j_row(tracks[k]) for k in order}
        per_video[name] = {a.seeding: score_arm(scorer, tracks, jrows, order, committed, nonoverlap=True),
                           "n_seeds": len(seeds), "n_objects": len(video.object_ids), "n_frames": video.n,
                           "seconds": round(time.time() - t0, 1)}
        print(f"[{i}/{len(videos)}] {name}: {a.seeding} {per_video[name][a.seeding]['jf']:.3f} "
              f"({len(seeds)} seeds)", flush=True)
        write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped,
                               summary=summarize(per_video, [a.seeding])))
    write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped,
                           summary=summarize(per_video, [a.seeding]), complete=True))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
