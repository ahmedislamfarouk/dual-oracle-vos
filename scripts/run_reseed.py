#!/usr/bin/env python3
"""QRD followed by track-anchored reseeding, paired per video with frame-0 and QRD.

Arms written for every video:
    frame0                 detector on frame 0 only
    qrd                    frame 0 plus quartile re-detection
    reseed                 commitment over all QRD tracks and the reseeds of QRD's committed tracks
    reseed_committed       commitment over QRD's committed tracks and their reseeds only
    replace_quality        each committed QRD track replaced by the highest-quality member of its family
                           (the track and its reseeds), so the track count stays QRD's
    replace_agreement      as above, choosing the reseed whose frame the detector agrees with most
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dovos.runtime import common_parser, configure, write_json


def main():
    p = common_parser(__doc__.split("\n")[0])
    p.add_argument("--cover-max", type=float, default=0.5)
    p.add_argument("--reseeds", type=int, default=3, help="reseeds per committed track")
    p.add_argument("--min-agreement", type=float, default=0.5,
                   help="minimum IoU between a detector box and the track's own box at a reseed frame")
    p.add_argument("--tiou-stride", type=int, default=4)
    a = p.parse_args()
    configure(a.threads)

    from dovos.commit import TemporalIoU, commit, track_stats
    from dovos.data import list_videos, load_video
    from dovos.detector import ObjectnessDetector
    from dovos.evaluate import score_arm, summarize
    from dovos.metrics import VideoScorer
    from dovos.propagation import FrameDir, Propagator
    from dovos.qrd import run_qrd
    from dovos.reseed import reseed_frames

    detect = ObjectnessDetector(a.device, objectness_threshold=a.objectness)
    propagator = Propagator(a.checkpoint_dir, a.sam2_size, a.device)
    arms = ["frame0", "qrd", "reseed", "reseed_committed", "replace_quality", "replace_agreement"]
    videos = list_videos(a.root, a.video_list, a.shard, a.num_shards, a.max_videos)
    per_video, skipped = {}, {}
    for i, name in enumerate(videos, 1):
        t0 = time.time()
        video = load_video(a.root, name, a.max_frames, a.spread)
        if video is None or not video.object_ids:
            skipped[name] = "no annotated objects"
            continue
        out = run_qrd(video.frames, detect, propagator, a.cover_max)
        if out is None:
            skipped[name] = "no detection at frame 0"
            continue
        tracks, conf, origin, calls = out
        stats = {t: track_stats(tracks[t], conf[t]) for t in tracks}
        tiou = TemporalIoU(tracks, a.tiou_stride)
        f0 = [t for t in tracks if origin[t] == "f0"]
        committed_f0 = commit(f0, stats, tiou)
        committed_qrd = commit(list(tracks), stats, tiou)

        # reseed every committed QRD track from the frames where the detector agrees with it most
        detections = [detect(fr) for fr in video.frames]
        seeds, family, agreement = [], {t: [t] for t in committed_qrd}, {}
        for t in committed_qrd:
            for j, (f, box, v) in enumerate(reseed_frames(tracks[t], detections, a.reseeds, a.min_agreement)):
                sid = f"r_{t}_{j}"
                seeds.append((sid, f, box)); family[t].append(sid); agreement[sid] = v
        h, w = video.masks[0].shape
        with FrameDir(video.frames) as fd:
            for anchor in sorted({s[1] for s in seeds}):
                t_, c_ = propagator.run(fd, (h, w), anchor, boxes={s[0]: s[2] for s in seeds if s[1] == anchor})
                tracks.update(t_); conf.update(c_)
        reseeds = [s[0] for s in seeds]
        for s in reseeds:
            stats[s] = track_stats(tracks[s], conf[s])
        tiou = TemporalIoU(tracks, a.tiou_stride)
        qrd_pool = [t for t in tracks if t not in set(reseeds)]

        def eligible(m):
            return stats[m].persistence >= 0.5 and 0 < stats[m].mean_area < 0.5

        def pick(t, by):
            members = [m for m in family[t] if eligible(m)] or [t]
            if by == "quality":
                return max(members, key=lambda m: stats[m].quality)
            reseeded = [m for m in members if m != t]
            return max(reseeded, key=lambda m: agreement[m]) if reseeded else t

        replace_q = commit([pick(t, "quality") for t in committed_qrd], stats, tiou)
        replace_a = commit([pick(t, "agreement") for t in committed_qrd], stats, tiou)
        scorer = VideoScorer(video.masks)
        jrows = {t: scorer.j_row(tracks[t]) for t in tracks}
        per_video[name] = {
            "frame0": score_arm(scorer, tracks, jrows, f0, committed_f0, nonoverlap=True),
            "qrd": score_arm(scorer, tracks, jrows, qrd_pool, committed_qrd, nonoverlap=True),
            "reseed": score_arm(scorer, tracks, jrows, qrd_pool + reseeds,
                                commit(qrd_pool + reseeds, stats, tiou), nonoverlap=True),
            "reseed_committed": score_arm(scorer, tracks, jrows, committed_qrd + reseeds,
                                          commit(committed_qrd + reseeds, stats, tiou), nonoverlap=True),
            "replace_quality": score_arm(scorer, tracks, jrows, qrd_pool + reseeds, replace_q, nonoverlap=True),
            "replace_agreement": score_arm(scorer, tracks, jrows, qrd_pool + reseeds, replace_a, nonoverlap=True),
            "n_reseeds": len(reseeds), "n_objects": len(video.object_ids), "n_frames": video.n,
            "detector_calls": calls + video.n, "seconds": round(time.time() - t0, 1),
        }
        r = per_video[name]
        print(f"[{i}/{len(videos)}] {name}: frame0 {r['frame0']['jf']:.3f} qrd {r['qrd']['jf']:.3f} "
              f"reseed {r['reseed']['jf']:.3f} replace_q {r['replace_quality']['jf']:.3f} "
              f"replace_a {r['replace_agreement']['jf']:.3f} "
              f"({len(reseeds)} reseeds)", flush=True)
        write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped, summary=summarize(per_video, arms)))
    write_json(a.out, dict(config=vars(a), per_video=per_video, skipped=skipped,
                           summary=summarize(per_video, arms), complete=True))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
