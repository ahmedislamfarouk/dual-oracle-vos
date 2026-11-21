import numpy as np
import pytest

from dovos import oracles
from dovos.commit import TemporalIoU, commit, track_stats
from dovos.qrd import admit, redetection_frames
from dovos.reseed import reseed_frames


def track(n, frames_on, y0=0, x0=0, s=4, h=16, w=16):
    out = []
    for f in range(n):
        m = np.zeros((h, w), np.uint8)
        if f in frames_on:
            m[y0:y0 + s, x0:x0 + s] = 1
        out.append(m)
    return out


def test_track_stats():
    t = track(4, {0, 1}, s=4)
    s = track_stats(t, [2.0, 2.0])
    assert s.persistence == 0.5
    assert s.mean_area == pytest.approx(16 / 256)
    assert s.quality == pytest.approx(1.0)


def test_commit_filters_by_persistence_and_deduplicates_by_quality():
    n = 4
    tracks = {"short": track(n, {0}), "good": track(n, set(range(n))),
              "dup": track(n, set(range(n))), "other": track(n, set(range(n)), y0=8, x0=8)}
    conf = {"short": [5.0], "good": [3.0] * n, "dup": [1.0] * n, "other": [2.0] * n}
    stats = {t: track_stats(tracks[t], conf[t]) for t in tracks}
    kept = commit(list(tracks), stats, TemporalIoU(tracks))
    assert kept == ["good", "other"]        # 'short' fails persistence, 'dup' overlaps 'good'


def test_redetection_frames():
    assert redetection_frames(60) == [15, 30, 45]
    assert redetection_frames(3) == [1, 2]
    assert redetection_frames(20, "stride:5") == [5, 10, 15]


def test_admission_gate():
    cover = np.zeros((20, 20), np.uint8)
    cover[0:10, 0:10] = 1
    boxes = [[0, 0, 10, 10], [0, 0, 10, 20], [10, 10, 20, 20]]
    admitted = admit(boxes, cover, cover_max=0.5)
    assert [list(map(int, b)) for b in admitted] == [[10, 10, 20, 20]]   # half-covered box is not admitted


def gt_video():
    masks = [np.zeros((32, 32), np.int32) for _ in range(8)]
    for f in range(8):
        masks[f][2:12, 2:12] = 1            # object 1 present throughout
    for f in range(4, 8):
        masks[f][20:30, 20:30] = 2          # object 2 appears at frame 4
    return masks


def test_detection_oracles():
    masks = gt_video()
    assert [(s.id, s.anchor) for s in oracles.o1(masks)] == [("o1", 0)]
    seeds = oracles.first_appearance(masks, use_mask=True)
    assert [(s.id, s.anchor) for s in seeds] == [("o1", 0), ("o2", 4)]
    assert seeds[1].mask.sum() == 100


def test_timing_oracle_frame_rules():
    masks = gt_video()
    frames = [np.zeros((32, 32, 3), np.uint8)] * 8

    def detect(frame):                        # a fake detector that sees both objects on every frame
        return [[2, 2, 12, 12], [20, 20, 30, 30]]

    best = oracles.o4(frames, masks, detect, "vid", seeds_per_object=1, frame_rule="best")
    assert [(s.id, s.anchor) for s in best] == [("o1_0", 0), ("o2_0", 4)]   # ties go to the earliest frame
    three = oracles.o4(frames, masks, detect, "vid", seeds_per_object=3, frame_rule="best")
    anchors = [s.anchor for s in three if s.id.startswith("o1")]
    assert len(anchors) == 3 and min(np.diff(anchors)) >= 8 // 6
    quart = oracles.o4(frames, masks, detect, "vid", seeds_per_object=3, frame_rule="quartile")
    assert sorted(s.anchor for s in quart if s.id.startswith("o1")) == [2, 4, 6]


def test_reseed_picks_best_agreeing_frames_with_spacing():
    n = 12
    trk = []
    for f in range(n):
        m = np.zeros((40, 40), np.uint8)
        m[10:20, 10:20] = 1
        trk.append(m)
    exact, loose = [10, 10, 20, 20], [10, 10, 24, 24]
    dets = [[exact] if f in (2, 9) else ([] if f == 5 else [loose]) for f in range(n)]
    frames = [f for f, _, _ in reseed_frames(trk, dets, k=3, min_agreement=0.5)]
    assert frames[:2] == [2, 9]
    assert len(frames) == 3 and 5 not in frames
    assert all(abs(a - b) >= n // 6 for i, a in enumerate(frames) for b in frames[i + 1:])


def test_reseed_respects_minimum_agreement():
    trk = [np.pad(np.ones((5, 5), np.uint8), ((0, 35), (0, 35)))] * 4
    assert reseed_frames(trk, [[[30, 30, 39, 39]]] * 4, k=3, min_agreement=0.5) == []
