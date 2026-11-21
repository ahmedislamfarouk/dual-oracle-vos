import numpy as np
import pytest

from dovos.metrics import VideoScorer, bijective_jf, exclusive_jf, f_measure, greedy_jf, sequence_iou


def square(h, w, y0, x0, s):
    m = np.zeros((h, w), np.uint8)
    m[y0:y0 + s, x0:x0 + s] = 1
    return m


def test_iou_counts_empty_frames_as_perfect():
    empty = np.zeros((8, 8), np.uint8)
    full = square(8, 8, 0, 0, 4)
    assert sequence_iou([empty, full], [empty, full]) == 1.0
    assert sequence_iou([empty, empty], [empty, full]) == 0.5


def test_f_measure_identical_and_disjoint():
    a = square(64, 64, 10, 10, 20)
    assert f_measure(a, a) == pytest.approx(1.0)
    assert f_measure(a, square(64, 64, 40, 40, 10)) == 0.0
    assert f_measure(np.zeros((64, 64)), np.zeros((64, 64))) == 0.0   # empty prediction has zero precision


def _two_object_video():
    h, w, n = 64, 64, 3
    gt = [np.zeros((h, w), np.int32) for _ in range(n)]
    for m in gt:
        m[5:25, 5:25] = 1
        m[35:55, 35:55] = 2
    obj1 = [(m == 1).astype(np.uint8) for m in gt]
    obj2 = [(m == 2).astype(np.uint8) for m in gt]
    return gt, obj1, obj2


def test_bijective_scores_unmatched_object_zero_and_greedy_reuses_tracks():
    gt, obj1, _ = _two_object_video()
    scorer = VideoScorer(gt)
    tracks = {"a": obj1, "dup": [m.copy() for m in obj1]}
    jrows = {t: scorer.j_row(m) for t, m in tracks.items()}
    b = bijective_jf(scorer, tracks, jrows, ["a"])
    assert b["jf"] == pytest.approx(0.5)            # object 1 perfect, object 2 unmatched
    assert b["recall"] == 0.5 and b["precision"] == 1.0
    assert greedy_jf(scorer, tracks, jrows, ["a", "dup"]) == pytest.approx(0.5)


def test_bijective_perfect_tracks():
    gt, obj1, obj2 = _two_object_video()
    scorer = VideoScorer(gt)
    tracks = {"a": obj1, "b": obj2}
    jrows = {t: scorer.j_row(m) for t, m in tracks.items()}
    assert bijective_jf(scorer, tracks, jrows, ["b", "a"])["jf"] == pytest.approx(1.0)


def test_nonoverlap_rescore_removes_credit_for_duplicates():
    gt, obj1, obj2 = _two_object_video()
    scorer = VideoScorer(gt)
    union = [np.maximum(a, b) for a, b in zip(obj1, obj2)]
    tracks = {"both": union, "o2": obj2}
    # painted in order, 'both' claims object 2's pixels first, so 'o2' is left empty
    assert exclusive_jf(scorer, tracks, ["both", "o2"]) < exclusive_jf(scorer, tracks, ["o2", "both"])
    assert exclusive_jf(scorer, tracks, ["o2", "both"], cap=1) == pytest.approx(0.5)
