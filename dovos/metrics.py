"""Region (J) and boundary (F) accuracy, and the matching rules used in the paper.

A track and a ground-truth object are compared frame by frame. J is the mean per-frame IoU, where a
frame on which both masks are empty counts as 1. F is the mean per-frame boundary F-measure. An
object's score is (J + F) / 2 of the track assigned to it, a video's score is the mean over its
objects, and an object counts as recovered when its assigned track has J > 0.5.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.optimize import linear_sum_assignment

Mask = np.ndarray            # (H, W) uint8 or bool
MaskSeq = Sequence[Mask]     # one mask per frame


def _boundary(mask: np.ndarray, boundary_thresh: float = 0.008) -> np.ndarray:
    h, w = mask.shape
    radius = max(1, int(np.round(boundary_thresh * np.sqrt(h * h + w * w))))
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(mask, structure=struct) & ~binary_erosion(mask, structure=struct)


def f_measure(pred: Mask, gt: Mask, boundary_thresh: float = 0.008) -> float:
    """Boundary F-measure of one predicted mask against one ground-truth mask."""
    pb = _boundary(pred.astype(bool), boundary_thresh)
    gb = _boundary(gt.astype(bool), boundary_thresh)
    precision = float((pb & gb).sum()) / pb.sum() if pb.sum() else 0.0
    recall = float((pb & gb).sum()) / gb.sum() if gb.sum() else 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def sequence_iou(pred: MaskSeq, gt: MaskSeq) -> float:
    """Mean per-frame IoU; a frame where both masks are empty scores 1."""
    n = len(gt)
    p = np.stack(pred).astype(bool).reshape(n, -1)
    g = np.stack(gt).astype(bool).reshape(n, -1)
    inter = np.logical_and(p, g).sum(1)
    union = np.logical_or(p, g).sum(1)
    return float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean())


class VideoScorer:
    """Scores tracks of one video against its ground-truth objects, caching J and F per pair."""

    def __init__(self, gt_masks: MaskSeq):
        self.n = len(gt_masks)
        self.gt_ids = sorted({int(v) for m in gt_masks for v in np.unique(m[m > 0])})
        self.gt = {g: [(m == g).astype(np.uint8) for m in gt_masks] for g in self.gt_ids}
        self._gt_flat = {g: np.stack(self.gt[g]).astype(bool).reshape(self.n, -1) for g in self.gt_ids}
        self._f: Dict[tuple, float] = {}

    def j_row(self, track: MaskSeq) -> np.ndarray:
        """J of one track against every ground-truth object."""
        p = np.stack(track).astype(bool).reshape(self.n, -1)
        out = []
        for g in self.gt_ids:
            gb = self._gt_flat[g]
            inter = np.logical_and(p, gb).sum(1)
            union = np.logical_or(p, gb).sum(1)
            out.append(float(np.where(union == 0, 1.0, inter / np.maximum(union, 1)).mean()))
        return np.array(out)

    def f(self, key, track: MaskSeq, gi: int) -> float:
        k = (key, gi)
        if k not in self._f:
            g = self.gt[self.gt_ids[gi]]
            self._f[k] = float(np.mean([f_measure(track[i], g[i]) for i in range(self.n)]))
        return self._f[k]


def greedy_jf(scorer: VideoScorer, tracks: Dict[str, MaskSeq], jrows: Dict[str, np.ndarray],
              ids: List[str]) -> float:
    """Candidate-pool score: every object takes its best track by J; a track may serve several."""
    if not ids:
        return 0.0
    J = np.stack([jrows[t] for t in ids])
    scores = []
    for gi in range(len(scorer.gt_ids)):
        t = ids[int(np.argmax(J[:, gi]))]
        scores.append((J[:, gi].max() + scorer.f(t, tracks[t], gi)) / 2)
    return float(np.mean(scores))


def bijective_jf(scorer: VideoScorer, tracks: Dict[str, MaskSeq], jrows: Dict[str, np.ndarray],
                 committed: List[str]) -> dict:
    """Committed-track score under one-to-one Hungarian assignment; unmatched objects score 0."""
    G = len(scorer.gt_ids)
    out = dict(jf=0.0, precision=0.0, recall=0.0, hits=0, n_committed=len(committed))
    if not committed:
        return out
    J = np.stack([jrows[t] for t in committed])
    rows, cols = linear_sum_assignment(-J)
    per_gt = np.zeros(G)
    hits = 0
    for r, c in zip(rows, cols):
        per_gt[c] = (J[r, c] + scorer.f(committed[r], tracks[committed[r]], c)) / 2
        hits += int(J[r, c] > 0.5)
    out.update(jf=float(per_gt.mean()), precision=hits / len(committed), recall=hits / G, hits=hits)
    return out


def exclusive_jf(scorer: VideoScorer, tracks: Dict[str, MaskSeq], order: List[str], cap: int = 0) -> float:
    """Non-overlapping re-score.

    Tracks are painted in the given order (highest quality first) and each pixel belongs to the first
    track that claims it, so duplicate tracks can no longer each cover the same object. With cap > 0
    only the first `cap` tracks are kept, as in the DAVIS-2019 unsupervised protocol (cap = 20).
    """
    order = list(order[:cap] if cap else order)
    if not order:
        return 0.0
    n = scorer.n
    h, w = tracks[order[0]][0].shape
    painted = {t: [] for t in order}
    for fi in range(n):
        taken = np.zeros((h, w), bool)
        for t in order:
            m = tracks[t][fi].astype(bool) & ~taken
            painted[t].append(m.astype(np.uint8))
            taken |= m
    J = np.stack([scorer.j_row(painted[t]) for t in order])
    rows, cols = linear_sum_assignment(-J)
    per_gt = np.zeros(len(scorer.gt_ids))
    for r, c in zip(rows, cols):
        g = scorer.gt[scorer.gt_ids[c]]
        fx = float(np.mean([f_measure(painted[order[r]][i], g[i]) for i in range(n)]))
        per_gt[c] = (J[r, c] + fx) / 2
    return float(per_gt.mean())
