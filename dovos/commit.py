"""Ground-truth-free track commitment (Eq. 1-3 of the paper).

For a track T with masks T_0..T_{n-1} and per-frame propagator confidences c_t:
    persistence  pi(T) = |{t : T_t non-empty}| / n
    quality       q(T) = pi(T) * mean_t c_t
A track is committed when pi(T) >= persist_min and 0 < mean relative area < area_max; committed
tracks are taken in decreasing quality and kept only if their temporal IoU with every track already
kept is below dedup_iou.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

PERSIST_MIN = 0.5
AREA_MAX = 0.5
DEDUP_IOU = 0.5


@dataclass
class TrackStats:
    persistence: float
    mean_area: float     # mean mask area over non-empty frames, as a fraction of the frame
    quality: float


def track_stats(masks: Sequence[np.ndarray], confidences: Sequence[float]) -> TrackStats:
    n = len(masks)
    h, w = masks[0].shape
    areas = [int(m.sum()) for m in masks if m.sum() > 0]
    persistence = len(areas) / n
    mean_area = float(np.mean(areas)) / (h * w) if areas else 0.0
    quality = persistence * (float(np.mean(confidences)) if len(confidences) else 0.0)
    return TrackStats(persistence, mean_area, quality)


class TemporalIoU:
    """Temporal IoU between tracks, sum of intersections over sum of unions across frames.

    `stride` subsamples each mask spatially before comparison. The QRD runs in the paper used
    stride 4 and the oracle runs stride 1; the option exists to reproduce both exactly.
    """

    def __init__(self, tracks: Dict[str, Sequence[np.ndarray]], stride: int = 1):
        self._stack = {t: np.stack([m[::stride, ::stride] for m in ms]).astype(bool) for t, ms in tracks.items()}

    def __call__(self, a: str, b: str) -> float:
        inter = np.logical_and(self._stack[a], self._stack[b]).sum()
        union = np.logical_or(self._stack[a], self._stack[b]).sum()
        return float(inter / union) if union > 0 else 0.0

    def union_mask(self, ids: Sequence[str]) -> Optional[np.ndarray]:
        if not ids:
            return None
        return np.logical_or.reduce([self._stack[t] for t in ids])

    def stacked(self, t: str) -> np.ndarray:
        return self._stack[t]


def commit(candidates: Sequence[str], stats: Dict[str, TrackStats], tiou: TemporalIoU,
           persist_min: float = PERSIST_MIN, area_max: float = AREA_MAX,
           dedup_iou: float = DEDUP_IOU) -> List[str]:
    """Commit a track set without ground truth; returns ids in decreasing quality."""
    eligible = [t for t in candidates
                if stats[t].persistence >= persist_min and 0 < stats[t].mean_area < area_max]
    eligible.sort(key=lambda t: stats[t].quality, reverse=True)
    kept: List[str] = []
    for t in eligible:
        if all(tiou(t, k) < dedup_iou for k in kept):
            kept.append(t)
    return kept
