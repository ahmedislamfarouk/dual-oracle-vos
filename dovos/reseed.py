"""Track-anchored reseeding: a ground-truth-free approximation of the multi-seed timing oracle.

The three-seed timing oracle seeds each object at the frames where the detector's box best overlaps
it, and needs ground truth only to say which box is the object. A propagated track can play that role
itself: at every frame we look for the detector box that best overlaps the track's own mask, and the
frames where detector and track agree most are where the detector sees that object best. Each track
is reseeded from its K best such frames (at least n/(2K) apart, as in the oracle), every reseed is
propagated forward and backward from its frame, and the unchanged commitment rule then chooses among
the original track and its reseeds. No ground truth is used.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .oracles import box_iou

MIN_PIXELS = 10


def mask_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.nonzero(mask)
    if xs.size < MIN_PIXELS:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def reseed_frames(track: Sequence[np.ndarray], detections: Sequence[Sequence[Sequence[float]]],
                  k: int = 3, min_agreement: float = 0.5) -> List[Tuple[int, List[float], float]]:
    """Up to k (frame, detector box, agreement) triples for one track, best agreement first."""
    n = len(track)
    candidates = []
    for f in range(n):
        tb = mask_box(track[f])
        if tb is None:
            continue
        best, best_box = 0.0, None
        for b in detections[f]:
            v = box_iou(tuple(float(x) for x in b[:4]), tb)
            if v > best:
                best, best_box = v, list(b)
        if best_box is not None and best >= min_agreement:
            candidates.append((best, f, best_box))
    candidates.sort(key=lambda c: -c[0])      # stable: ties keep the earliest frame
    gap = max(1, n // (2 * k))
    chosen: List[Tuple[int, List[float], float]] = []
    for v, f, b in candidates:
        if all(abs(f - c[0]) >= gap for c in chosen):
            chosen.append((f, b, v))
        if len(chosen) == k:
            break
    return chosen
