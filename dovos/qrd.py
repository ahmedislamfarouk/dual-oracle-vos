"""QRD: quartile re-detection (Algorithm 1 of the paper).

The detector runs on frame 0 and again on frames n/4, n/2 and 3n/4. At a re-detection frame k, a
box is admitted only when the tracks already held cover less than `cover_max` of it; admitted boxes
are propagated forward and backward from k, so an object first found late is not scored absent on
the frames before it. Discovered and frame-0 tracks are then committed by the same rule.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .propagation import FrameDir, Propagator, clip_box

COVER_MAX = 0.5


def redetection_frames(n: int, spec: str = "quartile") -> List[int]:
    """Re-detection frames: the quartiles, or every k-th frame with spec 'stride:k'."""
    if spec.startswith("stride:"):
        k = max(1, int(spec.split(":")[1]))
        return sorted({f for f in range(k, n, k) if 0 < f < n})
    return sorted({f for f in (n // 4, n // 2, (3 * n) // 4) if 0 < f < n})


def admit(boxes: Sequence[Sequence[float]], cover: np.ndarray, cover_max: float = COVER_MAX) -> List[Tuple[float, ...]]:
    """Boxes whose area is covered by existing tracks at less than `cover_max`."""
    h, w = cover.shape
    out = []
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in clip_box(b, w, h)]
        x2 = min(w, max(x1 + 1, x2))
        y2 = min(h, max(y1 + 1, y2))
        area = (x2 - x1) * (y2 - y1)
        if area > 0 and int(cover[y1:y2, x1:x2].sum()) / area < cover_max:
            out.append(clip_box(b, w, h))
    return out


def run_qrd(frames: Sequence[np.ndarray], detect: Callable, propagator: Propagator,
            cover_max: float = COVER_MAX, spec: str = "quartile"):
    """Run the frame-0 baseline and QRD on one video.

    Returns (tracks, confidences, origin, n_detector_calls), or None when frame 0 has no detection.
    Track ids are 'f0_<i>' for frame-0 tracks and 'd_<j>' for discovered ones.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    boxes0 = [clip_box(b, w, h) for b in detect(frames[0])]
    if not boxes0:
        return None
    tracks: Dict[str, List[np.ndarray]] = {}
    conf: Dict[str, List[float]] = {}
    origin: Dict[str, str] = {}
    calls = 1
    with FrameDir(frames) as fd:
        t, c = propagator.run(fd, (h, w), anchor=0, boxes={f"f0_{i}": b for i, b in enumerate(boxes0)},
                              bidirectional=False)
        tracks.update(t); conf.update(c); origin.update({k: "f0" for k in t})
        j = 0
        for k in redetection_frames(n, spec):
            cover = np.zeros((h, w), np.uint8)
            for ms in tracks.values():
                cover |= ms[k]
            calls += 1
            new = admit(detect(frames[k]), cover, cover_max)
            if not new:
                continue
            ids = {f"d_{j + i}": b for i, b in enumerate(new)}
            j += len(new)
            t, c = propagator.run(fd, (h, w), anchor=k, boxes=ids, bidirectional=True)
            tracks.update(t); conf.update(c); origin.update({k2: "disc" for k2 in t})
    return tracks, conf, origin, calls
