"""Seeding rules: the frame-0 baseline, the detection oracles and the timing oracle.

    frame0  real detector on frame 0 (the baseline)
    O1      ground-truth box of every object visible at frame 0
    O2      ground-truth box at each object's first appearance
    O3      ground-truth mask at each object's first appearance
    O4      the real detector's box at each object's best frame(s)
    tta     four detector calls on frame 0 (identity, mirror, 0.75x, 1.25x); the budget control

O4 runs the detector on every frame and scores each box against each object's ground-truth box on
that frame. With `frames='best'` an object is seeded at its K highest-overlap frames, at least
n/(2K) apart; with 'quartile', 'mid' or 'random' the frames are fixed without looking at ground
truth, and ground truth only picks which of the detector's boxes is the object.

Every rule returns a list of seeds (id, anchor frame, box or None, mask or None) in a fixed order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MIN_OBJECT_PIXELS = 10


@dataclass
class Seed:
    id: str
    anchor: int
    box: Optional[List[float]] = None
    mask: Optional[np.ndarray] = None


def _gt_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.where(mask)
    if xs.size < MIN_OBJECT_PIXELS:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def frame0(frames, detect: Callable) -> List[Seed]:
    return [Seed(f"b{i}", 0, list(b)) for i, b in enumerate(detect(frames[0]))]


def o1(masks) -> List[Seed]:
    m0 = masks[0]
    seeds = []
    for g in sorted({int(v) for v in np.unique(m0[m0 > 0])}):
        box = _gt_box(m0 == g)
        if box is not None:
            seeds.append(Seed(f"o{g}", 0, list(box)))
    return seeds


def first_appearance(masks, use_mask: bool) -> List[Seed]:
    """O2 (box) and O3 (mask) at each object's first annotated appearance."""
    first: Dict[int, int] = {}
    for fi, m in enumerate(masks):
        for g in np.unique(m[m > 0]):
            first.setdefault(int(g), fi)
    seeds = []
    for g, fi in sorted(first.items()):
        obj = masks[fi] == g
        box = _gt_box(obj)
        if box is not None:
            seeds.append(Seed(f"o{g}", fi, list(box), obj.astype(np.uint8) if use_mask else None))
    return seeds


def o4(frames, masks, detect: Callable, video_name: str, seeds_per_object: int = 1,
       frame_rule: str = "best", random_seed: int = 0) -> List[Seed]:
    n = len(frames)
    detections = [detect(frames[f]) for f in range(n)]
    per_frame: Dict[Tuple[int, int], Tuple[float, list]] = {}
    for f in range(n):
        m = masks[f]
        for g in np.unique(m[m > 0]):
            g = int(g)
            gt = _gt_box(m == g)
            if gt is None:
                continue
            for b in detections[f]:
                v = box_iou(tuple(float(x) for x in b[:4]), gt)
                if v > per_frame.get((g, f), (0.0, None))[0]:
                    per_frame[(g, f)] = (v, list(b))
    objects = sorted({g for (g, _), (v, _) in per_frame.items() if v > 0})
    k = seeds_per_object
    min_gap = max(1, n // (2 * k))
    if frame_rule == "quartile":
        fixed = sorted({min(n - 1, max(0, x)) for x in (n // 4, n // 2, (3 * n) // 4)})
    elif frame_rule == "mid":
        fixed = [n // 2]
    elif frame_rule == "random":
        rng = np.random.default_rng((sum(map(ord, video_name)) * 7919 + random_seed) % (2 ** 32))
        fixed = sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    elif frame_rule == "best":
        fixed = None
    else:
        raise ValueError(f"unknown frame rule {frame_rule!r}")
    seeds = []
    for g in objects:
        if fixed is None:
            cand = sorted([(v, f, b) for (gg, f), (v, b) in per_frame.items() if gg == g and v > 0],
                          key=lambda x: -x[0])
        else:
            cand = [(per_frame[(g, f)][0], f, per_frame[(g, f)][1]) for f in fixed
                    if (g, f) in per_frame and per_frame[(g, f)][0] > 0]
        chosen: List[Tuple[float, int, list]] = []
        for v, f, b in cand:
            if fixed is not None or all(abs(f - c[1]) >= min_gap for c in chosen):
                chosen.append((v, f, b))
            if len(chosen) == k:
                break
        seeds += [Seed(f"o{g}_{j}", int(f), b) for j, (v, f, b) in enumerate(chosen)]
    return seeds


def tta(frames, detect: Callable, cover_max: float = 0.5) -> List[Seed]:
    """Budget control: four detector calls on frame 0, extra calls gated like QRD re-detections."""
    img = frames[0]
    h0, w0 = img.shape[:2]
    variants = [("id", img), ("flip", np.ascontiguousarray(img[:, ::-1])),
                ("s075", cv2.resize(img, (max(2, int(w0 * 0.75)), max(2, int(h0 * 0.75))))),
                ("s125", cv2.resize(img, (int(w0 * 1.25), int(h0 * 1.25))))]
    raw = []
    for tag, im in variants:
        for b in detect(im):
            x1, y1, x2, y2 = b[:4]
            if tag == "flip":
                x1, x2 = w0 - x2, w0 - x1
            elif tag == "s075":
                x1, y1, x2, y2 = x1 / 0.75, y1 / 0.75, x2 / 0.75, y2 / 0.75
            elif tag == "s125":
                x1, y1, x2, y2 = x1 / 1.25, y1 / 1.25, x2 / 1.25, y2 / 1.25
            raw.append((tag == "id", [float(x1), float(y1), float(x2), float(y2)] + list(b[4:])))

    def pixel_box(b):
        x1, y1, x2, y2 = [int(max(0, v)) for v in b[:4]]
        return x1, y1, min(w0, max(x1 + 1, x2)), min(h0, max(y1 + 1, y2))

    cover = np.zeros((h0, w0), np.uint8)
    seeds = []
    for i, (is_id, b) in enumerate(raw):            # the identity call is admitted as the baseline admits it
        if not is_id:
            continue
        x1, y1, x2, y2 = pixel_box(b)
        if (x2 - x1) * (y2 - y1) <= 0:
            continue
        cover[y1:y2, x1:x2] = 1
        seeds.append(Seed(f"t{i}", 0, b))
    for i, (is_id, b) in enumerate(raw):            # the three extra calls pass the coverage gate
        if is_id:
            continue
        x1, y1, x2, y2 = pixel_box(b)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or int(cover[y1:y2, x1:x2].sum()) / area >= cover_max:
            continue
        cover[y1:y2, x1:x2] = 1
        seeds.append(Seed(f"t{i}", 0, b))
    return seeds
