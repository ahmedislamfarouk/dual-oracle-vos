"""Mask propagation with SAM 2.1 (bfloat16 autocast).

Each seed is a box (or, for oracle O3, a mask) at an anchor frame. Seeds at frame 0 are propagated
forward; seeds at any other frame, and every oracle seed, are propagated forward and then backward
from their anchor. For every visited frame we record the binary mask and the mean positive logit,
which feeds the track-quality statistic. Note that a bidirectional pass visits its anchor frame
twice, so that frame contributes two confidence values, exactly as in the runs reported in the paper.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SAM2_CONFIGS = {
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}

Box = Sequence[float]
Tracks = Dict[str, List[np.ndarray]]
Confidences = Dict[str, List[float]]


def clip_box(box: Box, width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


class FrameDir:
    """Writes the (resized) frames of one video to a temporary JPEG directory for SAM 2."""

    def __init__(self, frames: Sequence[np.ndarray]):
        self.path = tempfile.mkdtemp(prefix="dovos_")
        self.n = len(frames)
        for i, fr in enumerate(frames):
            cv2.imwrite(os.path.join(self.path, f"{i:05d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


class Propagator:
    def __init__(self, checkpoint_dir: str, size: str = "large", device: str = "cuda"):
        from sam2.build_sam import build_sam2_video_predictor
        ckpt, cfg = SAM2_CONFIGS[size]
        self.predictor = build_sam2_video_predictor(cfg, os.path.join(checkpoint_dir, ckpt), device=device)

    def _record(self, tracks: Tracks, conf: Confidences, ids: Dict[int, str], fi: int, oids, masks, shape):
        h, w = shape
        for oid, ml in zip(oids, masks):
            key = ids.get(int(oid))
            if key is None:
                continue
            logit = ml[0].float().cpu().numpy()
            m = (logit > 0).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            tracks[key][fi] = m
            pos = logit[logit > 0]
            conf[key].append(float(pos.mean()) if pos.size else 0.0)

    def run(self, frames: FrameDir, shape: Tuple[int, int], anchor: int,
            boxes: Optional[Dict[str, Box]] = None, masks: Optional[Dict[str, np.ndarray]] = None,
            bidirectional: bool = True) -> Tuple[Tracks, Confidences]:
        """Propagate all seeds sharing one anchor frame; returns per-seed masks and confidences."""
        import torch
        h, w = shape
        seeds = list((boxes or {}).keys()) + list((masks or {}).keys())
        tracks: Tracks = {k: [np.zeros((h, w), np.uint8) for _ in range(frames.n)] for k in seeds}
        conf: Confidences = {k: [] for k in seeds}
        if not seeds:
            return tracks, conf
        ids = {i: k for i, k in enumerate(seeds)}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(video_path=frames.path, offload_video_to_cpu=True)
            for i, k in ids.items():
                if masks and k in masks:
                    self.predictor.add_new_mask(state, frame_idx=anchor, obj_id=i, mask=masks[k])
                else:
                    box = np.array([clip_box(boxes[k], w, h)], np.float32)
                    self.predictor.add_new_points_or_box(state, frame_idx=anchor, obj_id=i, box=box)
            directions = (False, True) if bidirectional else (False,)
            for reverse in directions:
                kwargs = dict(start_frame_idx=anchor, reverse=reverse) if bidirectional else {}
                for fi, oids, ms in self.predictor.propagate_in_video(state, **kwargs):
                    if 0 <= fi < frames.n:
                        self._record(tracks, conf, ids, fi, oids, ms, shape)
            self.predictor.reset_state(state)
        return tracks, conf
