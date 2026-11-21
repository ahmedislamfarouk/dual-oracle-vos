"""Loading DAVIS-style video datasets.

A dataset root holds `JPEGImages/<video>/*.jpg` and `Annotations/<video>/*.png` (palette masks, one
integer id per object). Only frames with a released annotation are used, so YouTube-VOS is scored
on its annotated frames. Frames are resized to 854x480 (masks with nearest-neighbour interpolation).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

FRAME_SIZE = (854, 480)   # (width, height)


@dataclass
class Video:
    name: str
    frames: List[np.ndarray]   # RGB uint8, (480, 854, 3)
    masks: List[np.ndarray]    # int object ids, (480, 854)

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def object_ids(self) -> List[int]:
        return sorted({int(v) for m in self.masks for v in np.unique(m[m > 0])})


def list_videos(root: str, video_list: Optional[str] = None, shard: int = 0, num_shards: int = 1,
                max_videos: int = 0) -> List[str]:
    jroot = os.path.join(root, "JPEGImages")
    videos = sorted(v for v in os.listdir(jroot) if os.path.isdir(os.path.join(jroot, v)))
    if video_list:
        keep = set(open(video_list).read().split())
        videos = [v for v in videos if v in keep]
    if max_videos > 0:
        videos = videos[:max_videos]
    if num_shards > 1:
        videos = videos[shard::num_shards]
    return videos


def load_video(root: str, name: str, max_frames: int = 0, spread: bool = False) -> Optional[Video]:
    """Load one video; returns None when it has fewer than two annotated frames.

    If the video has more than `max_frames` annotated frames it is either truncated to the first
    `max_frames` (spread=False, used for capped MOSE) or subsampled evenly across its whole length
    (spread=True, used for LVOS).
    """
    frame_paths = sorted(glob.glob(os.path.join(root, "JPEGImages", name, "*.jpg")))
    ann = {os.path.splitext(os.path.basename(p))[0]: p
           for p in glob.glob(os.path.join(root, "Annotations", name, "*.png"))}
    pairs = [(p, ann[os.path.splitext(os.path.basename(p))[0]]) for p in frame_paths
             if os.path.splitext(os.path.basename(p))[0] in ann]
    if max_frames > 0 and len(pairs) > max_frames:
        if spread:
            idx, seen, kept = np.linspace(0, len(pairs) - 1, max_frames).round().astype(int), set(), []
            for i in idx:
                if i not in seen:
                    seen.add(i)
                    kept.append(pairs[i])
            pairs = kept
        else:
            pairs = pairs[:max_frames]
    if len(pairs) < 2:
        return None
    frames = [cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), FRAME_SIZE) for f, _ in pairs]
    masks = [cv2.resize(np.array(Image.open(a)), FRAME_SIZE, interpolation=cv2.INTER_NEAREST) for _, a in pairs]
    return Video(name, frames, masks)
