"""Shared command-line options and process setup for the run scripts."""
from __future__ import annotations

import argparse
import json
import os


def configure(threads: int = 2) -> None:
    """Keep CPU thread use small; set before importing torch-heavy modules."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(threads)
    try:  # some transformers versions probe an optional torchao install that is not needed here
        import transformers.utils.import_utils as iu
        iu._torchao_available = False
    except Exception:
        pass


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root", required=True, help="dataset root with JPEGImages/ and Annotations/")
    p.add_argument("--out", required=True, help="output JSON file")
    p.add_argument("--max-frames", type=int, default=0, help="cap on annotated frames per video (0 = all)")
    p.add_argument("--spread", action="store_true", help="subsample capped videos evenly instead of truncating")
    p.add_argument("--video-list", help="text file of video names to keep")
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--checkpoint-dir", default="checkpoints", help="directory with SAM 2.1 checkpoints")
    p.add_argument("--sam2-size", default="large", choices=["small", "base", "large"])
    p.add_argument("--objectness", type=float, default=0.05, help="DETR objectness floor")
    p.add_argument("--detector", default="detr", choices=["detr", "maskrcnn"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2)
    return p


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
