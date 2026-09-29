#!/usr/bin/env python3
"""Reject runtime-only CPD-2 variants unless saved scientific output is exact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RUNTIME_CONFIG_KEYS = {"evaluation_workers", "offload_video_to_cpu"}
NONSCIENTIFIC_VIDEO_KEYS = {"wall_seconds"}


def die(message: str) -> None:
    raise SystemExit(f"CPD-2 runtime-parity error: {message}")


def load_complete(path: Path) -> dict[str, Any]:
    if not path.exists():
        die(f"missing output: {path}")
    data = json.loads(path.read_text())
    if data.get("schema_version") != "cpd2-v1" or data.get("run_status") != "complete":
        die(f"incomplete or wrong schema: {path}")
    return data


def scientific_config(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data["config"].items() if key not in RUNTIME_CONFIG_KEYS}


def scientific_videos(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: value for key, value in video.items() if key not in NONSCIENTIFIC_VIDEO_KEYS}
            for video in data.get("videos", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite: {args.output}")
    reference = load_complete(args.reference)
    candidate = load_complete(args.candidate)
    if scientific_config(reference) != scientific_config(candidate):
        die("scientific configuration differs; this is not a runtime-only comparison")
    reference_videos = scientific_videos(reference)
    candidate_videos = scientific_videos(candidate)
    if reference_videos != candidate_videos:
        die("saved scientific video outputs differ")
    report = {
        "schema_version": "cpd2-runtime-parity-v1",
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_runtime_config": {key: reference["config"].get(key) for key in sorted(RUNTIME_CONFIG_KEYS)},
        "candidate_runtime_config": {key: candidate["config"].get(key) for key in sorted(RUNTIME_CONFIG_KEYS)},
        "scientific_config_exact": True,
        "scientific_video_outputs_exact_excluding_wall_seconds": True,
        "video_count": len(reference_videos),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"runtime parity verified for {len(reference_videos)} videos")


if __name__ == "__main__":
    main()
