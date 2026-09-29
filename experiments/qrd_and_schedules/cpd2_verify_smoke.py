#!/usr/bin/env python3
"""Strict schema/invariant verifier for a completed four-arm CPD-2 smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEDULES = ("fixed", "random", "global", "residual")


def die(message: str) -> None:
    raise SystemExit(f"CPD-2 smoke verification error: {message}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite: {args.output}")
    payloads: dict[str, dict] = {}
    for schedule in SCHEDULES:
        path = args.directory / f"{schedule}.json"
        if not path.exists():
            die(f"missing schedule output: {path}")
        item = json.loads(path.read_text())
        if item.get("schema_version") != "cpd2-v1" or item.get("run_status") != "complete":
            die(f"incomplete or wrong schema: {path}")
        if item.get("config", {}).get("schedule") != schedule:
            die(f"schedule/config mismatch: {path}")
        payloads[schedule] = item
    reference_config = {k: v for k, v in payloads["fixed"]["config"].items() if k != "schedule"}
    reference_ids: list[str] | None = None
    reference_baseline: dict[str, tuple] | None = None
    counts: dict[str, int] = {}
    for schedule, item in payloads.items():
        config = {k: v for k, v in item["config"].items() if k != "schedule"}
        if config != reference_config:
            die(f"non-schedule config mismatch: {schedule}")
        videos = item.get("videos", [])
        if len(videos) != item["config"].get("selected_video_count"):
            die(f"checkpoint/video count mismatch: {schedule}")
        ids = [x.get("video_id") for x in videos]
        if len(ids) != len(set(ids)):
            die(f"duplicate video IDs: {schedule}")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            die(f"video ordering/population mismatch: {schedule}")
        baselines: dict[str, tuple] = {}
        for video in videos:
            frames = video.get("redetection_frame_indices", [])
            if video.get("redetection_call_count") != 3 or len(frames) != 3 or len(set(frames)) != 3 or any(int(x) <= 0 for x in frames):
                die(f"invalid three-call schedule: {schedule}/{video.get('video_id')}")
            if video.get("inference_completed_before_gt") is not True:
                die(f"missing inference-before-GT attestation: {schedule}/{video.get('video_id')}")
            if not all(bool(value) for value in video.get("pre_discovery_masks_empty", {}).values()):
                die(f"pre-discovery mask invariant failed: {schedule}/{video.get('video_id')}")
            baselines[video["video_id"]] = (video.get("baseline_objects"), video.get("baseline_metrics"), video.get("baseline_committed_track_ids"))
        if reference_baseline is None:
            reference_baseline = baselines
        elif baselines != reference_baseline:
            die(f"baseline output mismatch across schedules: {schedule}")
        counts[schedule] = len(videos)
    report = dict(schema_version="cpd2-smoke-verify-v1", schedules=list(SCHEDULES), video_counts=counts, exact_baseline_parity=True, three_later_calls_per_video=True, pre_discovery_masks_empty=True, inference_completed_before_gt=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"verified CPD-2 smoke: {counts}")


if __name__ == "__main__":
    main()
