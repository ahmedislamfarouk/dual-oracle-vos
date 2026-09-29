#!/usr/bin/env python3
"""Schema/integrity verification and non-bootstrap aggregation for CPD-2 arms.

This is the acceptance gate that a completed CPD-2 arm set must pass before any
bootstrap, table, or manuscript sentence.  It re-checks on the full population
every invariant the five-video DAVIS smoke checked, and adds the cross-arm
check that the smoke made possible: the baseline arm is schedule-independent,
so every arm must report an identical baseline for every video.  A difference
there means the arms are not comparable and no contrast between them is valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_VIDEO_COUNTS = {"ytvos": 507, "davis": 30, "lvos": 50}
LATER_CALLS = 3


def die(message: str) -> None:
    raise SystemExit(f"CPD-2 verification error: {message}")


def shard_free_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items()
            if k not in {"num_shards", "shard_index", "selected_video_count",
                         "merged_from_shards", "schedule"}}


def stage_counts(objects: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    n = len(objects)
    a = sum(bool(x["A"]) for x in objects)
    b = sum(bool(x["B"]) for x in objects)
    c = sum(bool(x["C"]) for x in objects)
    if not (0 <= c <= b <= a <= n):
        die(f"A/B/C nesting failure over {n} objects")
    return n, a, b, c


def check_arm(path: Path, dataset: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != "cpd2-v1":
        die(f"unexpected schema: {path}")
    if data.get("run_status") != "complete":
        die(f"incomplete input: {path}")
    config = data.get("config", {})
    if config.get("dataset") != dataset:
        die(f"unexpected dataset: {path}")
    frozen = data["frozen_constants"]
    if frozen["resolution"] != [854, 480] or frozen["first_frame_topk"] != 10 \
            or frozen["later_frame_topk"] != 10 or frozen["later_calls"] != LATER_CALLS \
            or frozen["uncovered_min"] != 0.7:
        die(f"frozen interface mismatch: {path}")
    videos = data.get("videos", [])
    ids = [v["video_id"] for v in videos]
    expected = EXPECTED_VIDEO_COUNTS[dataset]
    if len(ids) != expected or len(ids) != len(set(ids)):
        die(f"expected {expected} distinct {dataset} videos, got {len(ids)} ({len(set(ids))} distinct): {path}")

    for video in videos:
        vid = video["video_id"]
        frame_count = len(video["frame_names"])
        calls = video["redetection_frame_indices"]
        if video["redetection_call_count"] != LATER_CALLS or len(calls) != LATER_CALLS \
                or len(set(calls)) != LATER_CALLS:
            die(f"later-call budget violated: {path}/{vid}: {calls}")
        if not all(0 < int(i) < frame_count for i in calls):
            die(f"later call outside video: {path}/{vid}: {calls}")
        if video.get("inference_completed_before_gt") is not True:
            die(f"GT-before-inference guard missing: {path}/{vid}")
        empties = video.get("pre_discovery_masks_empty", {})
        if len(empties) != video["later_admitted_track_count"] or not all(empties.values()):
            die(f"pre-discovery mask invariant failed: {path}/{vid}")
        if not video["metrics"].get("query_annotation_available", False) \
                or not video["baseline_metrics"].get("query_annotation_available", False):
            die(f"query annotation unavailable: {path}/{vid}")
        stage_counts(video["objects"])
        stage_counts(video["baseline_objects"])

    metrics = [v["metrics"] for v in videos]
    base_metrics = [v["baseline_metrics"] for v in videos]
    totals = np.array([stage_counts(v["objects"]) for v in videos], dtype=np.float64).sum(axis=0)
    n, a, b, c = totals
    return dict(
        input=str(path), schedule=config["schedule"], dataset=dataset,
        videos=len(videos), objects=int(n),
        pooled_discovery_loss=float((n - a) / n), pooled_propagation_loss=float((a - b) / n),
        pooled_commitment_loss=float((b - c) / n),
        strict_jf=float(np.mean([x["strict_jf"] for x in metrics])),
        baseline_strict_jf=float(np.mean([x["strict_jf"] for x in base_metrics])),
        committed_precision=float(np.mean([x["committed_precision"] for x in metrics])),
        committed_recall=float(np.mean([x["committed_recall"] for x in metrics])),
        baseline_committed_recall=float(np.mean([x["committed_recall"] for x in base_metrics])),
        mean_committed_tracks=float(np.mean([x["committed_count"] for x in metrics])),
        baseline_mean_committed_tracks=float(np.mean([x["committed_count"] for x in base_metrics])),
        mean_admitted_later_tracks=float(np.mean([v["later_admitted_track_count"] for v in videos])),
        videos_with_no_admitted_track=int(sum(v["later_admitted_track_count"] == 0 for v in videos)),
        mean_wall_seconds=float(np.mean([v["wall_seconds"] for v in videos])),
        _videos={v["video_id"]: v for v in videos},
        _config=shard_free_config(config), _frozen=frozen,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=sorted(EXPECTED_VIDEO_COUNTS), required=True)
    args = p.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite: {args.output}")

    cells = [check_arm(path, args.dataset) for path in args.inputs]
    schedules = [cell["schedule"] for cell in cells]
    if len(schedules) != len(set(schedules)):
        die("duplicate schedule input")

    reference = cells[0]
    for cell in cells[1:]:
        if cell["_frozen"] != reference["_frozen"]:
            die(f"frozen constants differ between arms: {cell['input']}")
        if cell["_config"] != reference["_config"]:
            die(f"non-schedule configuration differs between arms: {cell['input']}")
        if set(cell["_videos"]) != set(reference["_videos"]):
            die(f"arm video sets differ: {cell['input']}")

    # The baseline is first-frame-only and cannot depend on the schedule.
    baseline_mismatches: list[str] = []
    for vid, ref_video in reference["_videos"].items():
        for cell in cells[1:]:
            other = cell["_videos"][vid]
            if other["baseline_metrics"] != ref_video["baseline_metrics"] \
                    or other["baseline_objects"] != ref_video["baseline_objects"] \
                    or other["baseline_committed_track_ids"] != ref_video["baseline_committed_track_ids"]:
                baseline_mismatches.append(f"{cell['schedule']}:{vid}")
    exact_baseline_parity = not baseline_mismatches

    for cell in cells:
        for key in ("_videos", "_config", "_frozen"):
            cell.pop(key)
    payload = dict(schema_version="cpd2-aggregate-v1", dataset=args.dataset,
                   arms=sorted(schedules), exact_baseline_parity=exact_baseline_parity,
                   baseline_mismatch_count=len(baseline_mismatches),
                   baseline_mismatch_examples=sorted(baseline_mismatches)[:20],
                   cells=cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"verified {len(cells)} arm(s) on {args.dataset}; baseline parity "
          f"{'exact' if exact_baseline_parity else f'FAILED ({len(baseline_mismatches)} videos)'}; wrote {args.output}")
    if not exact_baseline_parity:
        die("arms are not comparable: schedule-independent baseline differs between arms")


if __name__ == "__main__":
    main()
