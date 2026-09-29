#!/usr/bin/env python3
"""Frozen CPD-1 video-level bootstrap analysis.

The CPD-1 protocol calls for 20,000 deterministic video-level resamples of
discovery loss minus the larger downstream stage loss.  This script implements
that analysis without opening a manuscript or choosing any method parameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def die(message: str) -> None:
    raise SystemExit(f"CPD-1 bootstrap error: {message}")


def quantile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


EXPECTED_VIDEO_COUNTS = {"ytvos": 507, "davis": 30, "lvos": 50}


def load_cell(path: Path, dataset: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != "cpd1-v1":
        die(f"unexpected schema: {path}")
    # The original AMG driver wrote only a final JSON and predates the
    # checkpointed runner's run_status field.  It is admissible only if the
    # complete-population checks below pass; a present non-complete status is
    # always rejected.
    if data.get("run_status") not in {None, "complete"}:
        die(f"incomplete input: {path}")
    if data.get("config", {}).get("dataset") != dataset:
        die(f"unexpected dataset: {path}")
    videos = data.get("videos", [])
    ids = [v.get("video_id") for v in videos]
    expected_videos = EXPECTED_VIDEO_COUNTS[dataset]
    if len(ids) != expected_videos or len(ids) != len(set(ids)):
        die(f"expected {expected_videos} distinct {dataset} videos: {path}")
    rows: list[tuple[int, int, int, int]] = []
    zero_denominator = 0
    for video in videos:
        if not video.get("metrics", {}).get("query_annotation_available", False):
            die(f"query annotation unavailable: {path}/{video.get('video_id')}")
        objects = video.get("objects", [])
        n = len(objects)
        if n == 0:
            zero_denominator += 1
            continue
        a = sum(bool(x["A"]) for x in objects)
        b = sum(bool(x["B"]) for x in objects)
        c = sum(bool(x["C"]) for x in objects)
        if not (0 <= c <= b <= a <= n):
            die(f"A/B/C nesting failure: {path}/{video.get('video_id')}")
        rows.append((n, a, b, c))
    if not rows:
        die(f"no query-visible-object videos: {path}")
    values = np.asarray(rows, dtype=np.float64)
    # Each eligible video contributes equally to the bootstrap, matching the
    # protocol's specified video-level resampling.  Query-object-free videos
    # have no valid A/B/C denominator and are explicitly reported, not silently
    # folded into a zero loss.
    n, a, b, c = values.T
    stages = np.column_stack(((n - a) / n, (a - b) / n, (b - c) / n))
    return dict(path=str(path), proposer=data["config"]["proposer"],
                eligible_video_count=len(rows), zero_denominator_video_count=zero_denominator,
                total_counts=dict(N=int(n.sum()), A=int(a.sum()), B=int(b.sum()), C=int(c.sum())),
                stage_per_video=stages)


def analyze(cell: dict[str, Any], resamples: int, seed: int) -> dict[str, Any]:
    stage = cell.pop("stage_per_video")
    rng = np.random.default_rng(seed)
    m = len(stage)
    means = stage.mean(axis=0)
    bootstrap = np.empty((resamples, 3), dtype=np.float64)
    for i in range(resamples):
        bootstrap[i] = stage[rng.integers(0, m, size=m)].mean(axis=0)
    difference = bootstrap[:, 0] - np.maximum(bootstrap[:, 1], bootstrap[:, 2])
    point_difference = float(means[0] - max(means[1], means[2]))
    n, a, b, c = (cell["total_counts"][key] for key in ("N", "A", "B", "C"))
    pooled = np.array([(n - a) / n, (a - b) / n, (b - c) / n], dtype=float)
    labels = ("discovery_loss", "propagation_loss", "commitment_loss")
    cell.update(dict(
        pooled_object_level={label: float(value) for label, value in zip(labels, pooled)},
        video_macro={label: float(value) for label, value in zip(labels, means)},
        video_bootstrap_95={label: quantile_interval(bootstrap[:, i]) for i, label in enumerate(labels)},
        discovery_minus_larger_downstream=dict(
            video_macro=point_difference,
            bootstrap_95=quantile_interval(difference),
            bootstrap_fraction_positive=float(np.mean(difference > 0)),
            larger_downstream_stage=("propagation_loss" if means[1] >= means[2] else "commitment_loss"),
            discovery_dominant_at_95=bool(quantile_interval(difference)[0] > 0),
        ),
    ))
    return cell


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=sorted(EXPECTED_VIDEO_COUNTS), default="ytvos",
                   help="frozen population represented by every input cell")
    p.add_argument("--resamples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20260822)
    args = p.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite: {args.output}")
    if args.resamples != 20_000:
        die("CPD-1 protocol requires exactly 20,000 resamples")
    loaded = [load_cell(path, args.dataset) for path in args.inputs]
    proposer_names = [cell["proposer"] for cell in loaded]
    if len(proposer_names) != len(set(proposer_names)):
        die("duplicate proposer input")
    output = dict(schema_version="cpd1-bootstrap-v1", dataset=args.dataset,
                  resamples=args.resamples, seed=args.seed,
                  bootstrap_unit="eligible query-visible-object video",
                  cells=[analyze(cell, args.resamples, args.seed) for cell in loaded])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"analyzed {len(loaded)} proposer cells; wrote {args.output}")


if __name__ == "__main__":
    main()
