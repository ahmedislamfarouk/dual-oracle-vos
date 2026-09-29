#!/usr/bin/env python3
"""Frozen CPD-2 paired video-level bootstrap analysis.

Written and hashed BEFORE any confirmatory CPD-2 result existed, so the
contrasts below are pre-specified rather than chosen after looking at the
numbers.  It opens no manuscript and selects no method parameter.

Pairing: every arm ran the identical video population under the identical
frozen interface, differing only in which three later frames the detector is
called on, so each bootstrap replicate resamples *video indices* once and
applies the same indices to every arm.  Unpaired resampling would discard the
variance reduction that the matched design exists to provide.

Pre-specified contrasts
-----------------------
1. Primary: `residual - fixed` on video-macro strict committed J&F.  `residual`
   is the proposed label-free adaptive policy; `fixed` is the simplest
   deterministic equal-budget control.  The protocol's decision rule is met for
   this contrast only if the 95% interval excludes zero with a positive point
   estimate.
2. Secondary, pre-specified: `residual - random`, `residual - global`.
3. Budget check, pre-specified: every arm minus the shared first-frame-only
   baseline, i.e. whether three later detector calls help at all.
4. Reported for completeness: all remaining pairwise arm contrasts.

Each contrast is reported on strict committed J&F and, because the mechanism
claim is about discovery, on per-video discovery loss as well.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_VIDEO_COUNTS = {"ytvos": 507, "davis": 30, "lvos": 50}
RESAMPLES = 20_000
SEED = 20260824
PRIMARY = ("residual", "fixed")
SECONDARY = (("residual", "random"), ("residual", "global"))


def die(message: str) -> None:
    raise SystemExit(f"CPD-2 bootstrap error: {message}")


def quantile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def discovery_loss(objects: list[dict[str, Any]]) -> float | None:
    n = len(objects)
    if n == 0:
        return None
    return (n - sum(bool(x["A"]) for x in objects)) / n


def load_arm(path: Path, dataset: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != "cpd2-v1":
        die(f"unexpected schema: {path}")
    if data.get("run_status") != "complete":
        die(f"incomplete input: {path}")
    if data.get("config", {}).get("dataset") != dataset:
        die(f"unexpected dataset: {path}")
    videos = data["videos"]
    if len(videos) != EXPECTED_VIDEO_COUNTS[dataset]:
        die(f"expected {EXPECTED_VIDEO_COUNTS[dataset]} videos: {path}")
    return dict(schedule=data["config"]["schedule"],
                by_id={v["video_id"]: v for v in videos})


def summarize(replicates: np.ndarray, point: float) -> dict[str, Any]:
    interval = quantile_interval(replicates)
    return dict(video_macro=point, bootstrap_95=interval,
                bootstrap_fraction_positive=float(np.mean(replicates > 0)),
                resolved_at_95=bool(interval[0] > 0 or interval[1] < 0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=sorted(EXPECTED_VIDEO_COUNTS), required=True)
    p.add_argument("--resamples", type=int, default=RESAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite: {args.output}")
    if args.resamples != RESAMPLES:
        die(f"CPD-2 protocol requires exactly {RESAMPLES} resamples")

    arms = [load_arm(path, args.dataset) for path in args.inputs]
    names = [arm["schedule"] for arm in arms]
    if len(names) != len(set(names)):
        die("duplicate schedule input")
    if set(names) != {"fixed", "random", "global", "residual"}:
        die(f"expected all four pre-specified arms, got {sorted(names)}")
    by_name = {arm["schedule"]: arm["by_id"] for arm in arms}

    video_ids = sorted(by_name["fixed"])
    for name, by_id in by_name.items():
        if sorted(by_id) != video_ids:
            die(f"arm {name} has a different video set")

    # Shared, schedule-independent baseline; taken from `fixed` and checked
    # equal on every other arm, so no arm's baseline can silently drift.
    baseline_jf = np.array([by_name["fixed"][v]["baseline_metrics"]["strict_jf"] for v in video_ids])
    baseline_disc = [discovery_loss(by_name["fixed"][v]["baseline_objects"]) for v in video_ids]
    for name, by_id in by_name.items():
        other = np.array([by_id[v]["baseline_metrics"]["strict_jf"] for v in video_ids])
        if not np.array_equal(other, baseline_jf):
            die(f"arm {name} has a different first-frame-only baseline; arms are not comparable")

    jf = {name: np.array([by_id[v]["metrics"]["strict_jf"] for v in video_ids])
          for name, by_id in by_name.items()}
    disc_raw = {name: [discovery_loss(by_id[v]["objects"]) for v in video_ids]
                for name, by_id in by_name.items()}
    eligible = np.array([i for i, x in enumerate(baseline_disc)
                         if x is not None and all(disc_raw[n][i] is not None for n in by_name)])
    disc = {name: np.array([disc_raw[name][i] for i in eligible]) for name in by_name}
    baseline_disc_arr = np.array([baseline_disc[i] for i in eligible])

    rng = np.random.default_rng(args.seed)
    m, k = len(video_ids), len(eligible)
    jf_idx = rng.integers(0, m, size=(args.resamples, m))
    disc_idx = rng.integers(0, k, size=(args.resamples, k))

    jf_means = {name: values[jf_idx].mean(axis=1) for name, values in jf.items()}
    jf_means["_baseline"] = baseline_jf[jf_idx].mean(axis=1)
    disc_means = {name: values[disc_idx].mean(axis=1) for name, values in disc.items()}
    disc_means["_baseline"] = baseline_disc_arr[disc_idx].mean(axis=1)

    def contrast(left: str, right: str) -> dict[str, Any]:
        return dict(
            contrast=f"{left} - {right}",
            strict_jf=summarize(jf_means[left] - jf_means[right],
                                float(jf[left].mean() if left != "_baseline" else baseline_jf.mean())
                                - float(jf[right].mean() if right != "_baseline" else baseline_jf.mean())),
            discovery_loss=summarize(disc_means[left] - disc_means[right],
                                     float(disc[left].mean() if left != "_baseline" else baseline_disc_arr.mean())
                                     - float(disc[right].mean() if right != "_baseline" else baseline_disc_arr.mean())),
        )

    order = ["fixed", "random", "global", "residual"]
    payload = dict(
        schema_version="cpd2-bootstrap-v1", dataset=args.dataset,
        resamples=args.resamples, seed=args.seed,
        bootstrap_unit="video, paired across arms",
        video_count=m, discovery_eligible_video_count=k,
        arm_means={name: dict(strict_jf=float(jf[name].mean()),
                              discovery_loss=float(disc[name].mean())) for name in order},
        baseline_mean=dict(strict_jf=float(baseline_jf.mean()),
                           discovery_loss=float(baseline_disc_arr.mean())),
        primary=contrast(*PRIMARY),
        secondary=[contrast(a, b) for a, b in SECONDARY],
        versus_baseline=[contrast(name, "_baseline") for name in order],
        all_pairwise=[contrast(a, b) for a, b in combinations(order, 2)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    primary = payload["primary"]["strict_jf"]
    print(f"paired bootstrap over {m} videos ({k} discovery-eligible); wrote {args.output}")
    print(f"primary {payload['primary']['contrast']} strict J&F: "
          f"{primary['video_macro']:+.4f} 95% [{primary['bootstrap_95'][0]:+.4f}, {primary['bootstrap_95'][1]:+.4f}] "
          f"resolved={primary['resolved_at_95']}")


if __name__ == "__main__":
    main()
