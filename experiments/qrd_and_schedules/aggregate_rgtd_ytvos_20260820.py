#!/usr/bin/env python3
"""Audit and aggregate the frozen full RGTD YouTube-VOS evaluation.

This script deliberately fails if a shard is missing, a video occurs twice, the
two schedules do not cover the same videos, baseline scores differ, or the
three-call budget was not observed.  It reports macro means over videos because
the evaluator's stored values are per-video means.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats


def load_schedule(root: Path, schedule: str) -> tuple[dict, list[Path]]:
    files = sorted(root.glob(f"results_{schedule}_all_shard*of4.json"))
    if len(files) != 4:
        raise RuntimeError(f"{schedule}: expected four shards, found {len(files)}")
    merged: dict = {}
    for path in files:
        with path.open() as f:
            data = json.load(f)
        if data.get("schedule") != schedule or data.get("num_shards") != 4:
            raise RuntimeError(f"{path}: unexpected run metadata")
        values = data.get("per_video", {})
        if data.get("n_videos") != len(values):
            raise RuntimeError(f"{path}: claimed and actual video count differ")
        overlap = set(merged).intersection(values)
        if overlap:
            raise RuntimeError(f"{path}: duplicate videos: {sorted(overlap)[:3]}")
        merged.update(values)
    if len(merged) != 507:
        raise RuntimeError(f"{schedule}: expected 507 videos, found {len(merged)}")
    return merged, files


def runtime_seconds(root: Path, schedule: str) -> int:
    values = []
    for path in sorted(root.glob(f"full_{schedule}_shard*of4_20260820.log")):
        lines = path.read_text(errors="replace").splitlines()
        matches = [re.search(r"\|\s*(\d+)s$", line) for line in lines]
        matches = [m for m in matches if m]
        if not matches:
            raise RuntimeError(f"{path}: no completion duration")
        values.append(int(matches[-1].group(1)))
    if len(values) != 4:
        raise RuntimeError(f"{schedule}: expected four completion logs")
    return sum(values)


def summarize(values: dict) -> dict:
    return {
        "mean_baseline_bijective_jf": float(np.mean([v["bij_JF_A"] for v in values.values()])),
        "mean_schedule_bijective_jf": float(np.mean([v["bij_JF_B"] for v in values.values()])),
        "mean_baseline_greedy_jf": float(np.mean([v["greedy_JF_A"] for v in values.values()])),
        "mean_schedule_greedy_jf": float(np.mean([v["greedy_JF_B"] for v in values.values()])),
        "mean_baseline_tracks": float(np.mean([v["n_track_A"] for v in values.values()])),
        "mean_schedule_tracks": float(np.mean([v["n_track_B"] for v in values.values()])),
        "new_tracks": int(sum(v["n_redet_new"] for v in values.values())),
        "late_recovered": int(sum(v["late_recovered"] for v in values.values())),
        "late_total": int(sum(v["late_total"] for v in values.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--global-dir",
        type=Path,
        help="Directory containing four global-motion shards for the 2026-08-21 mechanism ablation.",
    )
    parser.add_argument(
        "--residual-short-rerun",
        type=Path,
        help="Replacement records for the two short videos affected by the pre-acceptance call-budget correction.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed, fixed_files = load_schedule(args.input_dir, "fixed")
    residual, residual_files = load_schedule(args.input_dir, "residual")
    global_motion = global_files = None
    if args.global_dir:
        global_motion, global_files = load_schedule(args.global_dir, "global")
    amendment = None
    if args.residual_short_rerun:
        with args.residual_short_rerun.open() as f:
            rerun = json.load(f)
        replacements = rerun.get("per_video", {})
        expected = {"209d4e0c67", "67add1386d"}
        if set(replacements) != expected:
            raise RuntimeError("short-video rerun must contain exactly the two pre-specified videos")
        if not set(replacements).issubset(residual):
            raise RuntimeError("short-video rerun includes a video outside the residual aggregate")
        residual.update(replacements)
        amendment = {
            "kind": "two-video residual call-budget correction",
            "source": args.residual_short_rerun.name,
            "video_ids": sorted(replacements),
        }
    if set(fixed) != set(residual):
        raise RuntimeError("fixed/residual video sets differ")
    if global_motion is not None and set(fixed) != set(global_motion):
        raise RuntimeError("fixed/global video sets differ")

    ids = sorted(fixed)
    baseline_mismatches = [
        vid for vid in ids
        if fixed[vid]["bij_JF_A"] != residual[vid]["bij_JF_A"]
        or fixed[vid]["greedy_JF_A"] != residual[vid]["greedy_JF_A"]
        or fixed[vid]["n_track_A"] != residual[vid]["n_track_A"]
    ]
    if global_motion is not None:
        baseline_mismatches.extend(
            vid for vid in ids
            if fixed[vid]["bij_JF_A"] != global_motion[vid]["bij_JF_A"]
            or fixed[vid]["greedy_JF_A"] != global_motion[vid]["greedy_JF_A"]
            or fixed[vid]["n_track_A"] != global_motion[vid]["n_track_A"]
        )
    if baseline_mismatches:
        raise RuntimeError(f"baseline mismatch for {len(baseline_mismatches)} videos")

    schedules = (("fixed", fixed), ("residual", residual))
    if global_motion is not None:
        schedules += (("global", global_motion),)
    for name, records in schedules:
        bad_calls = [vid for vid, v in records.items() if len(v["redet_frames"]) != 3]
        if bad_calls:
            raise RuntimeError(f"{name}: non-three-call videos: {bad_calls[:3]}")

    fixed_stats = summarize(fixed)
    residual_stats = summarize(residual)
    delta = np.array([residual[vid]["bij_JF_B"] - fixed[vid]["bij_JF_B"] for vid in ids])
    statistic, pvalue = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
    rng = np.random.default_rng(20260820)
    bootstrap = np.mean(rng.choice(delta, size=(20000, len(delta)), replace=True), axis=1)
    output = {
        "protocol": "PROTOCOL_RESIDUAL_GUIDED_DISCOVERY_20260820.md",
        "aggregation": "macro mean over 507 matched YouTube-VOS validation videos",
        "coverage": {
            "fixed_videos": len(fixed),
            "residual_videos": len(residual),
            "same_video_ids": True,
            "baseline_identity": "exact stored per-video equality",
            "redetection_calls_per_video": 3,
            "fixed_shards": [p.name for p in fixed_files],
            "residual_shards": [p.name for p in residual_files],
        },
        "fixed": fixed_stats,
        "residual": residual_stats,
        "residual_minus_fixed": {
            "mean_bijective_jf": float(np.mean(delta)),
            "mean_bijective_jf_pp": float(100 * np.mean(delta)),
            "bootstrap_95_ci_pp": [float(100 * np.quantile(bootstrap, 0.025)),
                                   float(100 * np.quantile(bootstrap, 0.975))],
            "wilcoxon_greater": {"statistic": float(statistic), "pvalue": float(pvalue)},
        },
        "runtime_seconds": {
            "original_fixed_full_jobs_total_gpu_seconds": runtime_seconds(args.input_dir, "fixed"),
            "original_residual_full_jobs_total_gpu_seconds": runtime_seconds(args.input_dir, "residual"),
        },
    }
    if amendment:
        output["implementation_amendment"] = amendment
    if global_motion is not None:
        global_stats = summarize(global_motion)
        residual_global_delta = np.array([
            residual[vid]["bij_JF_B"] - global_motion[vid]["bij_JF_B"] for vid in ids
        ])
        global_fixed_delta = np.array([
            global_motion[vid]["bij_JF_B"] - fixed[vid]["bij_JF_B"] for vid in ids
        ])
        def paired_contrast(delta_values: np.ndarray) -> dict:
            stat, p = stats.wilcoxon(delta_values, alternative="greater", zero_method="wilcox")
            samples = np.mean(
                np.random.default_rng(20260821).choice(
                    delta_values, size=(20000, len(delta_values)), replace=True
                ), axis=1,
            )
            return {
                "mean_bijective_jf": float(np.mean(delta_values)),
                "mean_bijective_jf_pp": float(100 * np.mean(delta_values)),
                "bootstrap_95_ci_pp": [
                    float(100 * np.quantile(samples, 0.025)),
                    float(100 * np.quantile(samples, 0.975)),
                ],
                "wilcoxon_greater": {"statistic": float(stat), "pvalue": float(p)},
            }
        output["protocol_followup"] = "PROTOCOL_RGTD_GLOBAL_MOTION_ABLATION_20260821.md"
        output["coverage"]["global_videos"] = len(global_motion)
        output["coverage"]["global_shards"] = [p.name for p in global_files]
        output["global_motion"] = global_stats
        output["global_motion_minus_fixed"] = paired_contrast(global_fixed_delta)
        output["residual_minus_global_motion"] = paired_contrast(residual_global_delta)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
