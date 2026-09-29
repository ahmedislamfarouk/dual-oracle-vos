#!/usr/bin/env python3
"""CPD-2 v3: as v2, plus a --proposer switch over the frozen CPD-1 loaders.

Only the proposer selection differs from v2.  With --proposer detr the saved
scientific output must be byte-identical to the archived v2 arms; that parity is
the gate for using this driver at all.

CPD-2 deterministic, matched-budget offline temporal-discovery runner.

This is a new driver.  It reuses only frozen CPD-1 primitives, never changes
their evaluator, and keeps GT unread until every inference-only schedule action
for a video has completed.  It is not evidence until its smoke/parity gates
pass and its complete raw outputs are archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import cpd1_driver as cpd1


LATER_CALLS = 3
UNCOVERED_MIN = 0.70
RANDOM_SEED = 20260824


def die(message: str) -> None:
    raise SystemExit(f"CPD-2 error: {message}")


def stable_seed(video_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{video_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def temporal_bins(n_frames: int) -> list[np.ndarray]:
    """Three ordered bins over later frames, with no empty-bin fallback."""
    later = np.arange(1, n_frames, dtype=int)
    return [x.astype(int) for x in np.array_split(later, LATER_CALLS) if len(x)]


def motion_scores(frames: list[np.ndarray], baseline_masks: dict[str, list[np.ndarray]], residual: bool) -> dict[int, float]:
    """Offline grayscale frame difference, optionally outside baseline masks."""
    gray = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32) for x in frames]
    scores: dict[int, float] = {}
    for idx in range(1, len(frames)):
        delta = np.abs(gray[idx] - gray[idx - 1])
        if residual:
            covered = np.zeros(delta.shape, dtype=bool)
            for track_masks in baseline_masks.values():
                covered |= track_masks[idx].astype(bool)
            values = delta[~covered]
            scores[idx] = float(values.mean()) if values.size else 0.0
        else:
            scores[idx] = float(delta.mean())
    return scores


def select_frames(schedule: str, frames: list[np.ndarray], baseline_masks: dict[str, list[np.ndarray]], video_id: str, seed: int) -> tuple[list[int], dict[int, float]]:
    """Return exactly one selected later frame per non-empty temporal bin."""
    bins = temporal_bins(len(frames))
    if schedule == "fixed":
        raw = [len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4]
        result: list[int] = []
        for idx in raw:
            if 0 < idx < len(frames) and idx not in result:
                result.append(idx)
        # The normal protocol has three valid calls.  This fallback only makes
        # short-video execution deterministic and auditable.
        if len(result) < len(bins):
            for group in bins:
                candidate = int(group[0])
                if candidate not in result:
                    result.append(candidate)
                if len(result) == len(bins):
                    break
        return sorted(result), {}
    if schedule == "random":
        rng = np.random.default_rng(stable_seed(video_id, seed))
        return [int(rng.choice(group)) for group in bins], {}
    scores = motion_scores(frames, baseline_masks, residual=(schedule == "residual"))
    selected = [max((int(x) for x in group), key=lambda idx: (scores[idx], -idx)) for group in bins]
    return selected, {idx: scores[idx] for idx in selected}


def union_coverage(track_masks: dict[str, list[np.ndarray]], frame_idx: int, shape: tuple[int, int]) -> np.ndarray:
    covered = np.zeros(shape, dtype=bool)
    for values in track_masks.values():
        covered |= values[frame_idx].astype(bool)
    return covered


def uncovered_fraction(box: list[float], coverage: np.ndarray) -> float:
    h, w = coverage.shape
    x1, y1, x2, y2 = [int(round(float(x))) for x in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(max(x2, x1 + 1), w), min(max(y2, y1 + 1), h)
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    return float(1.0 - coverage[y1:y2, x1:x2].mean())


def commit_tracks(selected: list[dict[str, Any]], masks: dict[str, list[np.ndarray]], confidence: dict[str, list[float]]) -> tuple[set[str], dict[str, dict[str, float]]]:
    committed: set[str] = set()
    signals: dict[str, dict[str, float]] = {}
    for item in selected:
        tid = item["track_id"]
        areas = [int(x.sum()) for x in masks[tid]]
        nonempty = [x for x in areas if x]
        signals[tid] = dict(
            persistence=len(nonempty) / len(areas),
            mean_area=(float(np.mean(nonempty)) / (cpd1.TARGET_WH[0] * cpd1.TARGET_WH[1]) if nonempty else 0.0),
            mean_confidence=(float(np.mean(confidence[tid])) if confidence[tid] else 0.0),
        )
        signals[tid]["quality"] = signals[tid]["persistence"] * signals[tid]["mean_confidence"]
    candidates = sorted(
        [tid for tid, s in signals.items() if s["persistence"] >= cpd1.COMMIT_SETTINGS["persistence_min"] and 0 < s["mean_area"] < cpd1.COMMIT_SETTINGS["area_max"]],
        key=lambda tid: (-signals[tid]["quality"], tid),
    )
    for tid in candidates:
        if all(cpd1.temporal_iou(masks[tid], masks[prior]) < cpd1.COMMIT_SETTINGS["dedup_temporal_iou"] for prior in committed):
            committed.add(tid)
    return committed, signals


def write_status(path: Path, result: dict[str, Any], total: int) -> None:
    lines = ["# CPD-2 shard live status", "", f"- Status: **{result.get('run_status', 'running')}**", f"- Completed: **{len(result['videos'])}/{total}** videos", f"- Schedule: `{result['config']['schedule']}`"]
    if result["videos"]:
        latest = result["videos"][-1]
        lines.append(f"- Last completed video: `{latest['video_id']}` ({latest['wall_seconds']:.1f}s)")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["davis", "ytvos", "lvos", "generic"], required=True)
    p.add_argument("--dataset-root", type=Path, required=True); p.add_argument("--split", default="val")
    p.add_argument("--sam2-repo", type=Path, required=True); p.add_argument("--sam2-config", required=True); p.add_argument("--sam2-checkpoint", required=True)
    p.add_argument("--canonical-source", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--schedule", choices=["fixed", "random", "global", "residual"], required=True)
    p.add_argument("--proposer", choices=["detr", "maskrcnn", "amg"], default="detr")
    p.add_argument("--smoke-manifest", type=Path); p.add_argument("--query-frame-index", type=int, default=0)
    p.add_argument("--device", default=os.environ.get("CPD2_DEVICE", "cuda")); p.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluation-workers", type=int, default=1); p.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--shard-index", type=int, default=0); p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--resume", action="store_true"); p.add_argument("--overwrite", action="store_true"); p.add_argument("--status-markdown", type=Path)
    args = p.parse_args()
    if args.query_frame_index != 0: die("CPD-2 fixes the base query to frame index 0")
    if args.evaluation_workers < 1 or args.checkpoint_every < 1: die("worker and checkpoint counts must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards: die("invalid shard index/count")
    for required in (args.dataset_root, args.sam2_repo, args.canonical_source, Path(args.sam2_checkpoint)):
        if not required.exists(): die(f"missing required path: {required}")
    if args.output.exists() and not (args.resume or args.overwrite): die(f"output exists: {args.output}")
    sys.path[:0] = [str(args.sam2_repo), str(args.canonical_source)]
    videos = cpd1.find_videos(args.dataset_root, args.dataset, args.split)
    all_selected = cpd1.select_videos(videos, args.smoke_manifest, 0)
    chosen = [(ordinal, video, query) for ordinal, (video, query) in enumerate(all_selected, 1) if (ordinal - 1) % args.num_shards == args.shard_index]
    if not chosen: die("this shard selects no videos")
    if args.proposer == "detr":
        proposer, proposer_id = cpd1.load_detr(args.device)
    elif args.proposer == "maskrcnn":
        proposer, proposer_id = cpd1.load_maskrcnn(args.device)
    else:
        proposer, proposer_id = cpd1.load_amg(args.device, args.sam2_config, args.sam2_checkpoint)
    frozen = dict(resolution=list(cpd1.TARGET_WH), first_frame_topk=cpd1.TOPK, later_frame_topk=cpd1.TOPK, proposal_threshold=cpd1.PROPOSAL_THRESH, later_calls=LATER_CALLS, uncovered_min=UNCOVERED_MIN, commitment=cpd1.COMMIT_SETTINGS, schedule_policy="offline_baseline_mask_union_v1")
    config = dict(dataset=args.dataset, dataset_root=str(args.dataset_root), split=args.split, proposer=args.proposer, proposer_id=proposer_id, schedule=args.schedule, random_seed=args.random_seed, sam2_checkpoint=str(args.sam2_checkpoint), sam2_checkpoint_sha256=cpd1.sha256(Path(args.sam2_checkpoint)), offload_video_to_cpu=args.offload_video_to_cpu, evaluation_workers=args.evaluation_workers, num_shards=args.num_shards, shard_index=args.shard_index, selected_video_count=len(chosen), query_policy=("explicit_manifest" if args.smoke_manifest else "fixed_frame_index:0"))
    if args.resume:
        result = json.loads(args.output.read_text())
        if result.get("schema_version") != "cpd2-v1" or result.get("frozen_constants") != frozen or result.get("config") != config:
            die("resume output does not exactly match this CPD-2 shard configuration")
    else:
        result = dict(schema_version="cpd2-v1", frozen_constants=frozen, config=config, videos=[])
    completed_ids = {x["video_id"] for x in result["videos"]}
    result["run_status"] = "running"
    if args.status_markdown: write_status(args.status_markdown, result, len(chosen))
    for completed_ordinal, (ordinal, video, query_path) in enumerate((x for x in chosen if x[1].video_id not in completed_ids), len(completed_ids) + 1):
        started = time.time()
        frames = [cv2.resize(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), cpd1.TARGET_WH) for path in video.frame_paths]
        native0 = proposer(frames[0]); native0.sort(key=lambda x: (-x["score"], x["native_id"]))
        baseline = [dict(item, track_id=f"{video.video_id}:base:{i:02d}", query_frame_index=0) for i, item in enumerate(native0[:cpd1.TOPK])]
        base_masks, base_conf = cpd1.run_propagator(frames, baseline, args)
        later_indices, schedule_scores = select_frames(args.schedule, frames, base_masks, video.video_id, args.random_seed)
        if len(later_indices) != LATER_CALLS or len(set(later_indices)) != LATER_CALLS:
            die(f"schedule did not produce exactly {LATER_CALLS} distinct later calls for {video.video_id}: {later_indices}")
        later: list[dict[str, Any]] = []
        later_native_counts: dict[str, int] = {}
        for call_index, frame_idx in enumerate(later_indices):
            native = proposer(frames[frame_idx]); native.sort(key=lambda x: (-x["score"], x["native_id"]))
            later_native_counts[str(frame_idx)] = len(native)
            coverage = union_coverage(base_masks, frame_idx, frames[frame_idx].shape[:2])
            for item in native[:cpd1.TOPK]:
                if uncovered_fraction(item["box"], coverage) >= UNCOVERED_MIN:
                    later.append(dict(item, track_id=f"{video.video_id}:later:{call_index:02d}:{len(later):02d}", query_frame_index=frame_idx))
        new_masks, new_conf = cpd1.run_propagator(frames, later, args)
        pre_discovery_masks_empty: dict[str, bool] = {}
        for item in later:
            for frame_idx in range(item["query_frame_index"]):
                new_masks[item["track_id"]][frame_idx] = np.zeros_like(new_masks[item["track_id"]][frame_idx])
            is_empty = all(not new_masks[item["track_id"]][frame_idx].any() for frame_idx in range(item["query_frame_index"]))
            if not is_empty:
                die(f"pre-discovery mask invariant failed for {video.video_id}/{item['track_id']}")
            pre_discovery_masks_empty[item["track_id"]] = is_empty
        masks = dict(base_masks); masks.update(new_masks)
        confidence = dict(base_conf); confidence.update(new_conf)
        base_committed, base_signals = commit_tracks(baseline, base_masks, base_conf)
        all_tracks = baseline + later
        committed, signals = commit_tracks(all_tracks, masks, confidence)
        # GT is read only after baseline and every schedule-specific inference
        # action above has completed.
        base_objects, base_metrics = cpd1.evaluate(video, 0, baseline, base_masks, base_committed, evaluation_workers=args.evaluation_workers)
        objects, metrics = cpd1.evaluate(video, 0, all_tracks, masks, committed, evaluation_workers=args.evaluation_workers)
        result["videos"].append(dict(video_id=video.video_id, ordinal=ordinal, frame_names=[x.name for x in video.frame_paths], query_frame=query_path.name, query_frame_index=0, schedule=args.schedule, redetection_frame_indices=later_indices, redetection_call_count=len(later_indices), redetection_selection_scores={str(k): v for k, v in schedule_scores.items()}, later_native_proposal_counts=later_native_counts, baseline_native_proposal_count=len(native0), baseline_retained_topk_count=len(baseline), later_admitted_track_count=len(later), pre_discovery_masks_empty=pre_discovery_masks_empty, inference_completed_before_gt=True, baseline_track_signals=base_signals, track_signals=signals, baseline_committed_track_ids=sorted(base_committed), committed_track_ids=sorted(committed), baseline_objects=base_objects, objects=objects, baseline_metrics=base_metrics, metrics=metrics, wall_seconds=time.time() - started))
        print(f"[{completed_ordinal}/{len(chosen)}] {video.video_id}: schedule={args.schedule} calls={later_indices} admitted={len(later)}", flush=True)
        if completed_ordinal % args.checkpoint_every == 0 or completed_ordinal == len(chosen):
            cpd1.write_json_atomic(args.output, result)
            if args.status_markdown: write_status(args.status_markdown, result, len(chosen))
    result["run_status"] = "complete"
    cpd1.write_json_atomic(args.output, result)
    if args.status_markdown: write_status(args.status_markdown, result, len(chosen))


if __name__ == "__main__":
    main()
