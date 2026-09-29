#!/usr/bin/env python3
"""Merge only a complete, configuration-identical CPD-1 shard set.

This is deliberately stricter than concatenation: every frozen field must
match, every shard index must be present exactly once, and every video must be
in its deterministic ordinal modulo shard.  The merged JSON is then suitable
for the existing CPD-1 verifier.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def die(message: str) -> None:
    raise SystemExit(f"CPD-1 shard merge error: {message}")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def normal_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items()
            if k not in {"num_shards", "shard_index", "selected_video_count"}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        die(f"refusing to overwrite existing output: {args.output}")

    loaded = [(path, json.loads(path.read_text())) for path in args.inputs]
    if not loaded:
        die("no shard inputs")
    reference_path, reference = loaded[0]
    if reference.get("schema_version") != "cpd1-v1":
        die(f"unexpected schema: {reference_path}")
    frozen = reference.get("frozen_constants")
    config = reference.get("config", {})
    shard_count = config.get("num_shards")
    if not isinstance(shard_count, int) or shard_count < 1:
        die("input does not declare a valid num_shards")
    if len(loaded) != shard_count:
        die(f"expected exactly {shard_count} shards, got {len(loaded)}")

    seen_shards: set[int] = set()
    videos: list[dict[str, Any]] = []
    for path, item in loaded:
        candidate = item.get("config", {})
        if item.get("schema_version") != "cpd1-v1" or item.get("frozen_constants") != frozen:
            die(f"schema/frozen-constant mismatch: {path}")
        if item.get("run_status") != "complete":
            die(f"incomplete shard: {path}")
        if normal_config(candidate) != normal_config(config):
            die(f"non-shard configuration mismatch: {path}")
        if candidate.get("num_shards") != shard_count:
            die(f"shard-count mismatch: {path}")
        shard_index = candidate.get("shard_index")
        if not isinstance(shard_index, int) or not 0 <= shard_index < shard_count or shard_index in seen_shards:
            die(f"invalid or duplicate shard index: {path}")
        seen_shards.add(shard_index)
        shard_videos = item.get("videos", [])
        if len(shard_videos) != candidate.get("selected_video_count"):
            die(f"video count/checkpoint mismatch: {path}")
        for video in shard_videos:
            ordinal = video.get("ordinal")
            if not isinstance(ordinal, int) or (ordinal - 1) % shard_count != shard_index:
                die(f"video outside deterministic shard: {path}/{video.get('video_id')}")
            videos.append(video)
    if seen_shards != set(range(shard_count)):
        die("shard set is incomplete")
    ids = [v.get("video_id") for v in videos]
    ordinals = [v.get("ordinal") for v in videos]
    if len(ids) != len(set(ids)) or len(ordinals) != len(set(ordinals)):
        die("duplicate video ID or source ordinal across shards")
    videos.sort(key=lambda x: x["ordinal"])
    merged_config = normal_config(config)
    merged_config.update(dict(num_shards=1, shard_index=0, selected_video_count=len(videos),
                              merged_from_shards=shard_count))
    payload = dict(schema_version="cpd1-v1", frozen_constants=frozen, config=merged_config,
                   run_status="complete", videos=videos)
    atomic_write(args.output, payload)
    print(f"merged {len(videos)} videos from {shard_count} shards: {args.output}")


if __name__ == "__main__":
    main()
