"""Scoring one arm of one video: candidate-pool J&F, committed J&F, and the non-overlapping re-scores."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .metrics import VideoScorer, bijective_jf, exclusive_jf, greedy_jf


def score_arm(scorer: VideoScorer, tracks: Dict[str, List[np.ndarray]], jrows: Dict[str, np.ndarray],
              pool: Sequence[str], committed: Sequence[str], nonoverlap: bool = False) -> dict:
    b = bijective_jf(scorer, tracks, jrows, list(committed))
    out = {
        "jf": round(b["jf"], 4),                              # committed tracks, bijective (the headline metric)
        "pool_jf": round(greedy_jf(scorer, tracks, jrows, list(pool)), 4),   # candidate pool, greedy best-match
        "precision": round(b["precision"], 4),
        "recall": round(b["recall"], 4),
        "n_tracks": len(pool),
        "n_committed": len(committed),
    }
    if nonoverlap:
        out["jf_nonoverlap"] = round(exclusive_jf(scorer, tracks, list(committed)), 4)
        out["jf_nonoverlap_top20"] = round(exclusive_jf(scorer, tracks, list(committed), cap=20), 4)
    return out


def summarize(per_video: Dict[str, dict], arms: Sequence[str]) -> dict:
    summary = {"n_videos": len(per_video)}
    for arm in arms:
        rows = [v[arm] for v in per_video.values() if arm in v]
        if rows:
            summary[arm] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], (int, float))}
    return summary
