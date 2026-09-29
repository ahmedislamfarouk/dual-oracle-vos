#!/usr/bin/env python3
"""Schema verification and non-bootstrap CPD-1 aggregation for completed arms."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("inputs", nargs="+", type=Path); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    cells = []
    for path in args.inputs:
        d = json.loads(path.read_text())
        if d.get("schema_version") != "cpd1-v1": raise SystemExit(f"unexpected schema: {path}")
        frozen = d["frozen_constants"]
        if frozen["resolution"] != [854, 480] or frozen["topk"] != 10: raise SystemExit(f"frozen interface mismatch: {path}")
        objects = [o for v in d["videos"] for o in v["objects"]]
        if not objects: raise SystemExit(f"no object records: {path}")
        if not all(v["metrics"].get("query_annotation_available", False) for v in d["videos"]):
            raise SystemExit(f"one or more query frames lack annotation; cannot evaluate query-visible CPD-1: {path}")
        a, b, c = (sum(bool(o[k]) for o in objects) for k in "ABC"); n = len(objects)
        if not (c <= b <= a <= n): raise SystemExit(f"invalid A/B/C nesting: {path}")
        metrics = [v["metrics"] for v in d["videos"]]
        cells.append(dict(input=str(path), proposer=d["config"]["proposer"], dataset=d["config"]["dataset"], videos=len(d["videos"]), objects=n, A=a, B=b, C=c, discovery_loss=(n-a)/n, propagation_loss=(a-b)/n, commitment_loss=(b-c)/n, strict_jf=float(np.mean([x["strict_jf"] for x in metrics])), committed_precision=float(np.mean([x["committed_precision"] for x in metrics])), committed_recall=float(np.mean([x["committed_recall"] for x in metrics])), mean_native_proposals=float(np.mean([v["native_proposal_count"] for v in d["videos"]])), mean_retained_topk=float(np.mean([v["retained_topk_count"] for v in d["videos"]])), mean_committed_tracks=float(np.mean([x["committed_count"] for x in metrics])), structural_single_shot_blind_spot_objects=int(sum(x["structural_single_shot_blind_spot_count"] for x in metrics))))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(dict(schema_version="cpd1-aggregate-v1", cells=cells), indent=2, sort_keys=True))
    print(f"verified {len(cells)} cell(s); wrote {args.output}")
if __name__ == "__main__": main()
