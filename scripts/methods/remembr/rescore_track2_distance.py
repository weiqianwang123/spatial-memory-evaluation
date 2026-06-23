#!/usr/bin/env python
"""Re-score an existing Track 2 tool-LLM run with distance-based localization.

The original Track 2 eval (commit 51e7da5) was scored at the target object-name
level only, because the ScanEnts3D json carries no 3D bbox. We have since
resolved GT instance bboxes from the ScanNet scan geometry (track2.scannet_bbox)
and rebuilt the benchmark with ``target_bbox_3d``. Rather than re-invoke the LLM
(nondeterministic, ~30 min, and the method emits no instance bbox so the
localization conclusion is structural), this re-scores the *exact* persisted
predictions from ``eval_details.json`` against the new GT, adding distance-based
``acc@0.25m`` / ``acc@0.5m`` and ``mean_center_distance_m`` while preserving the
name-level accuracy.

Usage:
  python scripts/methods/remembr/rescore_track2_distance.py \
    --results-dir results/remembr/track2-tool_llm/remembr-track2-scene0207_00 \
    --benchmark-dir benchmarks/track2/scanents3d/scene0207_00
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.reporting import (
    evaluation_output_paths,
    render_evaluation_report,
    report_metadata,
    report_title,
    write_evaluation_outputs,
)
from spatial_memory_evaluation.track2.data import REFERRING_QUERIES_FILE
from spatial_memory_evaluation.track2.evaluator import (
    _score,
    _scene_objects_by_id,
    _summary_metrics,
)
from spatial_memory_evaluation.tool_llm.runner import _extract_final, _normalize_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score Track 2 run with distance metrics.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output eval_summary.json (default: <results-dir>/eval_summary.json, overwriting).",
    )
    return parser.parse_args()


def reconstruct_predictions(traces_by_query: dict) -> dict[str, list[dict]]:
    """Rebuild predictions_by_query from the final step of each persisted trace."""

    predictions_by_query: dict[str, list[dict]] = {}
    for query_id, steps in traces_by_query.items():
        preds: list[dict] = []
        for step in steps:
            final = _extract_final(step.get("llm_output"), response_kind="predictions")
            if final is not None and isinstance(final.get("predictions"), list):
                preds = [_normalize_prediction(p) for p in final["predictions"]]
        predictions_by_query[query_id] = preds
    return predictions_by_query


def main() -> int:
    args = parse_args()
    details = json.loads((args.results_dir / "eval_details.json").read_text(encoding="utf-8"))
    traces_by_query = details.get("tool_traces_by_query") or {}
    if not traces_by_query:
        print("no tool_traces_by_query in eval_details.json; nothing to re-score", file=sys.stderr)
        return 1

    old_per_query = {p["query_id"]: p for p in details.get("metrics", {}).get("per_query", [])}
    predictions_by_query = reconstruct_predictions(traces_by_query)
    latency_seconds_by_query = {
        qid: float(old_per_query.get(qid, {}).get("latency_ms", 0.0) or 0.0) / 1000.0
        for qid in traces_by_query
    }

    all_rows = read_jsonl(args.benchmark_dir / REFERRING_QUERIES_FILE)
    rows = [r for r in all_rows if str(r["query_id"]) in traces_by_query]
    bbox_rows = sum(1 for r in rows if isinstance(r.get("target_bbox_3d"), list))
    print(f"re-scoring {len(rows)} queries; {bbox_rows} carry target_bbox_3d")

    scene_objects = _scene_objects_by_id(args.benchmark_dir)
    metrics = _score(
        queries=rows,
        scene_objects=scene_objects,
        predictions_by_query=predictions_by_query,
        latency_seconds_by_query=latency_seconds_by_query,
    )

    base = {k: details.get(k) for k in ("status", "track", "mode", "package_dir", "method", "dataset", "explicit_memory")}
    summary = {**base, "status": "ok", "metrics": _summary_metrics(metrics)}
    full = {**base, "status": "ok", "metrics": metrics, "tool_traces_by_query": traces_by_query}

    output = args.output or (args.results_dir / "eval_summary.json")
    paths = evaluation_output_paths(output)
    report = render_evaluation_report(
        title=report_title(summary),
        metadata=report_metadata(summary),
        metrics=summary["metrics"],
        status="ok",
        summary_path=paths.summary,
        details_path=paths.details,
        result=None,
    )
    write_evaluation_outputs(summary_path=output, summary=summary, details=full, report_markdown=report)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
