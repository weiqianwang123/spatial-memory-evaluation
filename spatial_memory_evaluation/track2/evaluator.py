from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl, write_json, write_jsonl
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.matching import match_objects, mean, median, safe_div
from spatial_memory_evaluation.common.package_io import (
    copy_package_to_sandbox,
    fixed_api_capability,
    invalid_result,
    load_entrypoint,
    load_package,
    run_agent_command,
)
from spatial_memory_evaluation.output_paths import timestamped_result_dir


TRACK_KEY = "track2_object_location"
SPLITS = ("all_annotated", "detector_coverable")
K_VALUES = (1, 5, 10)


def evaluate_track2(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    track1_benchmark_dir: Path,
    mode: str,
    output: Path | None,
    agent_command: str | None = None,
    agent_output: Path | None = None,
    sandbox_root: Path | None = None,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track2-{mode}") / "eval_summary.json"

    aliases = load_aliases(benchmark_dir / "label_aliases.json")
    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, capabilities, benchmark_dir, method)
    elif mode == "agentic_memory_only":
        result = _run_agentic(
            package_dir=package_dir,
            benchmark_dir=benchmark_dir,
            output=output,
            agent_command=agent_command,
            agent_output=agent_output,
            sandbox_root=sandbox_root,
        )
    else:
        raise ValueError(f"unknown Track 2 mode: {mode}")

    summary: dict[str, Any] = {
        "status": result["status"],
        "track": "track2_object_location",
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
    }
    if result["status"] == "ok":
        predictions_by_query = result["predictions_by_query"]
        latencies = result.get("latency_seconds_by_query", {})
        summary["splits"] = {
            split: _score_split(
                gt_objects=read_jsonl(track1_benchmark_dir / f"{split}.jsonl"),
                queries=read_jsonl(benchmark_dir / f"queries_{split}.jsonl"),
                predictions_by_query=predictions_by_query,
                latency_seconds_by_query=latencies,
                aliases=aliases,
            )
            for split in SPLITS
        }
    else:
        summary["result"] = result
    write_json(output, summary)
    return summary


def _run_fixed_api(
    package_dir: Path,
    capabilities: dict[str, Any],
    benchmark_dir: Path,
    method: str,
) -> dict[str, Any]:
    cap = fixed_api_capability(capabilities, TRACK_KEY)
    if cap.get("status") != "supported":
        return invalid_result(
            method=method,
            package_dir=package_dir,
            track_key=TRACK_KEY,
            reason=str(cap.get("reason") or ""),
        )
    query_object = load_entrypoint(package_dir, str(cap["entrypoint"]))
    predictions_by_query: dict[str, list[dict[str, Any]]] = {}
    latency_seconds_by_query: dict[str, float] = {}
    for split in SPLITS:
        for query in read_jsonl(benchmark_dir / f"queries_{split}.jsonl"):
            query_id = str(query["query_id"])
            started = time.perf_counter()
            result = query_object(str(package_dir), {"query": query["query"], "top_k": int(query["top_k"])})
            latency_seconds_by_query[query_id] = time.perf_counter() - started
            predictions = result.get("predictions") if isinstance(result, dict) else None
            predictions_by_query[query_id] = predictions if isinstance(predictions, list) else []
    return {
        "status": "ok",
        "predictions_by_query": predictions_by_query,
        "latency_seconds_by_query": latency_seconds_by_query,
    }


def _run_agentic(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    output: Path,
    agent_command: str | None,
    agent_output: Path | None,
    sandbox_root: Path | None,
) -> dict[str, Any]:
    if agent_output is not None:
        return _load_agent_predictions(agent_output)
    if not agent_command:
        return {
            "status": "error",
            "message": "agentic_memory_only requires --agent-output or --agent-command",
        }
    sandbox_root = sandbox_root or (output.parent / "agent_sandbox")
    sandbox_root.mkdir(parents=True, exist_ok=True)
    sandbox_package = copy_package_to_sandbox(package_dir, sandbox_root)
    sandbox_queries = sandbox_root / "queries"
    _write_agent_query_files(benchmark_dir, sandbox_queries)
    prompt_path = sandbox_root / "track2_prompt.md"
    agent_output_path = sandbox_root / "track2_agent_output.json"
    prompt_path.write_text(_track2_prompt(sandbox_package, sandbox_queries), encoding="utf-8")
    run_agent_command(
        agent_command=agent_command,
        prompt_path=prompt_path,
        sandbox_dir=sandbox_root,
        output_path=agent_output_path,
    )
    return _load_agent_predictions(agent_output_path)


def _load_agent_predictions(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    predictions = value.get("predictions_by_query") if isinstance(value, dict) else None
    if not isinstance(predictions, dict):
        return {"status": "error", "message": f"agent output missing predictions_by_query: {path}"}
    return {"status": "ok", "predictions_by_query": predictions, "latency_seconds_by_query": {}}


def _write_agent_query_files(benchmark_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = []
        for query in read_jsonl(benchmark_dir / f"queries_{split}.jsonl"):
            rows.append(
                {
                    "query_id": query["query_id"],
                    "scene_id": query["scene_id"],
                    "split": query["split"],
                    "canonical_label": query["canonical_label"],
                    "query": query["query"],
                    "top_k": query["top_k"],
                }
            )
        write_jsonl(output_dir / f"queries_{split}.jsonl", rows)


def _score_split(
    *,
    gt_objects: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    predictions_by_query: dict[str, list[dict[str, Any]]],
    latency_seconds_by_query: dict[str, float],
    aliases: dict[str, str],
) -> dict[str, Any]:
    gt_by_id = {str(obj["gt_id"]): obj for obj in gt_objects}
    totals = {
        k: {"matched_targets": 0, "target_count": 0, "success_count": 0}
        for k in K_VALUES
    }
    reciprocal_ranks: list[float] = []
    first_hit_distances: list[float] = []
    per_query = []

    for query in queries:
        query_id = str(query["query_id"])
        targets = [gt_by_id[gt_id] for gt_id in query["target_gt_ids"] if gt_id in gt_by_id]
        predictions = predictions_by_query.get(query_id, [])
        query_row = {"query_id": query_id, "target_count": len(targets)}
        first_rank = None
        first_distance = None
        for k in K_VALUES:
            matches, _, _, _ = match_objects(targets, predictions[:k], aliases)
            matched_count = len({match.gt_id for match in matches})
            totals[k]["matched_targets"] += matched_count
            totals[k]["target_count"] += len(targets)
            if matched_count > 0:
                totals[k]["success_count"] += 1
                if first_rank is None:
                    first_rank = _first_hit_rank(targets, predictions[:k], aliases)
                    first_distance = min(match.distance_m for match in matches)
            query_row[f"recall@{k}"] = safe_div(matched_count, len(targets))
            query_row[f"success@{k}"] = matched_count > 0
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        if first_distance is not None:
            first_hit_distances.append(first_distance)
        query_row["first_hit_rank"] = first_rank
        query_row["first_hit_distance_m"] = first_distance
        query_row["latency_ms"] = latency_seconds_by_query.get(query_id, 0.0) * 1000.0
        per_query.append(query_row)

    latencies = [latency_seconds_by_query.get(str(query["query_id"]), 0.0) for query in queries]
    return {
        "query_count": len(queries),
        **{
            f"recall@{k}": safe_div(totals[k]["matched_targets"], totals[k]["target_count"])
            for k in K_VALUES
        },
        **{
            f"success@{k}": safe_div(totals[k]["success_count"], len(queries))
            for k in K_VALUES
        },
        "mrr": mean(reciprocal_ranks),
        "mean_first_hit_distance_m": mean(first_hit_distances),
        "mean_query_latency_ms": (mean(latencies) or 0.0) * 1000.0,
        "median_query_latency_ms": (median(latencies) or 0.0) * 1000.0,
        "p95_query_latency_ms": percentile(latencies, 95) * 1000.0,
        "total_query_runtime_seconds": sum(latencies),
        "queries_per_second": safe_div(len(queries), sum(latencies)),
        "per_query": per_query,
    }


def _first_hit_rank(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    aliases: dict[str, str],
) -> int | None:
    for index in range(len(predictions)):
        matches, _, _, _ = match_objects(targets, predictions[: index + 1], aliases)
        if matches:
            return index + 1
    return None


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * percentile_value / 100.0
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _track2_prompt(package_dir: Path, query_dir: Path) -> str:
    return f"""You are evaluating a spatial memory package in memory-only mode.

Package directory: {package_dir}
Query files are in: {query_dir}

Read only the package and the query files. Do not use raw_links or external
source data. Return JSON with this exact shape:

{{
  "predictions_by_query": {{
    "query_id": [
      {{
        "object_id": "string",
        "label": "chair",
        "position_3d": [0.0, 0.0, 0.0],
        "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "score": 0.0,
        "evidence": []
      }}
    ]
  }}
}}
"""
