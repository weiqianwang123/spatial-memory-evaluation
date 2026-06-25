"""Track 1: object-level location query + build-cost accounting.

This merges the old Track 1 (memory construction) and old Track 2 (object
location query) into one track. It scores category-level object-location queries
against the exported memory and, in the same summary, reports the build-cost
half: native memory size, package size, frame count, time per frame, and peak
RAM/VRAM from build accounting.

Modes:
- ``fixed_api``: call the package's declared ``track1_object_location`` entrypoint
  (``query_object``-style) per query.
- ``tool_llm``: per-query LLM + method-native retrieval tools.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.matching import (
    euclidean_distance,
    match_objects,
    mean,
    median,
    safe_div,
)
from spatial_memory_evaluation.common.package_io import (
    dir_size_bytes,
    fixed_api_capability,
    invalid_result,
    linked_raw_size_bytes,
    load_entrypoint,
    load_package,
)
from spatial_memory_evaluation.common.reporting import (
    evaluation_output_paths,
    render_evaluation_report,
    report_metadata,
    report_title,
    write_evaluation_outputs,
)
from spatial_memory_evaluation.output_paths import timestamped_result_dir
from spatial_memory_evaluation.tool_llm import run_tool_llm_query


TRACK_KEY = "track1_object_location"
SPLITS = ("detector_coverable",)
K_VALUES = (1, 5)
# Relaxed proximity thresholds (meters). The strict success@k uses match_threshold
# (~0.5-2m, object-size scaled). Caption-memory methods (ReMEmbR) emit the robot
# VIEWPOINT near an object, not the object center, so they score ~0 strictly. These
# relaxed thresholds give partial credit for "the method pointed within X m of the
# right object" — a fairer localization-coarseness view for viewpoint-based memory.
PROXIMITY_THRESHOLDS_M = (1.0, 3.0, 5.0)


def evaluate_track1(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    mode: str,
    output: Path | None,
    llm_command: str | None = None,
    max_tool_iterations: int = 3,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track1-{mode}") / "eval_summary.json"

    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, manifest, capabilities, benchmark_dir, method)
    elif mode == "tool_llm":
        result = _run_tool_llm(
            package_dir=package_dir,
            manifest=manifest,
            benchmark_dir=benchmark_dir,
            output=output,
            llm_command=llm_command,
            max_tool_iterations=max_tool_iterations,
        )
    else:
        raise ValueError(f"unknown Track 1 mode: {mode}")

    memory_size = _memory_size_summary(package_dir, manifest)
    build_runtime_seconds = _build_runtime_seconds(package_dir, manifest)
    base_summary: dict[str, Any] = {
        "status": result["status"],
        "track": TRACK_KEY,
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "explicit_memory": manifest.get("explicit_memory"),
        "native_memory_size_bytes": memory_size["native_memory_size_bytes"],
        "package_size_bytes": memory_size["package_size_bytes"],
        "memory_artifact_size_bytes": memory_size["memory_artifact_size_bytes"],
        "build_runtime_seconds": build_runtime_seconds,
        "time_per_frame_seconds": memory_size.get("time_per_frame_seconds"),
        "peak_ram_bytes": memory_size.get("peak_ram_bytes"),
        "peak_vram_bytes": memory_size.get("peak_vram_bytes"),
    }

    if result["status"] == "ok":
        aliases = load_aliases(benchmark_dir / "label_aliases.json")
        predictions_by_query = result["predictions_by_query"]
        latencies = result.get("latency_seconds_by_query", {})
        full_splits = {
            split: _score_split(
                gt_objects=read_jsonl(benchmark_dir / f"{split}.jsonl"),
                queries=read_jsonl(benchmark_dir / f"queries_{split}.jsonl"),
                predictions_by_query=predictions_by_query,
                latency_seconds_by_query=latencies,
                aliases=aliases,
            )
            for split in SPLITS
        }
        summary = {
            **base_summary,
            "metrics": _track1_summary_metrics(full_splits["detector_coverable"]),
        }
        details = {
            **base_summary,
            "memory_size": memory_size,
            "splits": full_splits,
        }
        if result.get("tool_traces_by_query") is not None:
            details["tool_traces_by_query"] = result.get("tool_traces_by_query")
        if result.get("query_errors_by_query") is not None:
            details["query_errors_by_query"] = result.get("query_errors_by_query")
    else:
        summary = {**base_summary, "result": result}
        details = {**base_summary, "memory_size": memory_size, "result": result}

    paths = evaluation_output_paths(output)
    report = render_evaluation_report(
        title=report_title(summary),
        metadata=report_metadata(summary),
        metrics=summary.get("metrics") if isinstance(summary.get("metrics"), dict) else None,
        status=str(summary["status"]),
        summary_path=paths.summary,
        details_path=paths.details,
        result=result if result["status"] != "ok" else None,
    )
    write_evaluation_outputs(summary_path=output, summary=summary, details=details, report_markdown=report)
    return summary


def _track1_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "query_count",
        "success@1",
        "success@5",
        "recall@1",
        "recall@5",
        "mrr",
        "mean_first_hit_distance_m",
        "proximity@1.0m",
        "proximity@3.0m",
        "proximity@5.0m",
        "proximity_top1@1.0m",
        "proximity_top1@3.0m",
        "proximity_top1@5.0m",
        "mean_query_latency_ms",
    )
    return {key: metrics.get(key) for key in keys}


def _run_fixed_api(
    package_dir: Path,
    manifest: dict[str, Any],
    capabilities: dict[str, Any],
    benchmark_dir: Path,
    method: str,
) -> dict[str, Any]:
    cap = fixed_api_capability(capabilities, TRACK_KEY)
    if cap.get("status") != "supported":
        method_meta = manifest.get("method") if isinstance(manifest.get("method"), dict) else {}
        return invalid_result(
            method=method,
            package_dir=package_dir,
            track_key=TRACK_KEY,
            reason=str(cap.get("reason") or ""),
            explicit_memory=manifest.get("explicit_memory"),
            method_family=method_meta.get("family"),
        )
    query_object = load_entrypoint(package_dir, str(cap["entrypoint"]))
    predictions_by_query: dict[str, list[dict[str, Any]]] = {}
    latency_seconds_by_query: dict[str, float] = {}
    for split in SPLITS:
        for query in read_jsonl(benchmark_dir / f"queries_{split}.jsonl"):
            query_id = str(query["query_id"])
            target_label = query.get("target_label") or query.get("canonical_label")
            started = time.perf_counter()
            result = query_object(
                str(package_dir),
                {
                    "query": query["query"],
                    "target_label": target_label,
                    "canonical_label": query.get("canonical_label"),
                    "top_k": int(query["top_k"]),
                },
            )
            latency_seconds_by_query[query_id] = time.perf_counter() - started
            predictions = result.get("predictions") if isinstance(result, dict) else None
            predictions_by_query[query_id] = predictions if isinstance(predictions, list) else []
    return {
        "status": "ok",
        "predictions_by_query": predictions_by_query,
        "latency_seconds_by_query": latency_seconds_by_query,
    }


def _run_tool_llm(
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    benchmark_dir: Path,
    output: Path,
    llm_command: str | None,
    max_tool_iterations: int,
) -> dict[str, Any]:
    if not llm_command:
        return {"status": "error", "message": "tool_llm mode requires --llm-command"}

    work_dir = output.parent / "tool_llm_traces"
    work_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_query: dict[str, list[dict[str, Any]]] = {}
    latency_seconds_by_query: dict[str, float] = {}
    tool_traces_by_query: dict[str, Any] = {}
    query_errors_by_query: dict[str, Any] = {}

    for split in SPLITS:
        for query in read_jsonl(benchmark_dir / f"queries_{split}.jsonl"):
            query_id = str(query["query_id"])
            try:
                query_result = run_tool_llm_query(
                    package_dir=package_dir,
                    manifest=manifest,
                    query={
                        "query_id": query["query_id"],
                        "scene_id": query["scene_id"],
                        "split": query["split"],
                        "canonical_label": query["canonical_label"],
                        "target_label": query.get("target_label") or query["canonical_label"],
                        "query": query["query"],
                        "top_k": query["top_k"],
                    },
                    llm_command=llm_command,
                    work_dir=work_dir,
                    max_tool_iterations=max_tool_iterations,
                )
            except Exception as exc:
                query_result = {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "predictions": [],
                    "latency_seconds": 0.0,
                    "trace": [],
                }

            if query_result.get("status") == "invalid":
                return {
                    "status": "invalid",
                    "reason_code": query_result.get("reason_code"),
                    "message": query_result.get("message"),
                }
            predictions = query_result.get("predictions")
            predictions_by_query[query_id] = predictions if isinstance(predictions, list) else []
            latency_seconds_by_query[query_id] = float(query_result.get("latency_seconds") or 0.0)
            tool_traces_by_query[query_id] = query_result.get("trace", [])
            if query_result.get("status") != "ok":
                query_errors_by_query[query_id] = {
                    "status": query_result.get("status"),
                    "message": query_result.get("message"),
                }

    return {
        "status": "ok",
        "predictions_by_query": predictions_by_query,
        "latency_seconds_by_query": latency_seconds_by_query,
        "tool_traces_by_query": tool_traces_by_query,
        "query_errors_by_query": query_errors_by_query,
    }


def _score_split(
    *,
    gt_objects: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    predictions_by_query: dict[str, list[dict[str, Any]]],
    latency_seconds_by_query: dict[str, float],
    aliases: dict[str, str],
) -> dict[str, Any]:
    gt_by_id = {str(obj["gt_id"]): obj for obj in gt_objects}
    totals = {k: {"matched_targets": 0, "target_count": 0, "success_count": 0} for k in K_VALUES}
    reciprocal_ranks: list[float] = []
    first_hit_distances: list[float] = []
    proximity_hits = {thr: 0 for thr in PROXIMITY_THRESHOLDS_M}
    proximity_top1_hits = {thr: 0 for thr in PROXIMITY_THRESHOLDS_M}
    proximity_scored = 0
    proximity_top1_scored = 0
    per_query = []

    for query in queries:
        query_id = str(query["query_id"])
        targets = [gt_by_id[gt_id] for gt_id in query["target_gt_ids"] if gt_id in gt_by_id]
        predictions = predictions_by_query.get(query_id, [])
        query_row = {"query_id": query_id, "target_count": len(targets)}
        # Relaxed proximity: nearest distance to any target center, both for the
        # full top-5 (best-of) and for the top-1 prediction alone (the method's
        # primary answer — stricter, more interpretable for ranked output).
        prox = _nearest_proximity(targets, predictions[: max(K_VALUES)])
        prox_top1 = _nearest_proximity(targets, predictions[:1])
        if targets and prox is not None:
            proximity_scored += 1
            for thr in PROXIMITY_THRESHOLDS_M:
                if prox <= thr:
                    proximity_hits[thr] += 1
        if targets and prox_top1 is not None:
            proximity_top1_scored += 1
            for thr in PROXIMITY_THRESHOLDS_M:
                if prox_top1 <= thr:
                    proximity_top1_hits[thr] += 1
        query_row["nearest_proximity_m"] = prox
        query_row["nearest_proximity_top1_m"] = prox_top1
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
        **{
            f"proximity@{thr}m": (safe_div(proximity_hits[thr], proximity_scored) if proximity_scored else None)
            for thr in PROXIMITY_THRESHOLDS_M
        },
        **{
            f"proximity_top1@{thr}m": (safe_div(proximity_top1_hits[thr], proximity_top1_scored) if proximity_top1_scored else None)
            for thr in PROXIMITY_THRESHOLDS_M
        },
        "proximity_scored_count": proximity_scored,
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


def _pred_position(prediction: dict[str, Any]) -> list[float] | None:
    pos = prediction.get("position_3d")
    if isinstance(pos, list) and len(pos) == 3:
        return pos
    bbox = prediction.get("bbox_3d")
    if isinstance(bbox, list) and len(bbox) == 6:
        return [(bbox[0] + bbox[3]) / 2.0, (bbox[1] + bbox[4]) / 2.0, (bbox[2] + bbox[5]) / 2.0]
    return None


def _nearest_proximity(
    targets: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> float | None:
    """Smallest Euclidean distance from any predicted position to any target center.

    Threshold-free, label-agnostic: it answers "how close did the method point to
    the right object?" Used for the relaxed proximity@Xm metrics (fair to
    viewpoint-based caption memory). Returns None if no usable positions exist.
    """
    centers: list[list[float]] = []
    for obj in targets:
        c = obj.get("center_3d")
        if isinstance(c, list) and len(c) == 3:
            centers.append(c)
        else:
            bbox = obj.get("bbox_3d")
            if isinstance(bbox, list) and len(bbox) == 6:
                centers.append([(bbox[0] + bbox[3]) / 2.0, (bbox[1] + bbox[4]) / 2.0, (bbox[2] + bbox[5]) / 2.0])
    if not centers:
        return None
    best: float | None = None
    for pred in predictions:
        pos = _pred_position(pred)
        if pos is None:
            continue
        for c in centers:
            d = euclidean_distance(pos, c)
            if best is None or d < best:
                best = d
    return best


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


# ---------------------------------------------------------------------------
# Build-cost accounting (the construction half of Track 1).
# ---------------------------------------------------------------------------


def _memory_size_summary(package_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    build_log = _load_build_log(package_dir)
    build = manifest.get("build") if isinstance(manifest.get("build"), dict) else {}
    linked_raw_size = linked_raw_size_bytes(manifest)
    return {
        "native_memory_size_bytes": _first_int(
            build_log, build, key="native_memory_size_bytes", fallback=linked_raw_size
        ),
        "package_size_bytes": _first_int(
            build_log, build, key="package_size_bytes", fallback=dir_size_bytes(package_dir)
        ),
        "memory_artifact_size_bytes": _first_int(
            build_log, build, key="memory_artifact_size_bytes", fallback=dir_size_bytes(package_dir / "memory")
        ),
        "linked_raw_size_bytes_if_available": linked_raw_size,
        "frame_count": _first_int(build_log, build, key="frame_count"),
        "time_per_frame_seconds": _first_float(build_log, build, key="time_per_frame_seconds"),
        "peak_ram_bytes": _first_int(build_log, build, key="peak_ram_bytes"),
        "peak_ram_unavailable_reason": _first_value(build_log, build, key="peak_ram_unavailable_reason"),
        "peak_vram_bytes": _first_int(build_log, build, key="peak_vram_bytes"),
        "peak_vram_unavailable_reason": _first_value(build_log, build, key="peak_vram_unavailable_reason"),
        "native_memory_artifacts": _first_value(build_log, build, key="native_memory_artifacts"),
    }


def _build_runtime_seconds(package_dir: Path, manifest: dict[str, Any]) -> float | None:
    build_log = _load_build_log(package_dir)
    build = manifest.get("build") if isinstance(manifest.get("build"), dict) else {}
    value = _first_float(build_log, build, key="build_runtime_seconds")
    if value is not None:
        return value
    return _first_float(build_log, build, key="runtime_seconds")


def _load_build_log(package_dir: Path) -> dict[str, Any]:
    build_log = package_dir / "build_log.json"
    if not build_log.exists():
        return {}
    try:
        with build_log.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_value(*sources: dict[str, Any], key: str) -> Any:
    for source in sources:
        if isinstance(source, dict) and key in source:
            return source[key]
    return None


def _first_int(*sources: dict[str, Any], key: str, fallback: int | None = None) -> int | None:
    value = _first_value(*sources, key=key)
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _first_float(*sources: dict[str, Any], key: str, fallback: float | None = None) -> float | None:
    value = _first_value(*sources, key=key)
    if value is None:
        return fallback
    if isinstance(value, bool):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
