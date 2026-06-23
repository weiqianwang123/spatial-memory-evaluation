"""Track 2: ScanRefer instance-level referring-expression query evaluation.

Scores fine-grained referring expressions ("the red chair next to the table")
against the exported memory. Supports ``fixed_api`` (package declares a native
``resolve_referring_expression`` entrypoint) and ``tool_llm`` (per-query LLM +
method-native referring/retrieval tools).

Scoring: target object-name accuracy (``referring_acc@1/@5``) plus distance-based
localization accuracy (``acc@0.25m`` / ``acc@0.5m`` — top-1 predicted 3D position
within X meters of the GT object center) and ``mean_center_distance_m``. Distance,
not IoU, is used because caption-memory methods emit a viewpoint position rather
than an instance bbox. The harness emits a ``data_unavailable`` result until the
referring benchmark is built (see ``track2/data.py``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.matching import euclidean_distance, mean, safe_div
from spatial_memory_evaluation.common.package_io import (
    fixed_api_capability,
    invalid_result,
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

from .data import REFERRING_QUERIES_FILE, SCENE_OBJECTS_FILE, track2_data_status


TRACK_KEY = "track2_scanrefer"
K_VALUES = (1, 5)
# Localization accuracy thresholds in METERS: a top-1 prediction counts as a hit
# if its 3D position is within the threshold of the GT object center. ReMEmbR
# (and other caption-memory methods) emit a viewpoint position, not an instance
# bbox, so distance — not IoU — is the meaningful localization metric here.
DISTANCE_THRESHOLDS_M = (0.25, 0.5)


def evaluate_track2(
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
        output = timestamped_result_dir(method, f"track2-{mode}") / "eval_summary.json"

    base_summary: dict[str, Any] = {
        "status": "ok",
        "track": TRACK_KEY,
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "explicit_memory": manifest.get("explicit_memory"),
    }

    data_status = track2_data_status(benchmark_dir)
    if data_status.get("status") != "ok":
        result = {"status": "data_unavailable", **data_status}
        return _finalize(base_summary, result, output)

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
        raise ValueError(f"unknown Track 2 mode: {mode}")

    if result["status"] == "ok":
        scene_objects = _scene_objects_by_id(benchmark_dir)
        queries = read_jsonl(benchmark_dir / REFERRING_QUERIES_FILE)
        metrics = _score(
            queries=queries,
            scene_objects=scene_objects,
            predictions_by_query=result["predictions_by_query"],
            latency_seconds_by_query=result.get("latency_seconds_by_query", {}),
        )
        summary = {**base_summary, "metrics": _summary_metrics(metrics)}
        details = {**base_summary, "metrics": metrics}
        if result.get("tool_traces_by_query") is not None:
            details["tool_traces_by_query"] = result["tool_traces_by_query"]
    else:
        summary = {**base_summary, "status": result["status"], "result": result}
        details = dict(summary)

    return _finalize(base_summary, result, output, summary=summary, details=details)


def _finalize(
    base_summary: dict[str, Any],
    result: dict[str, Any],
    output: Path,
    *,
    summary: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if summary is None:
        summary = {**base_summary, "status": result["status"], "result": result}
    if details is None:
        details = dict(summary)
    paths = evaluation_output_paths(output)
    report = render_evaluation_report(
        title=report_title(summary),
        metadata=report_metadata(summary),
        metrics=summary.get("metrics") if isinstance(summary.get("metrics"), dict) else None,
        status=str(summary["status"]),
        summary_path=paths.summary,
        details_path=paths.details,
        result=result if result.get("status") != "ok" else None,
    )
    write_evaluation_outputs(summary_path=output, summary=summary, details=details, report_markdown=report)
    return summary


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "query_count",
        "referring_acc@1",
        "referring_acc@5",
        "acc@0.25m",
        "acc@0.5m",
        "mean_center_distance_m",
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
    resolve = load_entrypoint(package_dir, str(cap["entrypoint"]))
    predictions_by_query: dict[str, list[dict[str, Any]]] = {}
    latency_seconds_by_query: dict[str, float] = {}
    for query in read_jsonl(benchmark_dir / REFERRING_QUERIES_FILE):
        query_id = str(query["query_id"])
        started = time.perf_counter()
        result = resolve(
            str(package_dir),
            {
                "query_id": query_id,
                "dataset": query.get("dataset", "scanrefer"),
                "scene_id": query.get("scene_id"),
                "utterance": query.get("utterance"),
                "top_k": int(query.get("top_k", 10)),
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

    for query in read_jsonl(benchmark_dir / REFERRING_QUERIES_FILE):
        query_id = str(query["query_id"])
        try:
            query_result = run_tool_llm_query(
                package_dir=package_dir,
                manifest=manifest,
                query={
                    "query_id": query_id,
                    "scene_id": query.get("scene_id"),
                    "query": query.get("utterance"),
                    "utterance": query.get("utterance"),
                    "top_k": int(query.get("top_k", 10)),
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

    return {
        "status": "ok",
        "predictions_by_query": predictions_by_query,
        "latency_seconds_by_query": latency_seconds_by_query,
        "tool_traces_by_query": tool_traces_by_query,
    }


def _scene_objects_by_id(benchmark_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Map scene_id -> {object_id -> object record} for distance/center scoring."""

    scene_objects_path = benchmark_dir / SCENE_OBJECTS_FILE
    by_scene: dict[str, dict[str, dict[str, Any]]] = {}
    if not scene_objects_path.exists():
        return by_scene
    for obj in read_jsonl(scene_objects_path):
        scene_id = str(obj.get("scene_id"))
        object_id = str(obj.get("object_id"))
        by_scene.setdefault(scene_id, {})[object_id] = obj
    return by_scene


def _score(
    *,
    queries: list[dict[str, Any]],
    scene_objects: dict[str, dict[str, dict[str, Any]]],
    predictions_by_query: dict[str, list[dict[str, Any]]],
    latency_seconds_by_query: dict[str, float],
) -> dict[str, Any]:
    distance_hits = {threshold: 0 for threshold in DISTANCE_THRESHOLDS_M}
    center_distances: list[float] = []
    name_hits_top1 = 0
    name_hits_topk = 0
    name_scored = 0
    distance_scored = 0
    per_query = []

    for query in queries:
        query_id = str(query["query_id"])
        target_bbox = query.get("target_bbox_3d")
        target_name = query.get("target_object_name")
        predictions = predictions_by_query.get(query_id, [])
        top1 = predictions[0] if predictions else None

        # Name-level referring accuracy (ScanEnts3D has no 3D bbox): does the
        # resolved object's label match the GT target object_name?
        name_top1 = None
        name_topk = None
        if isinstance(target_name, str) and target_name.strip():
            name_scored += 1
            name_top1 = bool(top1 is not None and _name_match(top1.get("label"), target_name))
            name_topk = any(_name_match(p.get("label"), target_name) for p in predictions)
            if name_top1:
                name_hits_top1 += 1
            if name_topk:
                name_hits_topk += 1

        # Distance-based localization accuracy: the top-1 prediction's 3D
        # position vs the GT object-bbox center. A query is "distance-scored"
        # only when both a GT bbox and a usable predicted position exist.
        center_distance = None
        if isinstance(target_bbox, list):
            center_distance = _center_distance(top1, target_bbox) if top1 is not None else None
            if center_distance is not None:
                distance_scored += 1
                center_distances.append(center_distance)
                for threshold in DISTANCE_THRESHOLDS_M:
                    if center_distance <= threshold:
                        distance_hits[threshold] += 1
        per_query.append(
            {
                "query_id": query_id,
                "target_object_name": target_name,
                "name_match_top1": name_top1,
                "name_match_topk": name_topk,
                "center_distance_m": center_distance,
                "latency_ms": latency_seconds_by_query.get(query_id, 0.0) * 1000.0,
            }
        )

    latencies = [latency_seconds_by_query.get(str(q["query_id"]), 0.0) for q in queries]
    return {
        "query_count": len(queries),
        "referring_acc@1": safe_div(name_hits_top1, name_scored) if name_scored else None,
        "referring_acc@5": safe_div(name_hits_topk, name_scored) if name_scored else None,
        "name_scored_count": name_scored,
        **{
            f"acc@{threshold}m": (safe_div(distance_hits[threshold], distance_scored) if distance_scored else None)
            for threshold in DISTANCE_THRESHOLDS_M
        },
        "distance_scored_count": distance_scored,
        "mean_center_distance_m": mean(center_distances),
        "mean_query_latency_ms": (mean(latencies) or 0.0) * 1000.0,
        "per_query": per_query,
    }


def _name_match(predicted_label: Any, target_name: str) -> bool:
    pred = str(predicted_label or "").strip().lower().replace("_", " ")
    target = str(target_name or "").strip().lower().replace("_", " ")
    if not pred or not target:
        return False
    # match if either name contains the other (handles "office chair" vs "chair")
    return target in pred or pred in target


def _center_distance(prediction: dict[str, Any], target_bbox: list[float]) -> float | None:
    target_center = _bbox_center(target_bbox)
    pred_center = prediction.get("position_3d")
    if not isinstance(pred_center, list):
        pred_bbox = prediction.get("bbox_3d")
        pred_center = _bbox_center(pred_bbox) if isinstance(pred_bbox, list) else None
    if pred_center is None or target_center is None:
        return None
    return euclidean_distance(pred_center, target_center)


def _bbox_center(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 6:
        return None
    try:
        return [
            (float(bbox[0]) + float(bbox[3])) / 2.0,
            (float(bbox[1]) + float(bbox[4])) / 2.0,
            (float(bbox[2]) + float(bbox[5])) / 2.0,
        ]
    except (TypeError, ValueError):
        return None
