from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl, write_jsonl
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.matching import match_objects, mean, median, safe_div
from spatial_memory_evaluation.common.package_io import (
    copy_agent_context_paths,
    copy_package_to_sandbox,
    default_agent_context_paths,
    fixed_api_capability,
    invalid_result,
    load_entrypoint,
    load_package,
    run_agent_command,
)
from spatial_memory_evaluation.common.reporting import (
    evaluation_output_paths,
    render_evaluation_report,
    report_metadata,
    report_title,
    write_evaluation_outputs,
)
from spatial_memory_evaluation.output_paths import timestamped_result_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_KEY = "track2_object_location"
SPLITS = ("detector_coverable",)
K_VALUES = (1, 5)


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
    agent_extra_paths: list[Path] | None = None,
    agent_include_build_code: bool = False,
    agent_include_source_code: bool = True,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track2-{mode}") / "eval_summary.json"

    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, manifest, capabilities, benchmark_dir, method)
    elif mode in ("agentic_memory_only", "agentic_full_access"):
        result = _run_agentic(
            package_dir=package_dir,
            benchmark_dir=benchmark_dir,
            output=output,
            agent_command=agent_command,
            agent_output=agent_output,
            sandbox_root=sandbox_root,
            context_paths=default_agent_context_paths(
                manifest=manifest,
                method=method,
                repo_root=REPO_ROOT,
                explicit_paths=agent_extra_paths or [],
                include_source_code=(
                    (agent_include_source_code and mode == "agentic_full_access")
                    or agent_include_build_code
                ),
            ),
        )
    else:
        raise ValueError(f"unknown Track 2 mode: {mode}")

    base_summary: dict[str, Any] = {
        "status": result["status"],
        "track": "track2_object_location",
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "explicit_memory": manifest.get("explicit_memory"),
    }
    if result["status"] == "ok":
        aliases = load_aliases(benchmark_dir / "label_aliases.json")
        predictions_by_query = result["predictions_by_query"]
        latencies = result.get("latency_seconds_by_query", {})
        full_splits = {
            split: _score_split(
                gt_objects=read_jsonl(track1_benchmark_dir / f"{split}.jsonl"),
                queries=read_jsonl(benchmark_dir / f"queries_{split}.jsonl"),
                predictions_by_query=predictions_by_query,
                latency_seconds_by_query=latencies,
                aliases=aliases,
            )
            for split in SPLITS
        }
        summary = {
            **base_summary,
            "metrics": _track2_summary_metrics(full_splits["detector_coverable"]),
        }
        details = {
            **base_summary,
            "splits": full_splits,
        }
    else:
        summary = {**base_summary, "result": result}
        details = {**base_summary, "result": result}

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


def _track2_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "query_count",
        "success@1",
        "success@5",
        "recall@1",
        "recall@5",
        "mean_first_hit_distance_m",
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


def _run_agentic(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    output: Path,
    agent_command: str | None,
    agent_output: Path | None,
    sandbox_root: Path | None,
    context_paths: list[Path],
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
    copied_context_paths = copy_agent_context_paths(context_paths, sandbox_root / "source_context")
    prompt_path = sandbox_root / "track2_prompt.md"
    agent_output_path = sandbox_root / "track2_agent_output.json"
    prompt_path.write_text(_track2_prompt(sandbox_package, sandbox_queries, copied_context_paths), encoding="utf-8")
    run_agent_command(
        agent_command=agent_command,
        prompt_path=prompt_path,
        sandbox_dir=sandbox_root,
        output_path=agent_output_path,
    )
    return _load_agent_predictions(agent_output_path)


def _load_agent_predictions(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = _load_agent_json_value(text, path)
    if isinstance(value, dict) and "predictions_by_query" not in value and isinstance(value.get("result"), str):
        value = _load_agent_json_value(value["result"], path)
    predictions = value.get("predictions_by_query") if isinstance(value, dict) else None
    if not isinstance(predictions, dict):
        return {"status": "error", "message": f"agent output missing predictions_by_query: {path}"}
    return {"status": "ok", "predictions_by_query": predictions, "latency_seconds_by_query": {}}


def _load_agent_json_value(text: str, path: Path) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for candidate in _json_object_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    preview = stripped[:300].replace("\n", "\\n")
    raise ValueError(f"could not parse agent JSON output from {path}; preview={preview!r}")


def _json_object_candidates(text: str) -> list[str]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


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
                    "target_label": query.get("target_label") or query["canonical_label"],
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


def _track2_prompt(package_dir: Path, query_dir: Path, context_paths: list[Path]) -> str:
    context_text = "\n".join(f"- {path}" for path in context_paths) if context_paths else "- none"
    access_label = "agentic full-access-to-code" if context_paths else "agentic memory-only access"
    return f"""You are evaluating a spatial memory package with {access_label}.

Package directory: {package_dir}
Query files are in: {query_dir}
Source-code context copied into this sandbox:
{context_text}

Your job:
Answer every object-location query using the spatial memory package. You may
design your own temporary Python scripts, small query functions, adapters, or
command-line tools inside the sandbox to inspect and interact with the memory.
You are not limited to package-provided fixed APIs.

Allowed resources:
- The copied memory package: manifest.json, capabilities.json, schema.md,
  memory/, evidence/, package tools, and build_log.json.
- The copied query files. They contain query_id, query text, top_k, and
  target_label/canonical_label fields.
- The copied source-code context, if present. This may include evaluation
  adapter code, shared module code, and original method root source code. Use it
  to understand native memory formats, object fields, coordinate conventions,
  and query utilities.

Forbidden resources/actions:
- Do not use benchmark GT annotations, benchmark answers, target object ids, or
  hidden scorer files.
- Do not follow raw_links or read raw scene frames unless they were explicitly
  copied into the sandbox for a declared ablation.
- Do not use external filesystem paths outside the sandbox.
- Do not use internet or external services.
- Do not modify the copied package as your answer source. Temporary scripts and
  scratch files in the sandbox are fine.
- Do not hard-code answers for specific query ids or scenes.

How to solve:
- Inspect the manifest/schema/build log first to learn what artifacts exist.
- If the package contains `memory/object_table.jsonl`, treat it as the primary
  Track 2 object universe unless schema.md explicitly says otherwise.
- If the package also contains debug/evidence tables such as
  `memory/background_object_table.jsonl`, use them only as supporting evidence
  unless schema.md says they are part of the Track 2 query object universe.
- Read the memory artifacts directly, or use/package/adapt method code to parse
  native maps, scene graphs, databases, feature files, or object tables.
- You may create a temporary query interface that exact-matches or otherwise
  searches `target_label` against memory object labels. Prefer exact target-label / normalized-label
  target_label/canonical_label over brittle natural-language parsing.
- Return up to `top_k` predictions for every query_id.
- Use object ids, labels, positions, bboxes, scores, and evidence grounded in
  the memory package.

Output requirements:
- Return only raw JSON. The first character must be `{{` and the last character
  must be `}}`.
- Do not include Markdown, code fences, headings, or explanations outside JSON.
- The output must contain every query_id from the query files, even if the list
  of predictions is empty.

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
