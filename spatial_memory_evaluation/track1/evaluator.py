from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.matching import inventory_metrics
from spatial_memory_evaluation.common.package_io import (
    copy_agent_context_paths,
    copy_package_to_sandbox,
    default_agent_context_paths,
    dir_size_bytes,
    fixed_api_capability,
    invalid_result,
    linked_raw_size_bytes,
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


TRACK_KEY = "track1_memory_construction"
SPLITS = ("detector_coverable",)


def evaluate_track1(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    mode: str,
    output: Path | None,
    agent_command: str | None = None,
    agent_output: Path | None = None,
    sandbox_root: Path | None = None,
    agent_extra_paths: list[Path] | None = None,
    agent_include_source_code: bool = True,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track1-{mode}") / "eval_summary.json"

    aliases = load_aliases(benchmark_dir / "label_aliases.json")
    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, capabilities, method)
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
                repo_root=Path(__file__).resolve().parents[2],
                explicit_paths=agent_extra_paths or [],
                include_source_code=agent_include_source_code and mode == "agentic_full_access",
            ),
        )
    else:
        raise ValueError(f"unknown Track 1 mode: {mode}")

    memory_size = _memory_size_summary(package_dir, manifest)
    build_runtime_seconds = _build_runtime_seconds(package_dir, manifest)
    base_summary: dict[str, Any] = {
        "status": result["status"],
        "track": "track1_memory_construction",
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "package_size_bytes": memory_size["package_size_bytes"],
        "memory_artifact_size_bytes": memory_size["memory_artifact_size_bytes"],
        "build_runtime_seconds": build_runtime_seconds,
    }
    if result["status"] == "ok":
        predictions = result["objects"]
        full_splits = {
            split: inventory_metrics(read_jsonl(benchmark_dir / f"{split}.jsonl"), predictions, aliases)
            for split in SPLITS
        }
        summary = {
            **base_summary,
            "metrics": _track1_summary_metrics(full_splits["detector_coverable"]),
        }
        details = {
            **base_summary,
            "memory_size": memory_size,
            "fixed_api_runtime_seconds": result.get("api_runtime_seconds"),
            "splits": full_splits,
        }
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
        "gt_count",
        "prediction_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_center_error_m",
    )
    return {key: metrics.get(key) for key in keys}


def _run_fixed_api(package_dir: Path, capabilities: dict[str, Any], method: str) -> dict[str, Any]:
    cap = fixed_api_capability(capabilities, TRACK_KEY)
    if cap.get("status") != "supported":
        return invalid_result(
            method=method,
            package_dir=package_dir,
            track_key=TRACK_KEY,
            reason=str(cap.get("reason") or ""),
        )
    func = load_entrypoint(package_dir, str(cap["entrypoint"]))
    started = time.perf_counter()
    result = func(str(package_dir), {})
    runtime = time.perf_counter() - started
    objects = result.get("objects") if isinstance(result, dict) else None
    if not isinstance(objects, list):
        return {"status": "error", "message": "Track 1 entrypoint did not return an objects list"}
    return {"status": "ok", "objects": objects, "api_runtime_seconds": runtime}


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
        return _load_agent_objects(agent_output)
    if not agent_command:
        return {
            "status": "error",
            "message": "agentic_memory_only requires --agent-output or --agent-command",
        }
    sandbox_root = sandbox_root or (output.parent / "agent_sandbox")
    sandbox_root.mkdir(parents=True, exist_ok=True)
    sandbox_package = copy_package_to_sandbox(package_dir, sandbox_root)
    copied_context_paths = copy_agent_context_paths(context_paths, sandbox_root / "source_context")
    prompt_path = sandbox_root / "track1_prompt.md"
    agent_output_path = sandbox_root / "track1_agent_output.json"
    prompt_path.write_text(_track1_prompt(sandbox_package, copied_context_paths), encoding="utf-8")
    run_agent_command(
        agent_command=agent_command,
        prompt_path=prompt_path,
        sandbox_dir=sandbox_root,
        output_path=agent_output_path,
    )
    return _load_agent_objects(agent_output_path)


def _load_agent_objects(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    objects = value.get("objects") if isinstance(value, dict) else None
    if not isinstance(objects, list):
        return {"status": "error", "message": f"agent output missing objects list: {path}"}
    return {"status": "ok", "objects": objects}


def _track1_prompt(package_dir: Path, context_paths: list[Path]) -> str:
    context_text = "\n".join(f"- {path}" for path in context_paths) if context_paths else "- none"
    access_label = "agentic full-access-to-code" if context_paths else "agentic memory-only access"
    return f"""You are evaluating a spatial memory package with {access_label}.

Package directory: {package_dir}
Source-code context copied into this sandbox:
{context_text}

Your job:
Infer the object inventory represented by this memory package. You may design
your own temporary Python scripts, small query functions, or command-line tools
inside the sandbox to inspect and interact with the memory. You are not limited
to package-provided fixed APIs.

Allowed resources:
- The copied memory package: manifest.json, capabilities.json, schema.md,
  memory/, evidence/, package tools, and build_log.json.
- The copied source-code context, if present. This may include evaluation
  adapter code, shared module code, and original method root source code. Use it
  to understand native memory formats, object fields, coordinate conventions,
  and query utilities.

Forbidden resources/actions:
- Do not use benchmark GT annotations, benchmark answers, or test labels.
- Do not follow raw_links or read raw scene frames unless they were explicitly
  copied into the sandbox for a declared ablation.
- Do not use external filesystem paths outside the sandbox.
- Do not use internet or external services.
- Do not modify the copied package as your answer source. Temporary scripts and
  scratch files in the sandbox are fine.
- Do not hard-code answers for specific query ids or scenes.

Output requirements:
- Return only raw JSON. The first character must be `{{` and the last character
  must be `}}`.
- Do not include Markdown, code fences, headings, or explanations outside JSON.
- `objects` should contain every object you believe the memory represents for
  the detector-coverable closed-vocabulary object inventory.
- Prefer object ids, labels, positions, bboxes, and evidence directly from the
  memory package. Use the original method code only to parse or understand the
  memory, not to reconstruct memory from raw inputs.

Return JSON with this exact shape:

{{
  "objects": [
    {{
      "object_id": "string",
      "label": "chair",
      "position_3d": [0.0, 0.0, 0.0],
      "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      "evidence": []
    }}
  ]
}}
"""


def _memory_size_summary(package_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "package_size_bytes": dir_size_bytes(package_dir),
        "memory_artifact_size_bytes": dir_size_bytes(package_dir / "memory"),
        "native_linked_size_bytes_if_available": linked_raw_size_bytes(manifest),
    }


def _build_runtime_seconds(package_dir: Path, manifest: dict[str, Any]) -> float | None:
    value = None
    if isinstance(manifest.get("build"), dict):
        value = manifest["build"].get("runtime_seconds")
    build_log = package_dir / "build_log.json"
    if build_log.exists():
        with build_log.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            value = loaded.get("runtime_seconds", value)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
