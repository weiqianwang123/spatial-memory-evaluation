from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl, write_json
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.matching import inventory_metrics
from spatial_memory_evaluation.common.package_io import (
    copy_package_to_sandbox,
    dir_size_bytes,
    fixed_api_capability,
    invalid_result,
    linked_raw_size_bytes,
    load_entrypoint,
    load_package,
    run_agent_command,
)
from spatial_memory_evaluation.output_paths import timestamped_result_dir


TRACK_KEY = "track1_memory_construction"
SPLITS = ("all_annotated", "detector_coverable")


def evaluate_track1(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    mode: str,
    output: Path | None,
    agent_command: str | None = None,
    agent_output: Path | None = None,
    sandbox_root: Path | None = None,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track1-{mode}") / "eval_summary.json"

    aliases = load_aliases(benchmark_dir / "label_aliases.json")
    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, capabilities, method)
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
        raise ValueError(f"unknown Track 1 mode: {mode}")

    summary: dict[str, Any] = {
        "status": result["status"],
        "track": "track1_memory_construction",
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "memory_size": _memory_size_summary(package_dir, manifest),
        "build_runtime_seconds": _build_runtime_seconds(package_dir, manifest),
    }
    if result["status"] == "ok":
        predictions = result["objects"]
        summary["fixed_api_runtime_seconds"] = result.get("api_runtime_seconds")
        summary["splits"] = {
            split: inventory_metrics(read_jsonl(benchmark_dir / f"{split}.jsonl"), predictions, aliases)
            for split in SPLITS
        }
    else:
        summary["result"] = result

    write_json(output, summary)
    return summary


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
    prompt_path = sandbox_root / "track1_prompt.md"
    agent_output_path = sandbox_root / "track1_agent_output.json"
    prompt_path.write_text(_track1_prompt(sandbox_package), encoding="utf-8")
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


def _track1_prompt(package_dir: Path) -> str:
    return f"""You are evaluating a spatial memory package in memory-only mode.

Package directory: {package_dir}

Read only the package files. Do not use raw_links or external source data.
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
