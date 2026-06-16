from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from spatial_memory_evaluation.memory_package_validator import validate_package


def load_package(package_dir: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    package_dir = Path(package_dir)
    report = validate_package(package_dir)
    if not report.valid:
        raise ValueError(json.dumps(report.to_json(), indent=2))
    return read_json(package_dir / "manifest.json"), read_json(package_dir / "capabilities.json")


def fixed_api_capability(capabilities: Mapping[str, Any], track_key: str) -> Mapping[str, Any]:
    fixed_api = capabilities.get("fixed_api")
    if not isinstance(fixed_api, Mapping):
        raise ValueError("capabilities.json missing fixed_api object")
    cap = fixed_api.get(track_key)
    if not isinstance(cap, Mapping):
        raise ValueError(f"capabilities.json missing fixed_api.{track_key}")
    return cap


def invalid_result(
    *,
    method: str,
    package_dir: Path,
    track_key: str,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "status": "invalid",
        "reason_code": "unsupported_fixed_api",
        "required_api": track_key,
        "method": method,
        "package_path": str(package_dir),
        "message": reason or f"Package does not declare support for {track_key}.",
    }


def load_entrypoint(package_dir: Path, entrypoint: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    module_path_text, function_name = entrypoint.split(":", 1)
    module_path = package_dir / module_path_text
    module_name = f"memory_package_{module_path.stem}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load entrypoint module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"entrypoint is not callable: {entrypoint}")
    return func


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def linked_raw_size_bytes(manifest: Mapping[str, Any]) -> int | None:
    total = 0
    saw_existing = False
    for item in manifest.get("raw_links", []) or []:
        if not isinstance(item, Mapping):
            continue
        path_text = item.get("path")
        if not isinstance(path_text, str) or not path_text.startswith("/"):
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        saw_existing = True
        total += dir_size_bytes(path)
    return total if saw_existing else None


def run_agent_command(
    *,
    agent_command: str,
    prompt_path: Path,
    sandbox_dir: Path,
    output_path: Path,
) -> None:
    command = agent_command.format(
        prompt_path=str(prompt_path),
        sandbox_dir=str(sandbox_dir),
        output_path=str(output_path),
    )
    subprocess.run(command, shell=True, cwd=sandbox_dir, check=True)


def copy_package_to_sandbox(package_dir: Path, sandbox_root: Path) -> Path:
    destination = sandbox_root / "package"
    if destination.exists():
        shutil.rmtree(destination)
    ignore = shutil.ignore_patterns("raw_links")
    shutil.copytree(package_dir, destination, ignore=ignore)
    return destination
