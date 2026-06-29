from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from spatial_memory_evaluation.memory_package_validator import (
    CONTROL_FAMILIES,
    validate_package,
)


# Per-track default explanation for why a no-explicit-memory control can never
# expose this fixed API. Used when the control package omits its own reason.
_CONTROL_TRACK_REASON = {
    "track1_object_location": (
        "no explicit object memory and no fixed object-location query API: this is "
        "a no-explicit-memory control with no object inventory (labels + 3D "
        "positions) and no deterministic native object-location query over memory"
    ),
    "track2_scanrefer": (
        "no explicit object memory: this is a no-explicit-memory control with no "
        "referring-expression resolver over object memory"
    ),
    "track3_openeqa": (
        "no explicit object memory: this is a no-explicit-memory control; any "
        "answering happens over raw frames or captions, not exported memory"
    ),
}


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


def is_control_package(
    *,
    explicit_memory: Any = None,
    method_family: Any = None,
) -> bool:
    """A package is a no-explicit-memory control if it declares
    ``explicit_memory=false`` or uses a control method family."""

    return explicit_memory is False or method_family in CONTROL_FAMILIES


def invalid_result(
    *,
    method: str,
    package_dir: Path,
    track_key: str,
    reason: str | None,
    explicit_memory: Any = None,
    method_family: Any = None,
) -> dict[str, Any]:
    """Build the canonical ``invalid`` fixed-API result.

    No-explicit-memory controls (e.g. Multi-frame VLM, LLM-with-captions) get a
    distinct ``reason_code`` and carry the ``explicit_memory``/``method_family``
    markers so a control outcome can never be read as an object-memory baseline
    that merely failed to implement a track.
    """

    control = is_control_package(
        explicit_memory=explicit_memory, method_family=method_family
    )
    reason_text = reason.strip() if isinstance(reason, str) else ""
    if not reason_text:
        if control:
            reason_text = _CONTROL_TRACK_REASON.get(
                track_key,
                "no-explicit-memory control: no fixed object-memory API on this track",
            )
        else:
            reason_text = f"Package does not declare support for {track_key}."
    result: dict[str, Any] = {
        "status": "invalid",
        "reason_code": "control_no_explicit_memory" if control else "unsupported_fixed_api",
        "required_api": track_key,
        "method": method,
        "package_path": str(package_dir),
        "message": reason_text,
        "control": control,
        "explicit_memory": bool(explicit_memory) if explicit_memory is not None else None,
        "method_family": method_family if isinstance(method_family, str) else None,
    }
    return result


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


def run_llm_command(
    *,
    llm_command: str,
    prompt_path: Path,
    output_path: Path,
    session_args: str = "",
    cwd: Path | None = None,
) -> None:
    # Resolve to absolute paths: the command runs with cwd=sandbox_dir, so a
    # repo-relative {prompt_path}/{output_path} would break (e.g. `cat` would not
    # find the prompt). Absolute paths work regardless of cwd.
    prompt_path = prompt_path.resolve()
    output_path = output_path.resolve()
    # cwd defaults to the prompt's dir (per-query). For a persistent agent SESSION
    # the caller pins a STABLE cwd across queries: the Claude CLI scopes sessions
    # to the working directory, so --resume only finds the session from the same
    # cwd where --session-id created it.
    run_cwd = (cwd or prompt_path.parent).resolve()
    command = llm_command
    for key, value in {
        "prompt_path": prompt_path,
        "sandbox_dir": run_cwd,
        "output_path": output_path,
        # {session_args} lets a caller inject --session-id/--resume for a
        # persistent multi-turn agent; empty (default) keeps stateless behavior.
        "session_args": session_args,
    }.items():
        command = command.replace("{" + key + "}", str(value))
    subprocess.run(command, shell=True, cwd=run_cwd, check=True)
