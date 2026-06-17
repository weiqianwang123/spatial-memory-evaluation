from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ACCOUNTING_VERSION = "0.1"
DEFAULT_PEAK_RAM_UNAVAILABLE_REASON = (
    "not measured by this package builder; use an external profiler for reliable peak RSS"
)
DEFAULT_PEAK_VRAM_UNAVAILABLE_REASON = (
    "not measured by this package builder; GPU memory sampling is not reliable in this wrapper"
)

_MANIFEST_BUILD_KEYS = (
    "command",
    "config_paths",
    "environment",
    "started_at",
    "finished_at",
    "build_runtime_seconds",
    "runtime_seconds",
    "frame_count",
    "time_per_frame_seconds",
    "native_memory_size_bytes",
    "native_memory_artifacts",
    "memory_artifact_size_bytes",
    "package_size_bytes",
    "peak_ram_bytes",
    "peak_ram_unavailable_reason",
    "peak_vram_bytes",
    "peak_vram_unavailable_reason",
    "accounting_version",
)


def path_size_bytes(path: Path | str) -> int:
    """Return file or directory size without following symlinked files."""

    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for item in path.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def build_accounting_snapshot(
    *,
    package_dir: Path | str,
    native_memory_artifact_paths: Iterable[Path | str],
    frame_count: int | None,
    build_runtime_seconds: float | int | None,
    peak_ram_bytes: int | None = None,
    peak_vram_bytes: int | None = None,
    peak_ram_unavailable_reason: str | None = DEFAULT_PEAK_RAM_UNAVAILABLE_REASON,
    peak_vram_unavailable_reason: str | None = DEFAULT_PEAK_VRAM_UNAVAILABLE_REASON,
) -> dict[str, Any]:
    package_dir = Path(package_dir)
    clean_frame_count = _clean_frame_count(frame_count)
    runtime = _clean_runtime(build_runtime_seconds)
    time_per_frame = (
        runtime / clean_frame_count
        if runtime is not None and clean_frame_count > 0
        else None
    )
    native_artifacts = _artifact_size_records(native_memory_artifact_paths)
    native_size = sum(
        int(record["size_bytes"])
        for record in native_artifacts
        if isinstance(record.get("size_bytes"), int)
    )

    accounting: dict[str, Any] = {
        "accounting_version": ACCOUNTING_VERSION,
        "frame_count": clean_frame_count,
        "build_runtime_seconds": runtime,
        "time_per_frame_seconds": time_per_frame,
        "native_memory_size_bytes": native_size,
        "native_memory_artifacts": native_artifacts,
        "memory_artifact_size_bytes": path_size_bytes(package_dir / "memory"),
        "package_size_bytes": path_size_bytes(package_dir),
        "peak_ram_bytes": peak_ram_bytes,
        "peak_vram_bytes": peak_vram_bytes,
    }
    if peak_ram_bytes is None:
        accounting["peak_ram_unavailable_reason"] = peak_ram_unavailable_reason
    if peak_vram_bytes is None:
        accounting["peak_vram_unavailable_reason"] = peak_vram_unavailable_reason
    return accounting


def write_build_log_with_accounting(
    *,
    package_dir: Path | str,
    build_log: Mapping[str, Any],
    native_memory_artifact_paths: Iterable[Path | str],
    frame_count: int | None,
) -> dict[str, Any]:
    """Write build_log.json and mirror stable accounting fields into manifest.json.

    The size of build_log.json itself changes when package_size_bytes is filled
    in, so this writes a small fixed-point pass. The final value is stable for
    normal package sizes and good enough for package-level accounting reports.
    """

    package_dir = Path(package_dir)
    output = dict(build_log)
    runtime = output.get("build_runtime_seconds", output.get("runtime_seconds"))

    accounting: dict[str, Any] = {}
    for _ in range(3):
        accounting = build_accounting_snapshot(
            package_dir=package_dir,
            native_memory_artifact_paths=native_memory_artifact_paths,
            frame_count=frame_count,
            build_runtime_seconds=runtime,
        )
        output.update(accounting)
        if runtime is not None:
            output.setdefault("runtime_seconds", runtime)
        _merge_manifest_build(package_dir, output)
        _write_json(package_dir / "build_log.json", output)

    return accounting


def _artifact_size_records(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        exists = path.exists()
        records.append(
            {
                "path": key,
                "exists": exists,
                "size_bytes": path_size_bytes(path) if exists else None,
            }
        )
    return records


def _merge_manifest_build(package_dir: Path, build_log: Mapping[str, Any]) -> None:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(manifest, dict):
        return
    build = manifest.setdefault("build", {})
    if not isinstance(build, dict):
        build = {}
        manifest["build"] = build
    for key in _MANIFEST_BUILD_KEYS:
        if key in build_log:
            build[key] = build_log[key]
    _write_json(manifest_path, manifest)


def _clean_frame_count(frame_count: int | None) -> int:
    try:
        value = int(frame_count or 0)
    except (TypeError, ValueError):
        value = 0
    return max(value, 0)


def _clean_runtime(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        runtime = float(value)
    except (TypeError, ValueError):
        return None
    return runtime if runtime >= 0 else None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
