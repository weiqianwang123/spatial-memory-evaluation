from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "0.1"

REQUIRED_FILES = (
    "manifest.json",
    "capabilities.json",
    "schema.md",
    "build_log.json",
)
REQUIRED_DIRS = (
    "memory",
    "evidence",
    "raw_links",
    "schemas",
    "tools",
)

METHOD_FAMILIES = {
    "object_map",
    "scene_graph",
    "caption_memory",
    "vector_db",
    "raw_frame_control",
    "caption_control",
    "other",
}
CONTROL_FAMILIES = {"raw_frame_control", "caption_control"}

TRACK_KEYS = (
    "track1_memory_construction",
    "track2_object_location",
    "track3_scanrefer",
    "track4_openeqa",
)

FIXED_API_STATUSES = {"supported", "invalid"}
AGENT_ACCESS_MODES = {"memory_only", "memory_plus_crops", "memory_plus_raw"}

REQUIRED_SCHEMA_MD_TOPICS = (
    ("coordinate_frame", ("coordinate", "坐标")),
    ("units", ("unit", "单位")),
    ("object_id", ("object id", "object_id", "对象id", "物体id", "对象 id", "物体 id")),
    ("timestamp", ("timestamp", "time", "时间")),
    ("relations", ("relation", "关系")),
    ("confidence_or_score", ("confidence", "score", "置信", "分数")),
    ("artifact_formats", ("artifact", "format", "工件", "格式")),
    ("limitations", ("limitation", "unsupported", "invalid", "限制", "不支持")),
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    path: str
    message: str

    def to_json(self) -> dict[str, str]:
        return {"severity": self.severity, "path": self.path, "message": self.message}


@dataclass
class ValidationReport:
    package_dir: Path
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def error(self, path: str, message: str) -> None:
        self.errors.append(ValidationIssue("error", path, message))

    def warning(self, path: str, message: str) -> None:
        self.warnings.append(ValidationIssue("warning", path, message))

    def to_json(self) -> dict[str, Any]:
        return {
            "package_dir": str(self.package_dir),
            "valid": self.valid,
            "errors": [issue.to_json() for issue in self.errors],
            "warnings": [issue.to_json() for issue in self.warnings],
        }


def validate_package(package_dir: Path | str) -> ValidationReport:
    package_dir = Path(package_dir)
    report = ValidationReport(package_dir=package_dir)

    if not package_dir.exists():
        report.error(".", "package directory does not exist")
        return report
    if not package_dir.is_dir():
        report.error(".", "package path is not a directory")
        return report

    _validate_required_layout(package_dir, report)
    manifest = _load_json(package_dir / "manifest.json", report)
    capabilities = _load_json(package_dir / "capabilities.json", report)
    build_log = _load_json(package_dir / "build_log.json", report)
    schema_text = _load_text(package_dir / "schema.md", report)

    if isinstance(manifest, dict):
        _validate_manifest(package_dir, manifest, report)
    if isinstance(capabilities, dict):
        explicit_memory = manifest.get("explicit_memory") if isinstance(manifest, dict) else None
        method_family = None
        if isinstance(manifest, dict) and isinstance(manifest.get("method"), dict):
            method_family = manifest["method"].get("family")
        _validate_capabilities(
            package_dir,
            capabilities,
            explicit_memory=explicit_memory,
            method_family=method_family,
            report=report,
        )
    if isinstance(build_log, dict):
        _validate_build_log(build_log, report)
    if schema_text is not None:
        _validate_schema_md(schema_text, report)

    return report


def load_schema(name: str) -> dict[str, Any]:
    """Load one of the packaged memory-package schema JSON files."""

    schema_path = (
        resources.files("spatial_memory_evaluation")
        / "schemas"
        / "memory_package"
        / name
    )
    with schema_path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Schema {name} must be a JSON object")
    return value


def _validate_required_layout(package_dir: Path, report: ValidationReport) -> None:
    for name in REQUIRED_FILES:
        path = package_dir / name
        if not path.is_file():
            report.error(name, "required file is missing")
    for name in REQUIRED_DIRS:
        path = package_dir / name
        if not path.is_dir():
            report.error(name, "required directory is missing")


def _load_json(path: Path, report: ValidationReport) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        report.error(path.name, f"invalid JSON: {exc}")
    return None


def _load_text(path: Path, report: ValidationReport) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        report.error(path.name, f"cannot decode UTF-8 text: {exc}")
    return None


def _validate_manifest(
    package_dir: Path,
    manifest: Mapping[str, Any],
    report: ValidationReport,
) -> None:
    _require_string(manifest, "schema_version", "manifest.json.schema_version", report)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        report.error(
            "manifest.json.schema_version",
            f"expected {SCHEMA_VERSION!r}",
        )

    _require_string(manifest, "package_id", "manifest.json.package_id", report)
    _require_object(manifest, "method", "manifest.json.method", report)
    _require_object(manifest, "dataset", "manifest.json.dataset", report)
    _require_object(manifest, "input", "manifest.json.input", report)
    _require_bool(manifest, "explicit_memory", "manifest.json.explicit_memory", report)
    _require_array(manifest, "memory_artifacts", "manifest.json.memory_artifacts", report)
    _require_array(manifest, "evidence_artifacts", "manifest.json.evidence_artifacts", report)
    _require_array(manifest, "raw_links", "manifest.json.raw_links", report)
    _require_array(manifest, "tools", "manifest.json.tools", report)
    _require_object(manifest, "build", "manifest.json.build", report)
    _require_object(manifest, "allowed_access", "manifest.json.allowed_access", report)

    method = manifest.get("method")
    if isinstance(method, Mapping):
        _require_string(method, "name", "manifest.json.method.name", report)
        _require_string(method, "display_name", "manifest.json.method.display_name", report)
        family = method.get("family")
        if not isinstance(family, str):
            report.error("manifest.json.method.family", "must be a string")
        elif family not in METHOD_FAMILIES:
            report.error(
                "manifest.json.method.family",
                f"must be one of {sorted(METHOD_FAMILIES)}",
            )

        explicit_memory = manifest.get("explicit_memory")
        if family in CONTROL_FAMILIES and explicit_memory is not False:
            report.error(
                "manifest.json.explicit_memory",
                f"{family} packages must set explicit_memory=false",
            )
        elif family not in CONTROL_FAMILIES and explicit_memory is False:
            report.warning(
                "manifest.json.explicit_memory",
                "non-control package declares explicit_memory=false",
            )

    dataset = manifest.get("dataset")
    if isinstance(dataset, Mapping):
        _require_string(dataset, "name", "manifest.json.dataset.name", report)
        if "scene_id" not in dataset and "episode_id" not in dataset:
            report.error(
                "manifest.json.dataset",
                "must include scene_id or episode_id, even if one is null",
            )

    input_data = manifest.get("input")
    if isinstance(input_data, Mapping):
        modality = input_data.get("modality")
        if not isinstance(modality, list) or not all(isinstance(item, str) for item in modality):
            report.error("manifest.json.input.modality", "must be an array of strings")
        if "frame_count" in input_data and not isinstance(input_data["frame_count"], int):
            report.error("manifest.json.input.frame_count", "must be an integer")
        _require_string(
            input_data,
            "coordinate_frame",
            "manifest.json.input.coordinate_frame",
            report,
        )

    allowed_access = manifest.get("allowed_access")
    if isinstance(allowed_access, Mapping):
        for key in (
            "contains_gt_annotations",
            "contains_benchmark_answers",
            "contains_test_labels",
            "contains_question_specific_rules",
        ):
            _require_bool(allowed_access, key, f"manifest.json.allowed_access.{key}", report)
            if allowed_access.get(key) is True:
                report.warning(
                    f"manifest.json.allowed_access.{key}",
                    "leakage-related access is declared true; evaluators should reject it unless explicitly allowed",
                )

    for list_key in ("memory_artifacts", "evidence_artifacts", "tools"):
        artifacts = manifest.get(list_key)
        if isinstance(artifacts, list):
            _validate_artifact_records(
                package_dir,
                artifacts,
                f"manifest.json.{list_key}",
                allow_absolute=False,
                report=report,
            )

    raw_links = manifest.get("raw_links")
    if isinstance(raw_links, list):
        _validate_artifact_records(
            package_dir,
            raw_links,
            "manifest.json.raw_links",
            allow_absolute=True,
            report=report,
        )


def _validate_artifact_records(
    package_dir: Path,
    artifacts: Sequence[Any],
    base_path: str,
    *,
    allow_absolute: bool,
    report: ValidationReport,
) -> None:
    for index, record in enumerate(artifacts):
        path_prefix = f"{base_path}[{index}]"
        if not isinstance(record, Mapping):
            report.error(path_prefix, "artifact record must be an object")
            continue

        _require_string(record, "name", f"{path_prefix}.name", report)
        _require_string(record, "type", f"{path_prefix}.type", report)
        _require_string(record, "path", f"{path_prefix}.path", report)
        if "description" in record and not isinstance(record["description"], str):
            report.error(f"{path_prefix}.description", "must be a string")
        required_for = record.get("required_for", [])
        if not isinstance(required_for, list) or not all(
            isinstance(item, str) for item in required_for
        ):
            report.error(f"{path_prefix}.required_for", "must be an array of strings")
        else:
            for item in required_for:
                if item not in TRACK_KEYS:
                    report.warning(
                        f"{path_prefix}.required_for",
                        f"unknown track key {item!r}",
                    )

        artifact_path = record.get("path")
        if isinstance(artifact_path, str):
            _validate_package_path(
                package_dir,
                artifact_path,
                f"{path_prefix}.path",
                allow_absolute=allow_absolute,
                must_exist=bool(required_for),
                report=report,
            )


def _validate_capabilities(
    package_dir: Path,
    capabilities: Mapping[str, Any],
    *,
    explicit_memory: Any,
    method_family: Any,
    report: ValidationReport,
) -> None:
    _require_string(capabilities, "schema_version", "capabilities.json.schema_version", report)
    if capabilities.get("schema_version") != SCHEMA_VERSION:
        report.error(
            "capabilities.json.schema_version",
            f"expected {SCHEMA_VERSION!r}",
        )

    fixed_api = capabilities.get("fixed_api")
    if not isinstance(fixed_api, Mapping):
        report.error("capabilities.json.fixed_api", "must be an object")
    else:
        supported_tracks = []
        for track_key in TRACK_KEYS:
            track_path = f"capabilities.json.fixed_api.{track_key}"
            track = fixed_api.get(track_key)
            if not isinstance(track, Mapping):
                report.error(track_path, "required track capability object is missing")
                continue
            status = track.get("status")
            if status not in FIXED_API_STATUSES:
                report.error(
                    f"{track_path}.status",
                    f"must be one of {sorted(FIXED_API_STATUSES)}",
                )
                continue

            if status == "supported":
                supported_tracks.append(track_key)
                entrypoint = track.get("entrypoint")
                if not isinstance(entrypoint, str) or not entrypoint.strip():
                    report.error(f"{track_path}.entrypoint", "supported track needs entrypoint")
                else:
                    _validate_python_entrypoint(
                        package_dir,
                        entrypoint,
                        f"{track_path}.entrypoint",
                        report,
                    )
                for schema_key in ("input_schema", "output_schema"):
                    schema_path = track.get(schema_key)
                    if schema_path is not None:
                        if not isinstance(schema_path, str):
                            report.error(f"{track_path}.{schema_key}", "must be a string")
                        else:
                            _validate_package_path(
                                package_dir,
                                schema_path,
                                f"{track_path}.{schema_key}",
                                allow_absolute=False,
                                must_exist=True,
                                report=report,
                            )
            else:
                if track.get("entrypoint") not in (None, ""):
                    report.error(f"{track_path}.entrypoint", "invalid track must not set entrypoint")
                reason = track.get("reason")
                if not isinstance(reason, str) or not reason.strip():
                    report.error(f"{track_path}.reason", "invalid track needs non-empty reason")

        if explicit_memory is False and supported_tracks:
            report.error(
                "capabilities.json.fixed_api",
                "explicit_memory=false packages cannot declare supported fixed memory APIs",
            )
        if method_family in CONTROL_FAMILIES and supported_tracks:
            report.error(
                "capabilities.json.fixed_api",
                f"{method_family} controls must declare fixed API tracks invalid",
            )

    agent_access = capabilities.get("agent_access")
    if not isinstance(agent_access, Mapping):
        report.error("capabilities.json.agent_access", "must be an object")
    else:
        _validate_agent_access(agent_access, report)


def _validate_agent_access(agent_access: Mapping[str, Any], report: ValidationReport) -> None:
    mode = agent_access.get("mode")
    if mode not in AGENT_ACCESS_MODES:
        report.error(
            "capabilities.json.agent_access.mode",
            f"must be one of {sorted(AGENT_ACCESS_MODES)}",
        )

    bool_keys = (
        "read_manifest",
        "read_schema",
        "read_memory_artifacts",
        "read_evidence",
        "read_raw_links",
        "read_raw_frames",
        "read_source_keyframes_or_crops",
        "run_package_tools",
        "write_package",
    )
    for key in bool_keys:
        _require_bool(agent_access, key, f"capabilities.json.agent_access.{key}", report)

    if agent_access.get("write_package") is not False:
        report.error("capabilities.json.agent_access.write_package", "must be false")

    if mode == "memory_only":
        for key in (
            "read_raw_links",
            "read_raw_frames",
            "read_source_keyframes_or_crops",
        ):
            if agent_access.get(key) is not False:
                report.error(
                    f"capabilities.json.agent_access.{key}",
                    "memory_only mode must disable raw/source-frame access",
                )

    if mode == "memory_plus_crops" and agent_access.get("read_raw_frames") is True:
        report.error(
            "capabilities.json.agent_access.read_raw_frames",
            "memory_plus_crops must not enable raw-frame access",
        )


def _validate_build_log(build_log: Mapping[str, Any], report: ValidationReport) -> None:
    status = build_log.get("status")
    if status not in {"ok", "error"}:
        report.error("build_log.json.status", "must be 'ok' or 'error'")
    if status == "error":
        report.error("build_log.json.status", "package build status is error")
    if "runtime_seconds" in build_log and build_log["runtime_seconds"] is not None:
        if not isinstance(build_log["runtime_seconds"], (int, float)):
            report.error("build_log.json.runtime_seconds", "must be a number or null")


def _validate_schema_md(schema_text: str, report: ValidationReport) -> None:
    normalized = _normalize_text(schema_text)
    if len(normalized.strip()) < 80:
        report.error("schema.md", "schema documentation is too short to be self-describing")

    for topic, needles in REQUIRED_SCHEMA_MD_TOPICS:
        if not any(_normalize_text(needle) in normalized for needle in needles):
            report.error("schema.md", f"missing required topic: {topic}")


def _validate_python_entrypoint(
    package_dir: Path,
    entrypoint: str,
    path: str,
    report: ValidationReport,
) -> None:
    if ":" not in entrypoint:
        report.error(path, "entrypoint must be relative/path.py:function_name")
        return

    module_path_text, function_name = entrypoint.split(":", 1)
    if not module_path_text or not function_name:
        report.error(path, "entrypoint must include both path and function")
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name):
        report.error(path, "function name must be a valid Python identifier")
        return

    module_path = _validate_package_path(
        package_dir,
        module_path_text,
        path,
        allow_absolute=False,
        must_exist=True,
        report=report,
    )
    if module_path is None:
        return
    if module_path.suffix != ".py":
        report.error(path, "entrypoint path must end with .py")
        return

    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    except SyntaxError as exc:
        report.error(path, f"entrypoint file has syntax error: {exc}")
        return
    except UnicodeDecodeError as exc:
        report.error(path, f"entrypoint file is not UTF-8: {exc}")
        return

    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_name not in function_names:
        report.error(path, f"entrypoint function {function_name!r} not found")


def _validate_package_path(
    package_dir: Path,
    value: str,
    path: str,
    *,
    allow_absolute: bool,
    must_exist: bool,
    report: ValidationReport,
) -> Optional[Path]:
    candidate = Path(value)
    if candidate.is_absolute():
        if allow_absolute:
            if must_exist and not candidate.exists():
                report.warning(path, "absolute external path does not exist on this machine")
            return candidate
        report.error(path, "must be package-relative, not absolute")
        return None

    resolved_package = package_dir.resolve()
    resolved = (package_dir / candidate).resolve()
    try:
        resolved.relative_to(resolved_package)
    except ValueError:
        report.error(path, "must not escape the package directory")
        return None

    if must_exist and not resolved.exists():
        report.error(path, "referenced package path does not exist")
        return None
    return resolved


def _require_object(
    data: Mapping[str, Any],
    key: str,
    path: str,
    report: ValidationReport,
) -> bool:
    if key not in data:
        report.error(path, "is required")
        return False
    if not isinstance(data[key], Mapping):
        report.error(path, "must be an object")
        return False
    return True


def _require_array(
    data: Mapping[str, Any],
    key: str,
    path: str,
    report: ValidationReport,
) -> bool:
    if key not in data:
        report.error(path, "is required")
        return False
    if not isinstance(data[key], list):
        report.error(path, "must be an array")
        return False
    return True


def _require_string(
    data: Mapping[str, Any],
    key: str,
    path: str,
    report: ValidationReport,
) -> bool:
    if key not in data:
        report.error(path, "is required")
        return False
    if not isinstance(data[key], str):
        report.error(path, "must be a string")
        return False
    if isinstance(data[key], str) and not data[key].strip():
        report.error(path, "must not be empty")
        return False
    return True


def _require_bool(
    data: Mapping[str, Any],
    key: str,
    path: str,
    report: ValidationReport,
) -> bool:
    if key not in data:
        report.error(path, "is required")
        return False
    if not isinstance(data[key], bool):
        report.error(path, "must be a boolean")
        return False
    return True


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold())


def _format_human_report(report: ValidationReport) -> str:
    lines = [
        f"package: {report.package_dir}",
        f"valid: {str(report.valid).lower()}",
        f"errors: {len(report.errors)}",
        f"warnings: {len(report.warnings)}",
    ]
    for issue in _iter_issues(report.errors, report.warnings):
        lines.append(f"{issue.severity}: {issue.path}: {issue.message}")
    return "\n".join(lines)


def _iter_issues(
    errors: Sequence[ValidationIssue],
    warnings: Sequence[ValidationIssue],
) -> Iterable[ValidationIssue]:
    yield from errors
    yield from warnings


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a minimal memory package.")
    parser.add_argument("package_dir", type=Path, help="memory package directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable validation report",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = validate_package(args.package_dir)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print(_format_human_report(report))
    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
