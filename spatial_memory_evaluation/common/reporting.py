from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from spatial_memory_evaluation.common.jsonl import write_json


@dataclass(frozen=True)
class EvaluationOutputPaths:
    summary: Path
    details: Path
    report: Path


def evaluation_output_paths(summary_path: Path | str) -> EvaluationOutputPaths:
    summary = Path(summary_path)
    if summary.name == "eval_summary.json":
        details = summary.with_name("eval_details.json")
        report = summary.with_name("eval_report.md")
    else:
        suffix = summary.suffix or ".json"
        details = summary.with_name(f"{summary.stem}_details{suffix}")
        report = summary.with_name(f"{summary.stem}_report.md")
    return EvaluationOutputPaths(summary=summary, details=details, report=report)


def write_evaluation_outputs(
    *,
    summary_path: Path | str,
    summary: Mapping[str, Any],
    details: Mapping[str, Any],
    report_markdown: str,
) -> EvaluationOutputPaths:
    paths = evaluation_output_paths(summary_path)
    write_json(paths.summary, dict(summary))
    write_json(paths.details, dict(details))
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    paths.report.write_text(report_markdown, encoding="utf-8")
    return paths


def render_evaluation_report(
    *,
    title: str,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    status: str,
    summary_path: Path,
    details_path: Path,
    result: Mapping[str, Any] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    lines.extend(["## Metadata", "", _markdown_table(_rows_from_mapping(metadata)), ""])
    if status == "ok" and metrics:
        lines.extend(["## Metrics", "", _markdown_table(_rows_from_mapping(metrics)), ""])
    else:
        lines.extend(["## Result", "", _markdown_table(_rows_from_mapping(_result_rows(status, result))), ""])
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- [Summary]({summary_path.name})",
            f"- [Details]({details_path.name})",
            "",
        ]
    )
    return "\n".join(lines)


def report_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), Mapping) else {}
    return {
        "status": summary.get("status"),
        "track": summary.get("track"),
        "method": summary.get("method"),
        "mode": summary.get("mode"),
        "scene_id": dataset.get("scene_id") if isinstance(dataset, Mapping) else None,
        "package_dir": summary.get("package_dir"),
    }


def report_title(summary: Mapping[str, Any]) -> str:
    dataset = summary.get("dataset") if isinstance(summary.get("dataset"), Mapping) else {}
    scene_id = dataset.get("scene_id") if isinstance(dataset, Mapping) else None
    pieces = [
        str(summary.get("method") or "unknown"),
        str(summary.get("track") or "unknown_track"),
        str(summary.get("mode") or "unknown_mode"),
    ]
    if scene_id:
        pieces.append(str(scene_id))
    return " / ".join(pieces)


def _result_rows(status: str, result: Mapping[str, Any] | None) -> dict[str, Any]:
    rows: dict[str, Any] = {"status": status}
    if not result:
        return rows
    for key in ("reason_code", "message", "reason", "required_api", "package_path"):
        if key in result:
            rows[key] = result[key]
    return rows


def _rows_from_mapping(values: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(str(key), _format_value(value)) for key, value in values.items()]


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, dict)):
        return "`" + str(value) + "`"
    return str(value)


def _markdown_table(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return "| Field | Value |\n| --- | --- |"
    output = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        output.append(f"| {_escape_cell(key)} | {_escape_cell(value)} |")
    return "\n".join(output)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
