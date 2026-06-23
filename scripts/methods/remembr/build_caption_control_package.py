"""Build an LLM-with-captions Track 1/2 *control* memory package.

This is the caption-memory control declared in
`.codex/baseline_registry.md` ("LLM-With-Captions Track 1/2 Control Outcome").
Caption-only memory may be packaged as captions/keyframes/provenance, but it is
NOT an object-memory fixed API: it has no native object-level labels + 3D
positions (Track 1) and no deterministic native object-location query API over
caption memory (Track 2). An LLM caption answerer is never fixed-API support.

The package therefore:

- sets `manifest.method.family = "caption_control"` and
  `manifest.explicit_memory = false`;
- declares all four fixed-API tracks `invalid`, each with a specific reason that
  points at the native capability gap (not at an adapter limitation);
- still preserves the caption artifact honestly under `memory/` so agentic
  Track 1/2 (with full sandbox access) can use it later.

Root-repo evidence for the capability gap lives in the ReMEmbR repo and is cited
in `schema.md` and the registry. The external repo is read-only; this script
only reads its caption JSON and never edits it.

Examples
--------
Build from a native ReMEmbR caption JSON (read-only input)::

    python scripts/methods/remembr/build_caption_control_package.py \
        --captions-json /home/robin_wang/remembr/data/captions/seq0/captions/captions_Llama-3-VILA1.5-8b_3_secs.json \
        --dataset oc-navqa --episode-id seq0 --run-id remembr-captions-20260617

Build the committed offline example fixture (synthetic captions, deterministic)::

    python scripts/methods/remembr/build_caption_control_package.py \
        --synthetic \
        --output-dir examples/caption_control_package \
        --dataset example --episode-id seq0 \
        --started-at 2026-06-17T00:00:00+08:00 \
        --finished-at 2026-06-17T00:00:00+08:00
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.build_accounting import (  # noqa: E402
    write_build_log_with_accounting,
)
from spatial_memory_evaluation.common.jsonl import write_json, write_jsonl  # noqa: E402
from spatial_memory_evaluation.memory_package_validator import validate_package  # noqa: E402


REMEMBR_REPO = Path("/home/robin_wang/remembr")

# Reasons are phrased against the *native* capability gap so an invalid fixed-API
# result is specific and actionable, not a vague "not implemented".
TRACK1_INVALID_REASON = (
    "Caption memory has no native object-level inventory and no deterministic "
    "native object-location query API. ReMEmbR MemoryItem "
    "(remembr/memory/memory.py:5-9) stores only caption/time/position/theta with "
    "no object field, so there is no native source of object labels + 3D positions "
    "to export as an object table. TextMemory/MilvusMemory.memory_to_string "
    "(remembr/memory/text_memory.py:40, remembr/memory/milvus_memory.py:231) "
    "return caption+pose+time strings, not object-level locations. The only "
    "answerer, NonAgent.query (remembr/agents/non_agent.py:40-91), is an LLM over "
    "caption context and is excluded from fixed-API support by design."
)
TRACK3_INVALID_REASON = (
    "No referring-expression resolver. Caption memory cannot ground a ScanRefer "
    "utterance to a target object id/bbox."
)
TRACK4_INVALID_REASON = (
    "No method-native QA/retrieval fixed API in caption memory. NonAgent.query is "
    "a no-retrieval LLM caption-context answerer; ReMEmbR's native QA path is the "
    "agentic ReMEmbRAgent tool-LLM Track 3 path, not this control."
)

# Small deterministic stand-in so the committed example builds with no external
# data. The shape mirrors ReMEmbR caption JSON entries (id/position/theta/time/
# caption) from remembr/scripts/preprocess_captions.py:112-120.
SYNTHETIC_CAPTIONS: list[dict[str, Any]] = [
    {
        "id": "000000.000",
        "position": [0.0, 0.0, 0.0],
        "theta": 3.14,
        "time": 0.0,
        "caption": "The robot is in a corridor with a closed wooden door ahead and a window on the right.",
    },
    {
        "id": "000003.000",
        "position": [1.2, 0.4, 0.0],
        "theta": 3.14,
        "time": 3.0,
        "caption": "A small office with a desk, a monitor, and an office chair near the wall.",
    },
    {
        "id": "000006.000",
        "position": [2.5, 1.1, 0.0],
        "theta": 3.14,
        "time": 6.0,
        "caption": "An open kitchen area; a refrigerator and a sink are visible on the left.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an LLM-with-captions Track 1/2 control memory package.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--captions-json",
        type=Path,
        default=None,
        help="Native ReMEmbR caption JSON (read-only input).",
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a small built-in synthetic caption set (offline/example build).",
    )
    parser.add_argument("--method", default="remembr_captions")
    parser.add_argument("--display-name", default="LLM-with-captions (ReMEmbR caption control)")
    parser.add_argument("--dataset", default="oc-navqa")
    parser.add_argument("--split", default="control")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--episode-id", default="seq0")
    parser.add_argument("--captioner", default="Llama-3-VILA1.5-8b")
    parser.add_argument("--repo-path", type=Path, default=REMEMBR_REPO)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit package output dir. Defaults to memories/<method>/<dataset>/<scene-or-episode>/<run-id>.",
    )
    parser.add_argument("--started-at", default=None, help="ISO time for deterministic builds.")
    parser.add_argument("--finished-at", default=None, help="ISO time for deterministic builds.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    captions, native_json_path = _load_captions(args)
    started_epoch = time.time()

    scene_or_episode = args.scene_id or args.episode_id or "control"
    run_id = args.run_id or f"{args.method}-{_safe(scene_or_episode)}"
    package_dir = args.output_dir or (
        REPO_ROOT / "memories" / args.method / args.dataset / scene_or_episode / run_id
    )
    package_dir = Path(package_dir)
    _reset_package_dirs(package_dir)

    caption_rows = _normalize_caption_rows(captions)
    write_jsonl(package_dir / "memory" / "captions.jsonl", caption_rows)
    native_artifacts: list[Path] = []
    if native_json_path is not None:
        native_copy = package_dir / "memory" / "native" / native_json_path.name
        native_copy.parent.mkdir(parents=True, exist_ok=True)
        native_copy.write_bytes(native_json_path.read_bytes())
        native_artifacts.append(native_copy)

    _write_schemas(package_dir)
    _write_docs(package_dir)
    _write_schema_md(package_dir, caption_count=len(caption_rows))
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        scene_or_episode=scene_or_episode,
        caption_count=len(caption_rows),
        native_json_path=native_json_path,
        has_native=bool(native_artifacts),
        started_at=args.started_at,
        finished_at=args.finished_at,
    )
    _write_capabilities(package_dir)

    started_at = args.started_at or _iso(started_epoch)
    finished_at = args.finished_at or _iso(time.time())
    runtime_seconds = 0.0 if args.started_at else max(0.0, time.time() - started_epoch)
    write_build_log_with_accounting(
        package_dir=package_dir,
        native_memory_artifact_paths=[
            package_dir / "memory" / "captions.jsonl",
            *native_artifacts,
        ],
        frame_count=len(caption_rows),
        build_log={
            "status": "ok",
            "started_at": started_at,
            "finished_at": finished_at,
            "build_runtime_seconds": runtime_seconds,
            "runtime_seconds": runtime_seconds,
            "command": " ".join(sys.argv),
            "config_paths": [],
            "source_outputs": [str(native_json_path)] if native_json_path else [],
            "caption_count": len(caption_rows),
            "control": True,
            "explicit_memory": False,
            "fixed_api": {
                "track1_object_location": "invalid",
                "track2_scanrefer": "invalid",
                "track3_openeqa": "invalid",
            },
            "warnings": [],
        },
    )

    report = validate_package(package_dir)
    print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    if not report.valid:
        return 1
    print(f"caption control package written to {package_dir}")
    return 0


def _load_captions(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path | None]:
    if args.synthetic or args.captions_json is None:
        return SYNTHETIC_CAPTIONS, None
    path = args.captions_json
    if not path.exists():
        raise FileNotFoundError(f"caption JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"expected a JSON list of caption entries in {path}")
    return loaded, path


def _normalize_caption_rows(captions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(captions):
        if not isinstance(entry, dict):
            continue
        rows.append(
            {
                "caption_id": str(entry.get("id", f"caption_{index:04d}")),
                "caption": str(entry.get("caption", "")),
                "time": _as_float(entry.get("time")),
                "position": _as_float_list(entry.get("position")),
                "theta": _as_float(entry.get("theta")),
                "file_start": entry.get("file_start"),
                "file_end": entry.get("file_end"),
            }
        )
    return rows


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(component) for component in value]
    except (TypeError, ValueError):
        return None


def _reset_package_dirs(package_dir: Path) -> None:
    for name in ("memory", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / name).mkdir(parents=True, exist_ok=True)


def _write_schemas(package_dir: Path) -> None:
    write_json(
        package_dir / "schemas" / "captions.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Caption memory row",
            "type": "object",
            "required": ["caption_id", "caption", "time", "position", "theta"],
            "properties": {
                "caption_id": {"type": "string"},
                "caption": {"type": "string"},
                "time": {"type": ["number", "null"]},
                "position": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                },
                "theta": {"type": ["number", "null"]},
                "file_start": {"type": ["string", "null"]},
                "file_end": {"type": ["string", "null"]},
            },
        },
    )


def _write_docs(package_dir: Path) -> None:
    package_dir.joinpath("evidence", "README.md").write_text(
        "# Evidence\n\n"
        "This caption control exports caption text, pose, and timestamp as its only\n"
        "provenance. There are no object crops or object-level evidence because the\n"
        "method has no native object inventory.\n",
        encoding="utf-8",
    )
    package_dir.joinpath("raw_links", "README.md").write_text(
        "# Raw Links\n\n"
        "Raw frames are not linked. Caption memory is itself derived from frames by\n"
        "the captioner; raw-frame access remains a separate ablation.\n",
        encoding="utf-8",
    )
    package_dir.joinpath("tools", "README.md").write_text(
        "# Tools\n\n"
        "No fixed-API Python entrypoints are provided. All four tracks are `invalid`\n"
        "for this caption control, so there is no `list_objects`/`query_object`/\n"
        "`resolve_referring_expression`/`answer_question` entrypoint to expose.\n\n"
        "Agentic Track 1/2 may later read `memory/captions.jsonl` directly in a\n"
        "full-access sandbox; that is separate from the fixed API.\n",
        encoding="utf-8",
    )


def _write_schema_md(package_dir: Path, *, caption_count: int) -> None:
    package_dir.joinpath("schema.md").write_text(
        f"""# LLM-with-captions Caption Control Schema

Coordinate frame and units: caption `position` is the mean robot/camera position
over the caption window, in the native ReMEmbR/dataset world frame. Distances are
in meters. `theta` is a yaw placeholder (radians); ReMEmbR sets it to a constant
(`remembr/scripts/preprocess_captions.py:114`), so it is not a reliable heading.

Object id format: not applicable. This control has no object-level memory and
therefore no object ids. Each row is a caption window keyed by `caption_id`
(the native ReMEmbR caption `id`, typically a frame timestamp string).

Timestamp format: `time` is a float (seconds) — the mean timestamp of the frames
in the caption window. `file_start`/`file_end` reference native frame names when
available.

Relation representation: none. Caption memory stores free text per window with no
object nodes and no relation edges.

Confidence or score meaning: none. Captions carry no per-object confidence or
retrieval score; this control exposes no scored predictions.

Native artifact formats: `memory/captions.jsonl` is UTF-8 JSONL with one caption
window per line (`caption_id`, `caption`, `time`, `position`, `theta`). When built
from a native ReMEmbR caption JSON, the original file is copied verbatim under
`memory/native/`. The native schema is ReMEmbR `MemoryItem`
(`remembr/memory/memory.py:5-9`): `caption`, `time`, `position`, `theta` only.
This package contains {caption_count} caption window(s).

Known limitations and unsupported tracks: this is a no-explicit-memory caption
**control** (`explicit_memory=false`), not an object-memory baseline. All fixed
APIs are `invalid`:

- Track 1 (object inventory): invalid — no native object labels + 3D positions.
- Track 2 (object location): invalid — no deterministic native object-location
  query API; the only answerer (`NonAgent.query`,
  `remembr/agents/non_agent.py:40-91`) is an LLM over caption context and may not
  be used as fixed-API support.
- Track 3 (ScanRefer): invalid — no referring-expression resolver.
- Track 4 (OpenEQA): invalid — caption memory has no method-native QA/retrieval
  fixed API; ReMEmbR's native QA path is the agentic `ReMEmbRAgent`, not this
  control.

The caption artifact is preserved so that agentic Track 1/2 (full sandbox access)
can read and reason over captions later. That agentic use is explicitly separate
from the fixed API and must never promote this control to an object-memory API.
""",
        encoding="utf-8",
    )


def _write_manifest(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    scene_or_episode: str,
    caption_count: int,
    native_json_path: Path | None,
    has_native: bool,
    started_at: str | None,
    finished_at: str | None,
) -> None:
    memory_artifacts: list[dict[str, Any]] = [
        {
            "name": "captions",
            "type": "jsonl",
            "path": "memory/captions.jsonl",
            "description": f"Caption memory: {caption_count} caption window(s) with text, pose, and time.",
            "required_for": [],
        }
    ]
    if has_native and native_json_path is not None:
        memory_artifacts.append(
            {
                "name": "native_captions",
                "type": "json",
                "path": f"memory/native/{native_json_path.name}",
                "description": "Verbatim native ReMEmbR caption JSON.",
                "required_for": [],
            }
        )

    raw_links: list[dict[str, Any]] = []
    if native_json_path is not None:
        raw_links.append(
            {
                "name": "native_caption_source",
                "type": "file",
                "path": str(native_json_path),
                "description": "Native ReMEmbR caption JSON source (read-only).",
                "required_for": [],
            }
        )

    write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.2",
            "package_id": f"{args.method}/{args.dataset}/{scene_or_episode}/{run_id}",
            "method": {
                "name": args.method,
                "display_name": args.display_name,
                "family": "caption_control",
                "repo_path": str(args.repo_path),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": args.dataset,
                "split": args.split,
                "scene_id": args.scene_id,
                "episode_id": args.episode_id,
            },
            "input": {
                "modality": ["rgb", "pose", "timestamp"],
                "frame_count": caption_count,
                "rgbd_root": None,
                "poses_path": None,
                "intrinsics_path": None,
                "timestamp_path": None,
                "coordinate_frame": "native ReMEmbR/dataset world frame; meters; theta is a constant yaw placeholder",
            },
            "vocabulary": {
                "vocabulary_mode": "module_ablation",
                "class_list_path": None,
                "source": "free-text VILA captions; no detector vocabulary",
                "captioner": args.captioner,
            },
            "modules": {
                "captioner": args.captioner,
                "answerer": "remembr.agents.non_agent.NonAgent (LLM caption-context; not a fixed API)",
                "native_memory": "remembr.memory.MemoryItem (caption/time/position/theta)",
            },
            "explicit_memory": False,
            "memory_artifacts": memory_artifacts,
            "evidence_artifacts": [],
            "raw_links": raw_links,
            "tools": [],
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [],
                "environment": None,
                "started_at": started_at,
                "finished_at": finished_at,
                "build_runtime_seconds": None,
                "runtime_seconds": None,
                "frame_count": caption_count,
                "time_per_frame_seconds": None,
                "native_memory_size_bytes": None,
                "native_memory_artifacts": [],
                "memory_artifact_size_bytes": None,
                "package_size_bytes": None,
                "peak_ram_bytes": None,
                "peak_ram_unavailable_reason": None,
                "peak_vram_bytes": None,
                "peak_vram_unavailable_reason": None,
            },
            "allowed_access": {
                "contains_gt_annotations": False,
                "contains_benchmark_answers": False,
                "contains_test_labels": False,
                "contains_question_specific_rules": False,
            },
            "notes": (
                "LLM-with-captions Track 1/2 control. Caption-only memory packaged as "
                "captions/provenance; not an object-memory fixed API. The external "
                "ReMEmbR repo was read-only and was not modified."
            ),
        },
    )


def _write_capabilities(package_dir: Path) -> None:
    write_json(
        package_dir / "capabilities.json",
        {
            "schema_version": "0.2",
            "fixed_api": {
                "track1_object_location": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": TRACK1_INVALID_REASON,
                },
                "track2_scanrefer": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": TRACK3_INVALID_REASON,
                },
                "track3_openeqa": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": TRACK4_INVALID_REASON,
                },
            },
            "agent_access": {
                "mode": "tool_llm",
                "read_manifest": True,
                "read_schema": True,
                "read_native_memory": True,
                "read_fixed_api_views": False,
                "read_evidence": True,
                "read_adapter_code": False,
                "read_shared_module_code": False,
                "read_method_root_source_code": True,
                "read_build_code": False,
                "read_raw_links": False,
                "read_raw_frames": False,
                "read_source_keyframes_or_crops": False,
                "run_method_native_tools": True,
                "write_package": False,
            },
        },
    )


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(epoch))


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
