"""Generate the ReMEmbR Track 1/2/3/4 fixed-API *invalid* declaration package.

ReMEmbR is a caption / spatio-temporal episodic memory. Its native memory has
no object inventory and no object-location read path, so the object-level fixed
APIs (Track 1/2/3) are ``invalid`` by design, and Track 4 / OC-NaVQA temporal QA
is served by the interactive ``ReMEmbRAgent.query`` LangGraph agent rather than a
non-interactive package entrypoint (see ``.codex/baseline_registry.md`` ReMEmbR
section for the full root-repo evidence).

This is a *declaration-only* package: it deliberately exports no native memory
artifacts. Because declaration packages are still generated artifacts, they live
under the gitignored ``memories/`` tree and are produced by running this script,
not committed to git.

Run:

    python scripts/methods/remembr/build_invalid_declaration.py
    python scripts/methods/remembr/build_invalid_declaration.py --run-id 20260617-000000

The script writes a validated package under
``memories/remembr/<dataset>/<episode>/<run-id>/`` and exits non-zero if the
package fails ``memory_package_validator``.

It does not read, run, or modify the ReMEmbR repo; the evidence below was audited
read-only against ``/home/robin_wang/remembr``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.build_accounting import (
    write_build_log_with_accounting,
)
from spatial_memory_evaluation.memory_package_validator import validate_package

DEFAULT_DATASET = "oc-navqa"
DEFAULT_EPISODE_ID = "sequence_0"
REMEMBR_REPO_PATH = "/home/robin_wang/remembr"

# Package-relative directories required by the memory-package spec. They are kept
# empty on purpose for a declaration-only package; a tracked .gitkeep is written
# so the layout is self-describing.
REQUIRED_DIRS = ("memory", "evidence", "raw_links", "schemas", "tools")


def _manifest(*, package_id: str, dataset: str, episode_id: str) -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "package_id": package_id,
        "method": {
            "name": "remembr",
            "display_name": "ReMEmbR",
            "family": "caption_memory",
            "repo_path": REMEMBR_REPO_PATH,
            "commit": None,
            "version": None,
        },
        "dataset": {
            "name": dataset,
            "split": "declaration-only",
            "scene_id": None,
            "episode_id": episode_id,
        },
        "input": {
            "modality": ["rgb", "pose", "timestamp"],
            "frame_count": 0,
            "rgbd_root": None,
            "poses_path": None,
            "intrinsics_path": None,
            "timestamp_path": None,
            "coordinate_frame": (
                "ReMEmbR stores per-memory robot pose (position + theta), not "
                "object positions; robot frame is dataset-defined, meters; see "
                "schema.md"
            ),
        },
        "explicit_memory": True,
        "memory_artifacts": [],
        "evidence_artifacts": [],
        "raw_links": [],
        "tools": [],
        "build": {
            "command": (
                "scripts/methods/remembr/build_invalid_declaration.py; no ReMEmbR "
                "memory was built. ReMEmbR has no object inventory or "
                "object-location API, so Track 1/2 fixed APIs are invalid by design."
            ),
            "config_paths": [],
            "environment": None,
            "started_at": None,
            "finished_at": None,
            "runtime_seconds": 0.0,
        },
        "allowed_access": {
            "contains_gt_annotations": False,
            "contains_benchmark_answers": False,
            "contains_test_labels": False,
            "contains_question_specific_rules": False,
        },
        "notes": (
            "Minimal invalid declaration for ReMEmbR fixed APIs. ReMEmbR is a real "
            "caption/spatio-temporal memory method (explicit_memory=true) but its "
            "native memory has no object inventory and no object-location read "
            "path, so Track 1/2/3 fixed APIs are invalid. Its native QA path "
            "(ReMEmbRAgent.query) is an interactive LangGraph agent over caption "
            "memory; Track 4 fixed API is declared invalid here pending a "
            "non-interactive smoke, and the primary comparison path is agentic. No "
            "object table was exported. See schema.md and "
            ".codex/baseline_registry.md (ReMEmbR section)."
        ),
    }


def _capabilities() -> dict[str, Any]:
    return {
        "schema_version": "0.2",
        "fixed_api": {
            "track1_object_location": {
                "status": "invalid",
                "entrypoint": None,
                "reason": (
                    "ReMEmbR memory has no object inventory and no object-location "
                    "query/read API. MemoryItem stores only "
                    "caption/time/position(robot)/theta (remembr/memory/memory.py:5-9) "
                    "and the Milvus collection has no object/label/bbox field "
                    "(remembr/memory/milvus_memory.py:37-45), so no label+3D object "
                    "table can be honestly exported. The only readers "
                    "(search_by_text/search_by_position/search_by_time) return "
                    "memory_to_string caption+robot-pose+time text "
                    "(remembr/memory/milvus_memory.py:173-250); position is the robot "
                    "pose, not an object position, and the output is free text, not "
                    "deterministic object locations."
                ),
            },
            "track2_scanrefer": {
                "status": "invalid",
                "entrypoint": None,
                "reason": (
                    "No referring-expression resolver exists in the ReMEmbR repo; "
                    "there is no object representation to resolve a referring "
                    "expression against."
                ),
            },
            "track3_openeqa": {
                "status": "invalid",
                "entrypoint": None,
                "reason": (
                    "ReMEmbR's native QA path is the interactive ReMEmbRAgent.query "
                    "LangGraph agent (remembr/agents/remembr_agent.py:390). It is not "
                    "exported as a non-interactive package entrypoint here; Track 3 / "
                    "OC-NaVQA temporal QA is ReMEmbR's natural first track but is "
                    "evaluated via the agentic full-access path, and stays a candidate "
                    "pending a non-interactive remembr+<llm> smoke."
                ),
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
    }


SCHEMA_MD = """# ReMEmbR Memory Schema (Track 1/2 Invalid Declaration)

This is a minimal **declaration-only** package. It records the honest fixed-API
outcome for ReMEmbR on the object-level tracks. No native ReMEmbR memory was
built or exported here; the `memory/`, `evidence/`, `raw_links/`, `schemas/`, and
`tools/` directories are intentionally empty.

All evidence below points at the read-only ReMEmbR root repo
(`/home/robin_wang/remembr`), never at evaluation-repo adapters.

## What ReMEmbR Memory Is

ReMEmbR is a caption / spatio-temporal episodic memory. A memory record is a
`MemoryItem` with exactly four fields — `caption`, `time`, `position`, `theta`
(`remembr/memory/memory.py:5-9`). The native store (`MilvusMemory`) persists
these as a Milvus collection with fields `id`, `text_embedding`, `position`,
`theta`, `time`, `caption` (`remembr/memory/milvus_memory.py:37-45`). There is
no object, label, class, or bounding-box field anywhere in the schema.

## Coordinate Frame and Units

`position` is the **robot's** average pose for a captioned moment, not an
object's position. It is a 3-vector in the dataset-defined robot/world frame,
with distances in meters; `theta` is the robot orientation in radians
(`remembr/memory/milvus_memory.py:40-41,247`). Because memory never localizes
objects, there is no object-level coordinate frame to document.

## Object ID Format

None. ReMEmbR memory has no object entities and therefore no object ids. This is
the core reason Track 1 (object inventory) and Track 2 (object location) are
invalid.

## Timestamp Format

`time` is a float second offset (stored as a 2-D vector with a fixed subtract
offset, `remembr/memory/milvus_memory.py:42,130`). `memory_to_string` renders it
as a `%Y-%m-%d %H:%M:%S` wall-clock string (`milvus_memory.py:244-247`). Time is
a first-class memory axis, which is why ReMEmbR's natural first track is temporal
QA (Track 4), not object tracks.

## Relation Representation

None. ReMEmbR stores free-text captions, not structured object-object relations.
Any spatial relation is implicit in the caption text and is only recoverable
through the LLM retrieval agent, not through a deterministic relation table.

## Confidence or Score Meaning

There is no per-object confidence. Retrieval returns Milvus L2 similarity scores
over the `text_embedding` / `position` / `time` vectors
(`remembr/memory/milvus_memory.py:50-69,294-312`), but these are caption/pose/time
retrieval distances, not object detection or localization confidences.

## Native Artifact Formats

- VILA caption JSON files written by `scripts/preprocess_captions.py`.
- A Milvus collection of `MemoryItem` rows (caption / robot pose / time / text
  embedding), schema at `remembr/memory/milvus_memory.py:37-45`.
- Read path: `search_by_text` / `search_by_position` / `search_by_time`, all of
  which return the `memory_to_string` free-text rendering
  (`remembr/memory/milvus_memory.py:173-250`).

None of these is an object table, so this package exports no `memory/`
artifacts.

## Known Limitations and Unsupported Tracks

- **Track 1 (object inventory): invalid.** No object/label/3D-position memory
  exists to enumerate (`memory/memory.py:5-9`, `milvus_memory.py:37-45`).
- **Track 2 (object location): invalid.** No object-location query/read API. The
  only readers return caption + robot-pose + time text, and `position` is the
  robot pose, not an object position (`milvus_memory.py:173-250`).
- **Track 3 (ScanRefer referring): invalid.** No referring-expression resolver
  and no object representation to resolve against.
- **Track 4 (OpenEQA / temporal QA): invalid fixed API in this package.** The
  native QA path is the interactive `ReMEmbRAgent.query` LangGraph agent
  (`remembr/agents/remembr_agent.py:390`) over the `retrieve_from_text` /
  `retrieve_from_position` / `retrieve_from_time` tools
  (`remembr_agent.py:168,183,199`). This is ReMEmbR's natural first track, but it
  is evaluated through the **agentic full-access** path, not a fixed object API.
  Track 4 stays a candidate pending a non-interactive `remembr+<llm>` smoke; this
  package does not promote it to `supported`.

Per `.codex/memory_package_spec.md` and the Task 19 implementation rules, the
correct outcome is to declare these tracks invalid rather than wrap captions in a
generic text-to-location LLM. The detailed audit lives in
`.codex/baseline_registry.md` (ReMEmbR section).
"""


def _readme(*, package_rel: str) -> str:
    return f"""# ReMEmbR — Track 1/2 Fixed API Invalid Declaration

Minimal declaration-only package recording the honest fixed-API outcome for
ReMEmbR on the object-level tracks.

- **Track 1 (object inventory):** `invalid` — ReMEmbR memory has no object
  inventory (`MemoryItem` = caption/time/position(robot)/theta).
- **Track 2 (object location):** `invalid` — no object-location query/read API;
  readers return caption + robot-pose + time text only.
- **Track 3 (ScanRefer):** `invalid` — no referring-expression resolver.
- **Track 4 (OpenEQA / temporal QA):** `invalid` fixed API here; the native
  `ReMEmbRAgent.query` retrieval agent is ReMEmbR's natural first track but is
  evaluated through the agentic full-access path, and remains a candidate
  pending a non-interactive `remembr+<llm>` smoke.

This package contains no native memory artifacts on purpose. See `schema.md` for
the full root-repo evidence and `.codex/baseline_registry.md` (ReMEmbR section)
for the registry decision.

This package is a generated artifact (it lives under the gitignored `memories/`
tree). Regenerate it with:

```bash
python scripts/methods/remembr/build_invalid_declaration.py
```

Validate it with:

```bash
python -m spatial_memory_evaluation.memory_package_validator \\
  {package_rel}
```
"""


def build_declaration_package(
    *,
    package_root: Path,
    run_id: str,
    dataset: str,
    episode_id: str,
) -> dict[str, Any]:
    package_dir = package_root / "remembr" / dataset / episode_id / run_id
    package_id = f"remembr/{dataset}/{episode_id}/{run_id}"

    package_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRS:
        sub = package_dir / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / ".gitkeep").write_text("", encoding="utf-8")

    _write_json(package_dir / "manifest.json", _manifest(
        package_id=package_id, dataset=dataset, episode_id=episode_id,
    ))
    _write_json(package_dir / "capabilities.json", _capabilities())
    (package_dir / "schema.md").write_text(SCHEMA_MD, encoding="utf-8")
    package_rel = f"memories/remembr/{dataset}/{episode_id}/{run_id}"
    (package_dir / "README.md").write_text(
        _readme(package_rel=package_rel), encoding="utf-8"
    )

    build_log = {
        "status": "ok",
        "started_at": None,
        "finished_at": None,
        "runtime_seconds": 0.0,
        "command": (
            "scripts/methods/remembr/build_invalid_declaration.py; no ReMEmbR "
            "memory was built."
        ),
        "config_paths": [],
        "source_outputs": [],
        "warnings": [
            "Declaration-only package: ReMEmbR Track 1/2 fixed APIs are invalid "
            "by design (object-free caption/pose/time memory). No native memory "
            "artifact was constructed or exported."
        ],
    }
    write_build_log_with_accounting(
        package_dir=package_dir,
        build_log=build_log,
        native_memory_artifact_paths=[],
        frame_count=0,
    )

    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))

    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "validation": report.to_json(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the ReMEmbR fixed-API invalid declaration package."
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("memories"),
        help="Root directory for generated memory packages (gitignored).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id under the package path. Default: current YYYYMMDD-HHMMSS.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episode-id", default=DEFAULT_EPISODE_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or _run_timestamp()
    summary = build_declaration_package(
        package_root=args.package_root,
        run_id=run_id,
        dataset=args.dataset,
        episode_id=args.episode_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
