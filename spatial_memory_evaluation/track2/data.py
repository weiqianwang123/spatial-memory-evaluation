"""Track 2 (ScanRefer instance-level referring) benchmark data.

Status (2026-06-23): ScanRefer annotations are not yet on NAS (see
`.codex/path_registry.md`). This builder is a skeleton: it knows where ScanRefer
should live, validates presence, and either builds the referring-query benchmark
or returns a structured ``data_unavailable`` summary so the evaluator can emit a
clear, non-silent result.

Target benchmark layout:

```text
benchmarks/track2/scanrefer/<scannet-split>/
  referring_queries.jsonl   # one row per ScanRefer utterance
  scene_objects.jsonl       # GT instance boxes per scene (for IoU / center scoring)
  metadata.json
```

Each referring query row (target schema):

```json
{
  "query_id": "scanrefer_0001",
  "dataset": "scanrefer",
  "scene_id": "scene0000_00",
  "utterance": "the red chair next to the table",
  "target_object_id": "<scannet instance id>",
  "target_bbox_3d": [x0,y0,z0,x1,y1,z1],
  "top_k": 10
}
```

GT (``target_object_id`` / ``target_bbox_3d``) is held by the evaluator only and
must never be copied into a tool-LLM sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_jsonl, write_json


# Candidate ScanNet geometry source for ScanRefer (annotations TBD).
DEFAULT_SCANNET_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannet")
REFERRING_QUERIES_FILE = "referring_queries.jsonl"
SCENE_OBJECTS_FILE = "scene_objects.jsonl"


def track2_data_status(benchmark_dir: Path) -> dict[str, Any]:
    """Report whether the ScanRefer benchmark is available at ``benchmark_dir``.

    Returns ``{"status": "ok", ...}`` when the referring queries exist, otherwise
    ``{"status": "data_unavailable", ...}`` with an acquisition pointer. The
    evaluator uses this to avoid silently scoring an empty benchmark.
    """

    queries_path = benchmark_dir / REFERRING_QUERIES_FILE
    if queries_path.exists():
        try:
            count = len(read_jsonl(queries_path))
        except (ValueError, OSError) as exc:
            return {
                "status": "data_unavailable",
                "reason": f"failed to read {queries_path}: {exc}",
                "benchmark_dir": str(benchmark_dir),
            }
        return {
            "status": "ok",
            "benchmark_dir": str(benchmark_dir),
            "query_count": count,
            "queries_path": str(queries_path),
            "scene_objects_path": str(benchmark_dir / SCENE_OBJECTS_FILE),
        }
    return {
        "status": "data_unavailable",
        "reason": (
            "ScanRefer referring queries not found. Acquire ScanRefer annotations "
            "aligned to ScanNet scans and build the benchmark; see "
            ".codex/path_registry.md (Track 2/3 datasets to acquire)."
        ),
        "benchmark_dir": str(benchmark_dir),
        "expected_files": [REFERRING_QUERIES_FILE, SCENE_OBJECTS_FILE],
        "scannet_root": str(DEFAULT_SCANNET_ROOT),
    }


# Track 2 referring data. The primary public source is ScanEnts3D
# (https://scanents3d.github.io/), a superset of ScanRefer whose val split has GT
# target object_id + object_name and an `entities` array grounding every phrase to
# ScanNet instance ids (target + anchors). Fields used: scene_id, object_id,
# object_name, ann_id, description, entities. ScanEnts3D references objects by
# ScanNet instance id (no bbox in the json), so a caption-memory method like
# ReMEmbR is scored at the target object-name level (does the resolved object match
# the GT object_name); anchor object names are kept for analysis. The older public
# ScanRefer test json has the same shared fields but no entities.
DEFAULT_SCANENTS3D_VAL_JSON = Path(
    "/data/mondo-training-dataset/semantic_mapping/scanents3d/ScanRefer_filtered_val_ScanEnts3D.json"
)
DEFAULT_SCANREFER_JSON = DEFAULT_SCANENTS3D_VAL_JSON


def build_track2_data(
    *,
    scanrefer_json: Path,
    output_dir: Path,
    scene_id: str | None = None,
    top_k: int = 10,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Build the Track 2 referring benchmark from a ScanRefer/ScanEnts3D JSON.

    Each query row carries the referring ``utterance`` plus GT
    ``target_object_id`` / ``target_object_name`` and, when present (ScanEnts3D),
    the ``anchor_object_names`` parsed from ``entities``. There is no 3D bbox in
    the json, so Track 2 scores referring at the target object-name level; if a
    future split provides ``bbox`` the evaluator can also score IoU.
    """

    rows = read_jsonl_or_json(scanrefer_json)
    if scene_id is not None:
        rows = [r for r in rows if str(r.get("scene_id")) == scene_id]
    if max_queries is not None:
        rows = rows[:max_queries]
    if not rows:
        raise ValueError(f"no referring rows found (scene_id={scene_id}) in {scanrefer_json}")

    output_dir.mkdir(parents=True, exist_ok=True)
    queries = []
    for index, r in enumerate(rows):
        sid = str(r.get("scene_id"))
        target_id = str(r.get("object_id"))
        queries.append(
            {
                "query_id": f"{sid}_{target_id}_{r.get('ann_id')}_{index:04d}",
                "dataset": "scanents3d",
                "scene_id": sid,
                "utterance": r.get("description"),
                "target_object_id": target_id,
                "target_object_name": _normalize_object_name(r.get("object_name")),
                "anchor_object_names": _anchor_object_names(r.get("entities"), target_id),
                "top_k": top_k,
            }
        )
    from spatial_memory_evaluation.common.jsonl import write_jsonl

    write_jsonl(output_dir / REFERRING_QUERIES_FILE, queries)
    summary = {
        "status": "ok",
        "track": "track2_scanrefer",
        "scene_id": scene_id,
        "query_count": len(queries),
        "scoring": "object_name (ScanEnts3D references instance ids, no 3D bbox in json)",
        "source": str(scanrefer_json),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metadata.json", summary)
    return summary


def _anchor_object_names(entities: Any, target_id: str) -> list[str]:
    """Extract anchor object names from a ScanEnts3D ``entities`` array.

    Each entity is ``[[token_idxs], ["<id>_<label>", ...]]``. Anchors are the
    referenced objects other than the target id.
    """

    names: list[str] = []
    if not isinstance(entities, list):
        return names
    for entity in entities:
        if not (isinstance(entity, list) and len(entity) == 2 and isinstance(entity[1], list)):
            continue
        for ref in entity[1]:
            if not isinstance(ref, str) or "_" not in ref:
                continue
            obj_id, _, label = ref.partition("_")
            if obj_id == target_id:
                continue
            name = _normalize_object_name(label)
            if name and name not in names:
                names.append(name)
    return names


def read_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    import json

    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    raise ValueError(f"expected a JSON list in {path}")


def _normalize_object_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("_", " ")


def write_unavailable_metadata(output_dir: Path, *, scanrefer_root: Path | None = None) -> dict[str, Any]:
    """Write a metadata stub recording that the ScanRefer benchmark is unbuilt."""

    summary = {
        "status": "data_unavailable",
        "track": "track2_scanrefer",
        "output_dir": str(output_dir),
        "scanrefer_root": str(scanrefer_root) if scanrefer_root else None,
        "note": "ScanRefer not yet acquired; see .codex/path_registry.md.",
    }
    write_json(output_dir / "metadata.json", summary)
    return summary
