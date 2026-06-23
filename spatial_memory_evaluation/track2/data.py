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


def build_track2_data(
    *,
    scanrefer_root: Path,
    scannet_root: Path,
    output_dir: Path,
    top_k: int = 10,
) -> dict[str, Any]:
    """Build the ScanRefer referring-query benchmark.

    SKELETON: the parsing of ScanRefer JSON into ``referring_queries.jsonl`` and
    of ScanNet instance boxes into ``scene_objects.jsonl`` is not implemented
    until the dataset is on NAS. The function documents the intended inputs and
    fails loudly rather than producing an empty benchmark.
    """

    raise NotImplementedError(
        "build_track2_data is a skeleton. Implement ScanRefer parsing once the "
        f"dataset is available. Expected ScanRefer root={scanrefer_root}, "
        f"ScanNet root={scannet_root}, output={output_dir}, top_k={top_k}. "
        "See .codex/agentic_eval_plan.md Phase 2 and .codex/path_registry.md."
    )


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
