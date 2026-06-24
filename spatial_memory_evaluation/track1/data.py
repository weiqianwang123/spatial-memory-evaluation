"""Track 1 (object-level location query) benchmark data.

After the 3-track refactor, Track 1 merges the old memory-construction inventory
and the old object-location queries into one benchmark directory:

```text
benchmarks/track1/scannetpp/<scene-id>/
  all_annotated.jsonl          # GT object inventory (every annotated object)
  detector_coverable.jsonl     # GT objects whose label is in the shared OV list
  queries_all_annotated.jsonl  # one "where is the <label>?" query per label
  queries_detector_coverable.jsonl
  label_aliases.json
  metadata.json
```

The GT inventory is only used to align predictions to target objects when scoring
the location queries; Track 1 no longer reports inventory recall / false-memory /
redundancy / localization error as separate metrics (those moved out in the
refactor). It keeps the object-location query metrics plus build-cost accounting.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from spatial_memory_evaluation.common.jsonl import read_json, write_json, write_jsonl
from spatial_memory_evaluation.common.labels import (
    DEFAULT_DETECTOR_COVERABLE_LABELS,
    DEFAULT_LABEL_ALIASES,
    SCANNETPP_NON_OBJECT_LABELS,
    normalize_label,
    write_default_alias_file,
)
from spatial_memory_evaluation.track2.scannet_bbox import (
    DEFAULT_SCANNET_SCANS_ROOT,
    resolve_scene_bboxes,
    scene_paths,
)


DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_TOP_K = 10
SPLITS = ("all_annotated", "detector_coverable")

# ScanNet semantic-only labels that are scene structure, not locatable objects;
# they are excluded from Track 1 ScanNet queries (a "where is the wall?" query is
# not a meaningful object-location task and inflates the query set with surfaces).
SCANNET_STRUCTURE_LABELS: set[str] = {"wall", "floor", "ceiling"}


def build_track1_data(
    *,
    scannetpp_root: Path,
    scene_id: str,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    aliases = aliases or DEFAULT_LABEL_ALIASES
    annotation_path = scannetpp_root / "data" / scene_id / "scans" / "segments_anno.json"
    data = read_json(annotation_path)
    groups = data.get("segGroups")
    if not isinstance(groups, list):
        raise ValueError(f"segments_anno.json missing segGroups list: {annotation_path}")

    all_objects = [_gt_object(scene_id, group, aliases) for group in groups]
    coverable = [
        obj
        for obj in all_objects
        if obj["canonical_label"] in DEFAULT_DETECTOR_COVERABLE_LABELS
    ]
    objects_by_split = {"all_annotated": all_objects, "detector_coverable": coverable}

    output_dir.mkdir(parents=True, exist_ok=True)
    query_counts: dict[str, int] = {}
    for split, objects in objects_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", objects)
        queries = _queries_for_split(split, objects, scene_id=scene_id, top_k=top_k)
        write_jsonl(output_dir / f"queries_{split}.jsonl", queries)
        query_counts[split] = len(queries)
    write_default_alias_file(output_dir / "label_aliases.json")

    summary = {
        "scene_id": scene_id,
        "annotation_path": str(annotation_path),
        "all_annotated_count": len(all_objects),
        "detector_coverable_count": len(coverable),
        "query_counts": query_counts,
        "top_k": top_k,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metadata.json", summary)
    return summary


def build_track1_scannet_data(
    *,
    scannet_scans_root: Path,
    scene_id: str,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Track 1 GT inventory + queries for a ScanNet scene.

    Mirrors :func:`build_track1_data` (ScanNet++) but derives the object inventory
    from the ScanNet aggregation (objectId -> label) and the axis-aligned instance
    bbox machinery in ``track2/scannet_bbox.py`` (the same GT geometry Track 2
    uses, so all three tracks share one ground-truth geometry for these scenes).
    Boxes are emitted in the *unaligned* mesh frame to match the poses the memory
    builders consume (``.sens`` export). Structure-only labels (wall/floor/ceiling)
    are dropped — they are not object-location targets.
    """

    aliases = aliases or DEFAULT_LABEL_ALIASES
    paths = scene_paths(scene_id, scannet_scans_root)
    agg_path = paths["aggregation"]
    agg = read_json(agg_path)
    groups = agg.get("segGroups")
    if not isinstance(groups, list):
        raise ValueError(f"aggregation.json missing segGroups list: {agg_path}")

    bboxes = resolve_scene_bboxes(scene_id, scans_root=scannet_scans_root, apply_axis_align=False)

    all_objects: list[dict[str, Any]] = []
    for group in groups:
        object_id = int(group.get("objectId"))
        raw_label = str(group.get("label") or "")
        canonical = normalize_label(raw_label, aliases)
        if canonical in SCANNET_STRUCTURE_LABELS or canonical in SCANNETPP_NON_OBJECT_LABELS:
            continue
        bbox = bboxes.get(object_id)
        if bbox is None:
            continue
        center = [
            (bbox[0] + bbox[3]) / 2.0,
            (bbox[1] + bbox[4]) / 2.0,
            (bbox[2] + bbox[5]) / 2.0,
        ]
        all_objects.append(
            {
                "gt_id": str(object_id),
                "object_id": object_id,
                "scene_id": scene_id,
                "raw_label": raw_label,
                "canonical_label": canonical,
                "center_3d": center,
                "bbox_3d": list(bbox),
                "bbox_diag_m": _bbox_diag(bbox[:3], bbox[3:]),
                "segment_count": len(group.get("segments") or []),
                "segments": group.get("segments") or [],
            }
        )

    coverable = [
        obj for obj in all_objects if obj["canonical_label"] in DEFAULT_DETECTOR_COVERABLE_LABELS
    ]
    objects_by_split = {"all_annotated": all_objects, "detector_coverable": coverable}

    output_dir.mkdir(parents=True, exist_ok=True)
    query_counts: dict[str, int] = {}
    for split, objects in objects_by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", objects)
        queries = _queries_for_split(split, objects, scene_id=scene_id, top_k=top_k)
        write_jsonl(output_dir / f"queries_{split}.jsonl", queries)
        query_counts[split] = len(queries)
    write_default_alias_file(output_dir / "label_aliases.json")

    summary = {
        "scene_id": scene_id,
        "dataset": "scannet",
        "annotation_path": str(agg_path),
        "all_annotated_count": len(all_objects),
        "detector_coverable_count": len(coverable),
        "query_counts": query_counts,
        "top_k": top_k,
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metadata.json", summary)
    return summary


def _queries_for_split(
    split: str,
    gt_objects: list[dict[str, Any]],
    *,
    scene_id: str,
    top_k: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in gt_objects:
        grouped[str(obj["canonical_label"])].append(obj)
    queries = []
    for index, label in enumerate(sorted(grouped)):
        targets = grouped[label]
        queries.append(
            {
                "query_id": f"{scene_id}_{split}_{index:04d}_{label.replace(' ', '_')}",
                "scene_id": scene_id,
                "split": split,
                "canonical_label": label,
                "target_label": label,
                "query": f"where is the {label}?",
                "top_k": top_k,
                "target_gt_ids": [str(obj["gt_id"]) for obj in targets],
            }
        )
    return queries


def _gt_object(scene_id: str, group: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    obb = group.get("obb")
    if not isinstance(obb, Mapping):
        raise ValueError(f"segGroup is missing obb: {group.get('id')}")
    center = _vector(obb.get("centroid"), 3)
    min_bound = _vector(obb.get("min"), 3)
    max_bound = _vector(obb.get("max"), 3)
    if center is None or min_bound is None or max_bound is None:
        raise ValueError(f"segGroup has invalid obb vectors: {group.get('id')}")
    raw_label = str(group.get("label") or "")
    return {
        "gt_id": str(group.get("id")),
        "object_id": group.get("objectId"),
        "scene_id": scene_id,
        "raw_label": raw_label,
        "canonical_label": normalize_label(raw_label, aliases),
        "center_3d": center,
        "bbox_3d": min_bound + max_bound,
        "bbox_diag_m": _bbox_diag(min_bound, max_bound),
        "segment_count": len(group.get("segments") or []),
        "segments": group.get("segments") or [],
    }


def _vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _bbox_diag(min_bound: list[float], max_bound: list[float]) -> float:
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(min_bound, max_bound)))
