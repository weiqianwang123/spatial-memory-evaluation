from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from spatial_memory_evaluation.common.jsonl import read_json, write_json, write_jsonl
from spatial_memory_evaluation.common.labels import (
    DEFAULT_DETECTOR_COVERABLE_LABELS,
    DEFAULT_LABEL_ALIASES,
    normalize_label,
    write_default_alias_file,
)


DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")


def build_track1_data(
    *,
    scannetpp_root: Path,
    scene_id: str,
    output_dir: Path,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "all_annotated.jsonl", all_objects)
    write_jsonl(output_dir / "detector_coverable.jsonl", coverable)
    write_default_alias_file(output_dir / "label_aliases.json")

    summary = {
        "scene_id": scene_id,
        "annotation_path": str(annotation_path),
        "all_annotated_count": len(all_objects),
        "detector_coverable_count": len(coverable),
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "metadata.json", summary)
    return summary


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
