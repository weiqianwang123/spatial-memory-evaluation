"""Track 1 (track1_object_location) fixed-API entrypoint for the example package.

Looks up objects in the exported object table by label. Prefers an exact match on
``target_label``/``canonical_label`` and falls back to a substring match on the
free-text query, returning up to ``top_k`` candidate objects in the Track 1
prediction format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_objects(package_dir: str) -> list[dict[str, Any]]:
    object_path = Path(package_dir) / "memory" / "object_table.jsonl"
    objects = []
    with object_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def query_object(package_dir: str, query: dict[str, Any]) -> dict[str, Any]:
    objects = _load_objects(package_dir)
    target_label = (query.get("target_label") or query.get("canonical_label") or "").strip().lower()
    free_text = (query.get("query") or "").strip().lower()
    try:
        top_k = int(query.get("top_k", 5))
    except (TypeError, ValueError):
        top_k = 5

    predictions = []
    for obj in objects:
        label = str(obj.get("label") or "").strip().lower()
        aliases = [str(a).strip().lower() for a in (obj.get("aliases") or [])]
        if target_label and (label == target_label or target_label in aliases):
            score = 1.0
        elif free_text and (label in free_text or any(a in free_text for a in aliases)):
            score = 0.5
        else:
            continue
        predictions.append(
            {
                "object_id": obj.get("object_id"),
                "label": obj.get("label"),
                "position_3d": obj.get("position_3d"),
                "bbox_3d": obj.get("bbox_3d"),
                "score": score,
                "evidence": obj.get("evidence", []),
            }
        )

    predictions.sort(key=lambda p: p["score"], reverse=True)
    return {"status": "ok", "predictions": predictions[: max(1, top_k)]}
