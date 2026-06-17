from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def query_object(package_dir: str, query: dict[str, Any]) -> dict[str, Any]:
    query_text = str(query.get("query") or query.get("object") or "").strip().lower()
    top_k = int(query.get("top_k") or 5)
    objects = _load_objects(Path(package_dir))
    predictions = _rank_by_label_and_size(objects, query_text)[:top_k]
    return {"status": "ok", "predictions": predictions}


def _load_objects(package_dir: Path) -> list[dict[str, Any]]:
    path = package_dir / "memory" / "object_table.jsonl"
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def _rank_by_label_and_size(objects: list[dict[str, Any]], query_text: str) -> list[dict[str, Any]]:
    ranked = []
    query_tokens = set(query_text.split())
    for obj in objects:
        label = str(obj.get("label") or "object").lower()
        label_tokens = set(label.split())
        if query_text and query_text == label:
            score = 1.0
        elif query_text and query_text in label:
            score = 0.85
        elif query_tokens and query_tokens & label_tokens:
            score = 0.7
        elif query_text in ("", "object", "objects"):
            score = 0.5
        else:
            score = 0.05
        score += min(float(obj.get("num_points") or 0) / 100000.0, 0.25)
        ranked.append(
            {
                "object_id": obj.get("object_id"),
                "label": obj.get("label"),
                "position_3d": obj.get("position_3d"),
                "bbox_3d": obj.get("bbox_3d"),
                "score": score,
                "evidence": obj.get("evidence", []),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item.get("object_id"))))
    return ranked
