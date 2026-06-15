from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_objects(package_dir: str, query: dict[str, Any]) -> dict[str, Any]:
    object_path = Path(package_dir) / "memory" / "object_table.jsonl"
    objects = []
    with object_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return {"status": "ok", "objects": objects}
