from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LABEL_ALIASES: dict[str, str] = {
    "babyfoot table": "table",
    "backpack": "bag",
    "bag": "bag",
    "basket": "basket",
    "blanket": "blanket",
    "blinds": "blinds",
    "book": "book",
    "bottle": "bottle",
    "box": "box",
    "boxes": "box",
    "brief": "briefcase",
    "broom": "broom",
    "cabinet": "cabinet",
    "can": "can",
    "carpet": "carpet",
    "ceiling lamp": "lamp",
    "chair": "chair",
    "clock": "clock",
    "container": "container",
    "counter": "counter",
    "cup": "cup",
    "curtain": "curtain",
    "curtains": "curtain",
    "cushion": "cushion",
    "foot rest": "foot rest",
    "heater": "heater",
    "keyboard": "keyboard",
    "lamp": "lamp",
    "medical machine": "medical machine",
    "monitor": "monitor",
    "mouse": "mouse",
    "pc": "computer",
    "projection curtain": "curtain",
    "sink": "sink",
    "sofa": "sofa",
    "speaker": "speaker",
    "speakers": "speaker",
    "standing lamp": "lamp",
    "suitcase": "suitcase",
    "table": "table",
    "tablet": "tablet",
    "tissu box": "tissue box",
    "tissue box": "tissue box",
    "toilet paper dispensor": "toilet paper dispenser",
    "trash bin": "trash can",
    "trash can": "trash can",
    "tripod": "tripod",
    "whiteboard": "whiteboard",
}

DEFAULT_DETECTOR_COVERABLE_LABELS: set[str] = {
    "bag",
    "basket",
    "blanket",
    "blinds",
    "book",
    "bottle",
    "box",
    "broom",
    "cabinet",
    "can",
    "carpet",
    "chair",
    "clock",
    "computer",
    "container",
    "counter",
    "cup",
    "curtain",
    "cushion",
    "foot rest",
    "heater",
    "keyboard",
    "lamp",
    "medical machine",
    "monitor",
    "mouse",
    "sink",
    "sofa",
    "speaker",
    "suitcase",
    "table",
    "tablet",
    "tissue box",
    "toilet paper dispenser",
    "trash can",
    "tripod",
    "whiteboard",
}

DEFAULT_DETECTOR_CLASS_LIST_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "class_lists" / "detector_coverable.txt"
)


def canonical_detector_labels() -> list[str]:
    return sorted(DEFAULT_DETECTOR_COVERABLE_LABELS)


def read_detector_class_list(path: Path | str) -> list[str]:
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = normalize_text(line)
            if text:
                labels.append(text)
    return labels


def detector_class_list_mismatch(path: Path | str) -> dict[str, Any]:
    expected = canonical_detector_labels()
    found = read_detector_class_list(path)
    return {
        "expected": expected,
        "found": found,
        "missing": [label for label in expected if label not in found],
        "extra": [label for label in found if label not in expected],
        "order_matches": found == expected,
    }


def validate_detector_class_list(path: Path | str) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"detector class list not found: {path}")
    mismatch = detector_class_list_mismatch(path)
    if mismatch["missing"] or mismatch["extra"] or not mismatch["order_matches"]:
        raise ValueError(
            "Detector class list must exactly match "
            "DEFAULT_DETECTOR_COVERABLE_LABELS in spatial_memory_evaluation.common.labels. "
            f"path={path} missing={mismatch['missing']} extra={mismatch['extra']} "
            f"order_matches={mismatch['order_matches']}"
        )


def write_default_detector_class_list(path: Path | str = DEFAULT_DETECTOR_CLASS_LIST_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label in canonical_detector_labels():
            f.write(f"{label}\n")


def load_aliases(path: Path | str | None) -> dict[str, str]:
    aliases = dict(DEFAULT_LABEL_ALIASES)
    if path is None:
        return aliases
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"label aliases must be a JSON object: {path}")
    for key, value in loaded.items():
        aliases[normalize_text(str(key))] = normalize_text(str(value))
    return aliases


def normalize_label(label: Any, aliases: Mapping[str, str] | None = None) -> str:
    text = normalize_text(str(label or ""))
    alias_map = aliases or DEFAULT_LABEL_ALIASES
    return alias_map.get(text, text)


def normalize_text(value: str) -> str:
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value


def write_default_alias_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_LABEL_ALIASES, f, indent=2, sort_keys=True)
