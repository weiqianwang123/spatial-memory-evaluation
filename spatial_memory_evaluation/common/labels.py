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
    # --- ScanNet++ -> ScanNet200 surface-form reconciliation (2026-06-24) ---
    # Map common ScanNet++ raw labels onto their ScanNet200 equivalents so they
    # land in the detector_coverable split instead of all_annotated-only. Targets
    # verified present in scannet200.txt.
    "books": "book",
    "notebook": "book",
    "papers": "paper",
    "folders": "file cabinet",
    "filer organizer": "file cabinet",
    "heater": "radiator",
    "sofa": "couch",
    "office visitor chair": "chair",
    "office chair": "chair",
    "storage cabinet": "cabinet",
    "kitchen cabinet": "kitchen cabinet",
    "window frame": "window",
    "window blind": "blinds",
    "window sill": "windowsill",
    "blind rail": "blinds",
    "door frame": "door",
    "suspended ceiling": "ceiling",
    "power socket": "power strip",
    "socket": "power strip",
    "mug": "cup",
    "jar": "bottle",
    "spray bottle": "bottle",
    "cardboard box": "box",
    "cardboards": "box",
    "storage rack": "shelf",
    "rolling cart": "cart",
    "tap": "sink",
}

# ScanNet++ annotation edit-tags that are not real objects; drop from queries.
SCANNETPP_NON_OBJECT_LABELS: set[str] = {"remove", "split", "objects", "object", ""}

# Shared detector class list — the single OV prompt/eval vocabulary used by Track 1
# query generation AND every detector-based method's prompt (set_classes), across
# all scenes. 2026-06-24: switched from the 37-label hand-picked list to the
# standard **ScanNet200** vocabulary (200 labels), which covers ~95% of the Track 2
# ScanEnts3D referring targets (the 37-list covered 44%) and is what DualMap /
# ConceptGraphs project onto natively. The list is loaded from the asset file so it
# stays the single source of truth; to revert, point this at the old list.
_CLASS_LIST_DIR = Path(__file__).resolve().parents[1] / "assets" / "class_lists"
DEFAULT_DETECTOR_CLASS_LIST_PATH = _CLASS_LIST_DIR / "scannet200.txt"
# Legacy 37-label list kept for reference / coarse-eval ablation.
LEGACY_DETECTOR_COVERABLE_LIST_PATH = _CLASS_LIST_DIR / "detector_coverable.txt"


def _load_label_set(path: Path) -> set[str]:
    labels: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            text = re.sub(r"\s+", " ", str(line).strip().lower())
            if text:
                labels.add(text)
    return labels


DEFAULT_DETECTOR_COVERABLE_LABELS: set[str] = _load_label_set(DEFAULT_DETECTOR_CLASS_LIST_PATH)


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
