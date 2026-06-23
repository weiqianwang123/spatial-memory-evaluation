from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from spatial_memory_evaluation.common.jsonl import read_jsonl


class NativeToolExecutor:
    """Execute declared method-native retrieval tools over package-local memory.

    This module intentionally avoids evaluator GT and benchmark answers. It is a
    first thin runtime layer for the new tool-LLM evaluation path: tools read
    raw/native artifacts when they are packaged, and otherwise report that the
    method package needs a native artifact rather than silently falling back to a
    fixed-API conversion view.
    """

    def __init__(self, package_dir: Path, manifest: Mapping[str, Any]):
        self.package_dir = Path(package_dir)
        method = manifest.get("method") if isinstance(manifest.get("method"), Mapping) else {}
        self.method_name = str(method.get("name") or "unknown").lower()
        self.method_family = str(method.get("family") or "unknown").lower()

    def tool_specs(self) -> list[dict[str, Any]]:
        if self.method_name == "daaam":
            return [
                {
                    "name": "get_matching_subjects",
                    "description": (
                        "DAAAM-native semantic subject retrieval over package-local "
                        "native DAAAM extraction/DSG artifacts. Use this for object "
                        "lookup queries such as 'monitor' or 'wooden chair'."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                            "include_background": {"type": "boolean", "default": True},
                        },
                        "required": ["description"],
                    },
                }
            ]
        if self.method_name in {"remembr", "remembr_captions"} or self.method_family in {
            "caption_memory",
            "caption_control",
        }:
            return [
                {
                    "name": "retrieve_from_text",
                    "description": (
                        "ReMEmbR-native text retrieval over caption/time/robot-pose "
                        "memory. Returns caption memories, not object ids."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "retrieve_from_position",
                    "description": (
                        "ReMEmbR-native position retrieval over robot-pose memory. "
                        "Input is an [x, y, z] robot/world position."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["position"],
                    },
                },
            ]
        if self.method_name in {"claws", "spatialrag", "spatial_rag"}:
            return [
                {
                    "name": "query_spatial_memory",
                    "description": (
                        "ClawS/SpatialRAG native memory retrieval. Requires the "
                        "native sqlite/sqlite-vec DB to be packaged under memory/native."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "default": 5},
                        },
                        "required": ["query"],
                    },
                }
            ]
        return []

    def execute(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "get_matching_subjects" and self.method_name == "daaam":
            return self._daaam_get_matching_subjects(arguments)
        if name == "retrieve_from_text":
            return self._remembr_retrieve_from_text(arguments)
        if name == "retrieve_from_position":
            return self._remembr_retrieve_from_position(arguments)
        if name == "query_spatial_memory":
            return self._claws_query_spatial_memory(arguments)
        return {"status": "error", "message": f"unknown or unavailable native tool: {name}"}

    def _daaam_get_matching_subjects(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        native_dir = self.package_dir / "memory" / "native"
        dsg_path = _first_existing(
            native_dir / "dsg_updated.json",
            native_dir / "clustered_dsg.json",
            native_dir / "dsg.json",
        )
        label_names = _daaam_label_names(native_dir / "corrections.yaml")
        background_objects = _daaam_background_objects(native_dir / "background_objects.yaml")
        description = str(arguments.get("description") or "")
        top_k = _positive_int(arguments.get("top_k"), default=5)
        include_background = bool(arguments.get("include_background", True))
        candidates = _daaam_correction_object_candidates(
            label_names=label_names,
            object_positions_path=native_dir / "object_positions.json",
        )
        if not candidates and dsg_path is not None:
            dsg = _read_json(dsg_path)
            candidates = _daaam_dsg_object_candidates(dsg=dsg, label_names=label_names, source_path=dsg_path)
        if not candidates:
            return {
                "status": "error",
                "message": (
                    "DAAAM raw memory did not contain usable corrections/object_positions "
                    "or DSG object nodes under memory/native."
                ),
            }
        if include_background:
            candidates.extend(background_objects)
        scored = []
        for row in candidates:
            if not isinstance(row, Mapping):
                continue
            text = " ".join(
                str(row.get(key) or "")
                for key in ("object_id", "raw_label", "semantic_label")
            )
            score = _lexical_score(description, text)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, row in scored[:top_k]:
            results.append(
                {
                    "object_id": row.get("object_id"),
                    "raw_label": row.get("raw_label"),
                    "position_3d": row.get("position_3d"),
                    "bbox_3d": row.get("bbox_3d"),
                    "is_background": row.get("is_background"),
                    "semantic_label": row.get("semantic_label"),
                    "score": score,
                    "source": row.get("source"),
                }
            )
        return {"status": "ok", "tool": "get_matching_subjects", "results": results}

    def _remembr_retrieve_from_text(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        captions_path = self.package_dir / "memory" / "captions.jsonl"
        if not captions_path.exists():
            return {
                "status": "error",
                "message": "ReMEmbR captions not found at memory/captions.jsonl",
            }
        query = str(arguments.get("query") or "")
        top_k = _positive_int(arguments.get("top_k"), default=5)
        scored = []
        for row in read_jsonl(captions_path):
            text = str(row.get("caption") or "")
            scored.append((_lexical_score(query, text), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, row in scored[:top_k]:
            results.append(
                {
                    "caption_id": row.get("caption_id"),
                    "caption": row.get("caption"),
                    "time": row.get("time"),
                    "position": row.get("position"),
                    "theta": row.get("theta"),
                    "score": score,
                    "source": "memory/captions.jsonl",
                }
            )
        return {"status": "ok", "tool": "retrieve_from_text", "results": results}

    def _remembr_retrieve_from_position(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        captions_path = self.package_dir / "memory" / "captions.jsonl"
        if not captions_path.exists():
            return {
                "status": "error",
                "message": "ReMEmbR captions not found at memory/captions.jsonl",
            }
        position = arguments.get("position")
        if not isinstance(position, list) or len(position) != 3:
            return {"status": "error", "message": "position must be a 3-number list"}
        query_position = [float(value) for value in position]
        top_k = _positive_int(arguments.get("top_k"), default=5)
        scored = []
        for row in read_jsonl(captions_path):
            row_position = row.get("position")
            if not isinstance(row_position, list) or len(row_position) != 3:
                continue
            distance = math.dist(query_position, [float(value) for value in row_position])
            scored.append((distance, row))
        scored.sort(key=lambda item: item[0])
        results = []
        for distance, row in scored[:top_k]:
            results.append(
                {
                    "caption_id": row.get("caption_id"),
                    "caption": row.get("caption"),
                    "time": row.get("time"),
                    "position": row.get("position"),
                    "theta": row.get("theta"),
                    "distance": distance,
                    "source": "memory/captions.jsonl",
                }
            )
        return {"status": "ok", "tool": "retrieve_from_position", "results": results}

    def _claws_query_spatial_memory(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        native_dir = self.package_dir / "memory" / "native"
        db_candidates = sorted(native_dir.glob("*.db")) if native_dir.exists() else []
        if not db_candidates:
            return {
                "status": "error",
                "message": (
                    "ClawS/SpatialRAG native DB is not packaged under memory/native. "
                    "Rebuild the package with the native sqlite/sqlite-vec DB copied "
                    "into memory/native before running tool_llm eval."
                ),
            }
        return {
            "status": "error",
            "message": (
                "ClawS native DB found, but native query runtime is not wired yet. "
                "Next step: call ClawS root repo memory_loader / SpatialStorage over "
                f"{db_candidates[0]}."
            ),
        }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return _read_daaam_yaml_subset(path)
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return value if isinstance(value, dict) else {}


def _read_daaam_yaml_subset(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.name == "corrections.yaml":
        return {"label_names": _parse_daaam_correction_label_names(text)}
    if path.name == "background_objects.yaml":
        return {"objects": _parse_daaam_background_yaml_objects(text)}
    return {}


def _parse_daaam_correction_label_names(text: str) -> list[dict[str, Any]]:
    rows = []
    current: dict[str, Any] | None = None
    collecting_name = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("- "):
            if current is not None:
                rows.append(current)
            current = {}
            collecting_name = False
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("label:"):
            current["label"] = stripped.split(":", 1)[1].strip()
            collecting_name = False
        elif stripped.startswith("name:"):
            current["name"] = stripped.split(":", 1)[1].strip()
            collecting_name = True
        elif collecting_name and line.startswith("    ") and stripped and not _looks_like_yaml_key(stripped):
            current["name"] = f"{current.get('name', '')} {stripped}".strip()
        elif stripped:
            collecting_name = False
    if current is not None:
        rows.append(current)
    return rows


def _parse_daaam_background_yaml_objects(text: str) -> list[dict[str, Any]]:
    rows = []
    current: dict[str, Any] | None = None
    collecting_label = False
    collecting_position = False
    position_values: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("- "):
            if current is not None:
                if position_values:
                    current["position_world"] = position_values[:3]
                rows.append(current)
            current = {}
            collecting_label = False
            collecting_position = False
            position_values = []
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("semantic_id:"):
            current["semantic_id"] = stripped.split(":", 1)[1].strip()
            collecting_label = False
            collecting_position = False
        elif stripped.startswith("label:"):
            current["label"] = stripped.split(":", 1)[1].strip()
            collecting_label = True
            collecting_position = False
        elif stripped.startswith("position_world:"):
            collecting_label = False
            collecting_position = True
            position_values = []
        elif collecting_position and stripped.startswith("- "):
            try:
                position_values.append(float(stripped[2:].strip()))
            except ValueError:
                pass
        elif collecting_label and line.startswith("  ") and stripped and not _looks_like_yaml_key(stripped):
            current["label"] = f"{current.get('label', '')} {stripped}".strip()
        elif stripped:
            collecting_label = False
            collecting_position = False
    if current is not None:
        if position_values:
            current["position_world"] = position_values[:3]
        rows.append(current)
    return rows


def _looks_like_yaml_key(text: str) -> bool:
    head = text.split(":", 1)[0]
    return ":" in text and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", head))


def _daaam_label_names(path: Path) -> dict[int, str]:
    data = _read_yaml(path)
    result: dict[int, str] = {}
    for row in data.get("label_names", []) if isinstance(data.get("label_names"), list) else []:
        if not isinstance(row, Mapping):
            continue
        label = row.get("label")
        name = row.get("name")
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        if isinstance(name, str) and name.strip():
            result[label_int] = name.strip()
    return result


def _daaam_background_objects(path: Path) -> list[dict[str, Any]]:
    data = _read_yaml(path)
    rows = data.get("objects")
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        semantic_id = row.get("semantic_id")
        label = row.get("label")
        result.append(
            {
                "object_id": f"daaam_background_{semantic_id}",
                "raw_label": label,
                "position_3d": row.get("position_world"),
                "bbox_3d": None,
                "is_background": True,
                "semantic_label": semantic_id,
                "source": "memory/native/background_objects.yaml",
            }
        )
    return result


def _daaam_correction_object_candidates(
    *,
    label_names: Mapping[int, str],
    object_positions_path: Path,
) -> list[dict[str, Any]]:
    if not object_positions_path.exists():
        return []
    data = _read_json(object_positions_path)
    result = []
    for semantic_label, raw_label in label_names.items():
        positions = data.get(str(semantic_label))
        center = _mean_world_position(positions)
        result.append(
            {
                "object_id": f"daaam_semantic_{semantic_label}",
                "raw_label": raw_label,
                "position_3d": center,
                "bbox_3d": None,
                "is_background": False,
                "semantic_label": semantic_label,
                "source": "memory/native/corrections.yaml + memory/native/object_positions.json",
            }
        )
    return result


def _mean_world_position(rows: Any) -> list[float] | None:
    if not isinstance(rows, list):
        return None
    positions = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        position = row.get("position_world")
        if not isinstance(position, list) or len(position) != 3:
            continue
        try:
            positions.append([float(value) for value in position])
        except (TypeError, ValueError):
            continue
    if not positions:
        return None
    return [
        sum(position[axis] for position in positions) / len(positions)
        for axis in range(3)
    ]


def _daaam_dsg_object_candidates(
    *,
    dsg: Mapping[str, Any],
    label_names: Mapping[int, str],
    source_path: Path,
) -> list[dict[str, Any]]:
    nodes = dsg.get("nodes")
    if not isinstance(nodes, list):
        return []
    result = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        attrs = node.get("attributes")
        if not isinstance(attrs, Mapping):
            continue
        semantic_label = attrs.get("semantic_label")
        try:
            semantic_label_int = int(semantic_label)
        except (TypeError, ValueError):
            semantic_label_int = None
        raw_label = label_names.get(semantic_label_int) if semantic_label_int is not None else None
        if not raw_label:
            raw_label = str(attrs.get("name") or semantic_label or "")
        bbox = _daaam_bbox(attrs.get("bounding_box"))
        result.append(
            {
                "object_id": f"daaam_dsg_{node.get('id')}",
                "raw_label": raw_label,
                "position_3d": attrs.get("position") or _bbox_center(bbox),
                "bbox_3d": bbox,
                "is_background": False,
                "semantic_label": semantic_label,
                "source": f"memory/native/{source_path.name}",
            }
        )
    return result


def _daaam_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, Mapping):
        return None
    center = value.get("world_P_center")
    dimensions = value.get("dimensions")
    if not isinstance(center, list) or len(center) != 3:
        return None
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        return None
    try:
        c = [float(item) for item in center]
        d = [float(item) for item in dimensions]
    except (TypeError, ValueError):
        return None
    return [
        c[0] - d[0] / 2.0,
        c[1] - d[1] / 2.0,
        c[2] - d[2] / 2.0,
        c[0] + d[0] / 2.0,
        c[1] + d[1] / 2.0,
        c[2] + d[2] / 2.0,
    ]


def _bbox_center(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 6:
        return None
    return [
        (float(value[0]) + float(value[3])) / 2.0,
        (float(value[1]) + float(value[4])) / 2.0,
        (float(value[2]) + float(value[5])) / 2.0,
    ]


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _lexical_score(query: str, text: str) -> float:
    query_terms = set(_tokens(query))
    text_terms = set(_tokens(text))
    if not query_terms:
        return 0.0
    overlap = len(query_terms & text_terms)
    return overlap / max(1, len(query_terms))


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
