from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from spatial_memory_evaluation import ObjectPrediction, RGBDSequence


DEFAULT_DUALMAP_ROOT = Path("/home/robin_wang/DualMap")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_MAP_DIR = Path(
    "/data/mondo-training-dataset/semantic_mapping/dualmap/"
    "scannetpp_036bce3393/map"
)
DEFAULT_CLASS_NAMES = DEFAULT_DUALMAP_ROOT / "config/class_list/gpt_indoor_general.txt"

logger = logging.getLogger(__name__)

QUERY_ALIASES = {
    "display": {"display", "monitor", "screen", "tv", "television"},
    "bottle": {"bottle", "shampoo bottle", "water bottle", "spray bottle"},
    "chair": {"chair", "desk chair", "office chair"},
}


@dataclass(frozen=True)
class DualMapAdapterConfig:
    dualmap_root: Path = DEFAULT_DUALMAP_ROOT
    map_dir: Path = DEFAULT_MAP_DIR
    scene_id: str = DEFAULT_SCENE_ID
    class_names_path: Path = DEFAULT_CLASS_NAMES
    top_k: int = 5
    use_clip: bool = False
    clip_model_name: str = "MobileCLIP-S2"
    clip_pretrained: str = "datacompdr"
    device: str = "cuda"
    allow_label_fallback: bool = True


class DualMapObjectMethod:
    """Expose a saved DualMap concrete map through the evaluation interface.

    DualMap's public query app is interactive and visual. This adapter reuses
    its saved ``map/*.pkl`` objects and implements the same object lookup path
    headlessly: class-name matching by default, or CLIP similarity when
    ``use_clip`` is enabled and the matching CLIP dependencies are installed.
    """

    def __init__(self, sequence: RGBDSequence, config: DualMapAdapterConfig) -> None:
        self.sequence = sequence
        self.config = config
        self._objects: Optional[List[dict[str, Any]]] = None
        self._class_names: Optional[dict[int, str]] = None
        self._clip_model = None
        self._clip_tokenizer = None

    def get_memory_text(self, question: str) -> str:
        raise NotImplementedError("DualMap adapter only supports get_object().")

    def get_object(self, query: str):
        self._ensure_loaded()
        assert self._objects is not None

        scored = self._score_objects(query)
        if scored and scored[0][1] == "label":
            scored = [item for item in scored if item[0] > 0.0]
        return [
            ObjectPrediction(
                label=obj["label"],
                score=float(score),
                object_id=str(obj["object_id"]),
                bbox_3d=obj.get("bbox_3d"),
                position_3d=obj.get("position_3d"),
                attributes={
                    "source": "dualmap",
                    "scene_id": self.config.scene_id,
                    "map_dir": str(self.config.map_dir),
                    "class_id": obj.get("class_id"),
                    "num_points": obj.get("num_points"),
                    "ranking": rank + 1,
                    "score_kind": score_kind,
                },
            )
            for rank, (score, score_kind, obj) in enumerate(scored[: self.config.top_k])
        ]

    def export_spatial_memory_db(self, output_db: str | Path) -> Path:
        """Write DualMap objects as a SpatialRAG-compatible sqlite table."""

        self._ensure_loaded()
        assert self._objects is not None

        output = Path(output_db)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()

        conn = sqlite3.connect(str(output))
        try:
            conn.execute(
                """
                CREATE TABLE spatial_memories (
                    memory_id TEXT PRIMARY KEY,
                    scene_id TEXT,
                    object_name TEXT,
                    snapshot_text TEXT,
                    pos_x REAL,
                    pos_y REAL,
                    pos_z REAL,
                    timestamp REAL,
                    confidence REAL,
                    observation_count INTEGER
                )
                """
            )
            for index, obj in enumerate(self._objects):
                position = obj.get("position_3d") or [0.0, 0.0, 0.0]
                snapshot = _snapshot_text(obj, self.config.scene_id)
                conn.execute(
                    """
                    INSERT INTO spatial_memories (
                        memory_id, scene_id, object_name, snapshot_text,
                        pos_x, pos_y, pos_z, timestamp, confidence,
                        observation_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(obj.get("object_id") or index),
                        self.config.scene_id,
                        str(obj["label"]),
                        snapshot,
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(index),
                        None,
                        None,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return output

    def _ensure_loaded(self) -> None:
        if self._objects is not None:
            return

        _add_dualmap_to_path(self.config.dualmap_root)
        if not self.config.map_dir.exists():
            raise FileNotFoundError(
                "DualMap map directory does not exist: "
                f"{self.config.map_dir}\n"
                "Generate it first with DualMap runner_dataset, or pass a "
                "different map_dir in method kwargs."
            )

        self._class_names = _load_class_names(self.config.class_names_path)
        self._objects = _load_dualmap_objects(self.config.map_dir, self._class_names)
        if not self._objects:
            raise ValueError(f"No DualMap object .pkl files found in {self.config.map_dir}")

        if self.config.use_clip:
            self._init_clip()

    def _score_objects(self, query: str) -> list[tuple[float, str, dict[str, Any]]]:
        assert self._objects is not None

        if self.config.use_clip and self._clip_model is not None:
            try:
                scored = self._score_with_clip(query)
                if scored:
                    return scored
            except Exception as exc:
                if not self.config.allow_label_fallback:
                    raise
                logger.warning("DualMap CLIP query failed; falling back to labels: %s", exc)

        return self._score_with_labels(query)

    def _score_with_labels(self, query: str) -> list[tuple[float, str, dict[str, Any]]]:
        assert self._objects is not None
        query_norm = _normalize_text(query)
        query_variants = _query_variants(query_norm)
        query_tokens = set(query_norm.split())
        scored = []
        for obj in self._objects:
            label_norm = _normalize_text(obj["label"])
            label_tokens = set(label_norm.split())
            if label_norm in query_variants:
                score = 1.0
            elif any(variant and variant in label_norm for variant in query_variants):
                score = 0.9
            elif label_norm and label_norm in query_norm:
                score = 0.85
            else:
                overlap = len(query_tokens & label_tokens)
                score = 0.25 + 0.15 * overlap if overlap else 0.0
            scored.append((float(score), "label", obj))
        scored.sort(key=lambda item: (-item[0], item[2]["label"], item[2]["object_id"]))
        return scored

    def _score_with_clip(self, query: str) -> list[tuple[float, str, dict[str, Any]]]:
        assert self._objects is not None
        import torch
        import torch.nn.functional as F

        text = self._clip_tokenizer([query]).to(self.config.device)
        with torch.no_grad():
            text_ft = self._clip_model.encode_text(text)
            text_ft = text_ft / text_ft.norm(dim=-1, keepdim=True)

        valid_objects = [
            obj
            for obj in self._objects
            if obj.get("clip_ft") is not None and np.asarray(obj["clip_ft"]).size
        ]
        values = [torch.from_numpy(obj["clip_ft"]).float() for obj in valid_objects]
        if not values:
            return []
        map_ft = torch.stack(values, dim=0).to(self.config.device)
        cos_sim = F.cosine_similarity(text_ft, map_ft, dim=-1).detach().cpu().numpy()
        scored = [
            (float(score), "clip", obj)
            for score, obj in zip(cos_sim.tolist(), valid_objects)
        ]
        scored.sort(key=lambda item: (-item[0], item[2]["label"], item[2]["object_id"]))
        return scored

    def _init_clip(self) -> None:
        import open_clip

        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
        )
        self._clip_model = self._clip_model.to(self.config.device)
        self._clip_model.eval()
        if "MobileCLIP" in self.config.clip_model_name:
            from mobileclip.modules.common.mobileone import reparameterize_model

            self._clip_model = reparameterize_model(self._clip_model)
        self._clip_tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)


def create_method(sequence: RGBDSequence, **kwargs: Any) -> DualMapObjectMethod:
    config = DualMapAdapterConfig(
        dualmap_root=Path(kwargs.get("dualmap_root", DEFAULT_DUALMAP_ROOT)),
        map_dir=Path(kwargs.get("map_dir", DEFAULT_MAP_DIR)),
        scene_id=str(kwargs.get("scene_id", DEFAULT_SCENE_ID)),
        class_names_path=Path(kwargs.get("class_names_path", DEFAULT_CLASS_NAMES)),
        top_k=int(kwargs.get("top_k", 5)),
        use_clip=bool(kwargs.get("use_clip", False)),
        clip_model_name=str(kwargs.get("clip_model_name", "MobileCLIP-S2")),
        clip_pretrained=str(kwargs.get("clip_pretrained", "datacompdr")),
        device=str(kwargs.get("device", "cuda")),
        allow_label_fallback=bool(kwargs.get("allow_label_fallback", True)),
    )
    return DualMapObjectMethod(sequence=sequence, config=config)


def _add_dualmap_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_class_names(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"DualMap class names file does not exist: {path}")
    with path.open("r") as f:
        names = [line.strip() for line in f if line.strip()]
    return {idx: name for idx, name in enumerate(names)}


def _load_dualmap_objects(map_dir: Path, class_names: dict[int, str]) -> list[dict[str, Any]]:
    from utils.object import BaseObject

    objects = []
    for path in sorted(map_dir.glob("*.pkl")):
        obj = BaseObject.load_from_disk(str(path))
        record = _object_record(obj, path, class_names)
        if record is not None:
            objects.append(record)
    return objects


def _object_record(obj: Any, path: Path, class_names: dict[int, str]) -> dict[str, Any] | None:
    class_id = getattr(obj, "class_id", None)
    label = class_names.get(int(class_id), f"class_{class_id}") if class_id is not None else "unknown"
    pcd = getattr(obj, "pcd", None)
    if pcd is None:
        return None

    points = np.asarray(pcd.points)
    if points.size == 0:
        return None
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = np.asarray(bbox.min_bound, dtype=float)
    max_bound = np.asarray(bbox.max_bound, dtype=float)
    center = np.asarray(bbox.get_center(), dtype=float)
    clip_ft = np.asarray(getattr(obj, "clip_ft", np.empty(0)), dtype=np.float32)

    return {
        "object_id": str(getattr(obj, "uid", path.stem)),
        "label": label,
        "class_id": int(class_id) if class_id is not None else None,
        "bbox_3d": [
            float(min_bound[0]),
            float(min_bound[1]),
            float(min_bound[2]),
            float(max_bound[0]),
            float(max_bound[1]),
            float(max_bound[2]),
        ],
        "position_3d": [float(center[0]), float(center[1]), float(center[2])],
        "num_points": int(len(points)),
        "clip_ft": clip_ft,
        "path": str(path),
    }


def _snapshot_text(obj: dict[str, Any], scene_id: str) -> str:
    bbox = obj.get("bbox_3d") or []
    pos = obj.get("position_3d") or [0.0, 0.0, 0.0]
    return "\n".join(
        [
            f"Object: {obj['label']}",
            f"Scene: {scene_id}",
            f"Source: DualMap concrete map",
            f"Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
            f"BBox: {json.dumps(bbox)}",
            f"Object ID: {obj.get('object_id')}",
        ]
    )


def _normalize_text(text: str) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split()
    )


def _query_variants(query_norm: str) -> set[str]:
    variants = {query_norm}
    for canonical, aliases in QUERY_ALIASES.items():
        if query_norm == canonical or query_norm in aliases:
            variants.update(_normalize_text(alias) for alias in aliases)
    return {variant for variant in variants if variant}
