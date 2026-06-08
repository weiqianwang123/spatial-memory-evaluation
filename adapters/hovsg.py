from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from spatial_memory_evaluation import ObjectPrediction, RGBDSequence


DEFAULT_HOVSG_ROOT = Path("/home/robin_wang/HOV-SG")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_HOVSG_RESULT_PATH = Path(
    "/data/mondo-training-dataset/semantic_mapping/hovsg/"
    "scannetpp_036bce3393/scannet"
)
DEFAULT_CLASS_NAMES = Path("/home/robin_wang/DualMap/config/class_list/gpt_indoor_general.txt")

logger = logging.getLogger(__name__)

QUERY_ALIASES = {
    "display": {"display", "monitor", "screen", "tv", "television"},
    "bottle": {"bottle", "shampoo bottle", "water bottle", "spray bottle"},
    "chair": {"chair", "desk chair", "office chair"},
}


@dataclass(frozen=True)
class HOVSGAdapterConfig:
    hovsg_root: Path = DEFAULT_HOVSG_ROOT
    result_path: Path = DEFAULT_HOVSG_RESULT_PATH
    scene_id: str = DEFAULT_SCENE_ID
    class_names_path: Path = DEFAULT_CLASS_NAMES
    top_k: int = 5
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    device: str = "cuda"
    batch_size: int = 64
    normalize_features: bool = True
    classify_objects: bool = True
    templates: Sequence[str] = field(
        default_factory=lambda: ("{}", "There is the {} in the scene.")
    )
    allow_label_fallback: bool = True


class HOVSGObjectMethod:
    """Expose a saved HOV-SG object feature map through the eval interface.

    HOV-SG's ScanNet/Replica semantic-segmentation path saves object point
    clouds in ``objects/pcd_i.ply`` and corresponding open-vocabulary features
    in ``mask_feats.pt``. This adapter loads that saved map, classifies each
    object into the configured class list with the same CLIP text encoder, and
    answers ``get_object`` by ranking object features against a text query.
    """

    def __init__(self, sequence: RGBDSequence, config: HOVSGAdapterConfig) -> None:
        self.sequence = sequence
        self.config = config
        self._objects: Optional[List[dict[str, Any]]] = None
        self._class_names: Optional[List[str]] = None
        self._clip_model = None
        self._device = "cpu"

    def get_memory_text(self, question: str) -> str:
        raise NotImplementedError("HOV-SG adapter only supports get_object().")

    def get_object(self, query: str):
        self._ensure_loaded()
        assert self._objects is not None

        try:
            scored = self._score_with_clip(query)
        except Exception as exc:
            if not self.config.allow_label_fallback:
                raise
            logger.warning("HOV-SG CLIP query failed; falling back to labels: %s", exc)
            scored = self._score_with_labels(query)

        return [
            ObjectPrediction(
                label=obj["label"],
                score=float(score),
                object_id=str(obj["object_id"]),
                bbox_3d=obj.get("bbox_3d"),
                position_3d=obj.get("position_3d"),
                attributes={
                    "source": "hovsg",
                    "scene_id": self.config.scene_id,
                    "result_path": str(self.config.result_path),
                    "num_points": obj.get("num_points"),
                    "ranking": rank + 1,
                    "score_kind": score_kind,
                    "label_score": obj.get("label_score"),
                    "feature_index": obj.get("feature_index"),
                },
            )
            for rank, (score, score_kind, obj) in enumerate(scored[: self.config.top_k])
        ]

    def export_spatial_memory_db(self, output_db: str | Path) -> Path:
        """Write all HOV-SG objects as a SpatialRAG-compatible sqlite table."""

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
                        _snapshot_text(obj, self.config.scene_id),
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        float(index),
                        obj.get("label_score"),
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

        _add_hovsg_to_path(self.config.hovsg_root)
        result_path = self.config.result_path.expanduser()
        if not result_path.exists():
            raise FileNotFoundError(
                "HOV-SG result directory does not exist: "
                f"{result_path}\n"
                "Expected a precomputed HOV-SG feature map containing "
                "mask_feats.pt and objects/pcd_*.ply. Generate it with "
                "HOV-SG application/semantic_segmentation.py or pass "
                "--hovsg-result-path to the evaluation script."
            )
        if not (result_path / "mask_feats.pt").exists():
            raise FileNotFoundError(f"Missing HOV-SG feature file: {result_path / 'mask_feats.pt'}")
        if not (result_path / "objects").exists():
            raise FileNotFoundError(f"Missing HOV-SG object directory: {result_path / 'objects'}")

        features = _load_mask_features(result_path / "mask_feats.pt", self.config.normalize_features)
        self._objects = _load_hovsg_objects(result_path / "objects", features)
        if not self._objects:
            raise ValueError(f"No usable HOV-SG object .ply files found in {result_path / 'objects'}")

        self._class_names = _load_class_names(self.config.class_names_path)
        if self.config.classify_objects:
            self._init_clip()
            self._classify_objects()

    def _score_with_clip(self, query: str) -> list[tuple[float, str, dict[str, Any]]]:
        self._init_clip()
        assert self._objects is not None

        query_feat = self._encode_text([query])[0]
        object_feats = np.stack([obj["clip_ft"] for obj in self._objects], axis=0)
        scores = object_feats @ query_feat
        scored = [
            (float(score), "clip", obj)
            for score, obj in zip(scores.tolist(), self._objects)
        ]
        scored.sort(key=lambda item: (-item[0], item[2]["label"], item[2]["object_id"]))
        return scored

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
            if score > 0.0:
                scored.append((float(score), "label", obj))
        scored.sort(key=lambda item: (-item[0], item[2]["label"], item[2]["object_id"]))
        return scored

    def _classify_objects(self) -> None:
        assert self._objects is not None
        assert self._class_names is not None

        class_feats = self._encode_text(self._class_names)
        object_feats = np.stack([obj["clip_ft"] for obj in self._objects], axis=0)
        scores = object_feats @ class_feats.T
        best = np.argmax(scores, axis=1)
        for row_index, (obj, class_index) in enumerate(zip(self._objects, best.tolist())):
            obj["label"] = self._class_names[class_index]
            obj["label_score"] = float(scores[row_index, class_index])

    def _init_clip(self) -> None:
        if self._clip_model is not None:
            return

        import torch
        import open_clip

        self._device = self.config.device
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for HOV-SG adapter, but CUDA is unavailable; using CPU.")
            self._device = "cpu"

        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
            device=self._device,
        )
        self._clip_model.eval()

    def _encode_text(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import open_clip

        self._init_clip()
        assert self._clip_model is not None

        prompts = [template.format(text) for text in texts for template in self.config.templates]
        text_features = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.config.batch_size):
                batch = prompts[start : start + self.config.batch_size]
                tokens = open_clip.tokenize(batch).to(self._device)
                feats = self._clip_model.encode_text(tokens).float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                text_features.append(feats.detach().cpu())
        features = torch.cat(text_features, dim=0).numpy().astype(np.float32)
        features = features.reshape(len(texts), len(self.config.templates), -1).mean(axis=1)
        features = _normalize_rows(features)

        object_dim = None
        if self._objects:
            object_dim = int(np.asarray(self._objects[0]["clip_ft"]).shape[-1])
        if object_dim is not None and features.shape[-1] != object_dim:
            raise ValueError(
                "HOV-SG text-feature dimension does not match saved mask_feats: "
                f"text={features.shape[-1]}, mask={object_dim}. "
                "Set clip_model_name/clip_pretrained to the exact CLIP model "
                "used when generating the HOV-SG map."
            )
        return features


def create_method(sequence: RGBDSequence, **kwargs: Any) -> HOVSGObjectMethod:
    templates = kwargs.get("templates")
    if templates is None:
        templates = ("{}", "There is the {} in the scene.")

    config = HOVSGAdapterConfig(
        hovsg_root=Path(kwargs.get("hovsg_root", DEFAULT_HOVSG_ROOT)),
        result_path=Path(kwargs.get("result_path", DEFAULT_HOVSG_RESULT_PATH)),
        scene_id=str(kwargs.get("scene_id", DEFAULT_SCENE_ID)),
        class_names_path=Path(kwargs.get("class_names_path", DEFAULT_CLASS_NAMES)),
        top_k=int(kwargs.get("top_k", 5)),
        clip_model_name=str(kwargs.get("clip_model_name", "ViT-B-32")),
        clip_pretrained=str(kwargs.get("clip_pretrained", "laion2b_s34b_b79k")),
        device=str(kwargs.get("device", "cuda")),
        batch_size=int(kwargs.get("batch_size", 64)),
        normalize_features=bool(kwargs.get("normalize_features", True)),
        classify_objects=bool(kwargs.get("classify_objects", True)),
        templates=tuple(str(item) for item in templates),
        allow_label_fallback=bool(kwargs.get("allow_label_fallback", True)),
    )
    return HOVSGObjectMethod(sequence=sequence, config=config)


def _add_hovsg_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Class names file does not exist: {path}")
    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def _load_mask_features(path: Path, normalize: bool) -> np.ndarray:
    import torch

    loaded = torch.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        features = loaded.detach().cpu().float().numpy()
    else:
        features = np.asarray(loaded, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D HOV-SG mask features in {path}, got {features.shape}")
    features = features.astype(np.float32, copy=False)
    if normalize:
        features = _normalize_rows(features)
    return features


def _load_hovsg_objects(objects_dir: Path, features: np.ndarray) -> list[dict[str, Any]]:
    objects = []
    paths = sorted(objects_dir.glob("*.ply"), key=_natural_key)
    for ordinal, path in enumerate(paths):
        feature_index = _feature_index(path, ordinal)
        if feature_index >= len(features):
            logger.warning(
                "Skipping %s: feature index %s exceeds mask_feats rows %s",
                path,
                feature_index,
                len(features),
            )
            continue
        points = _read_ply_points(path)
        if points.size == 0:
            continue
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        center = (min_bound + max_bound) / 2.0
        objects.append(
            {
                "object_id": path.stem,
                "label": "object",
                "label_score": None,
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
                "clip_ft": features[feature_index],
                "feature_index": int(feature_index),
                "path": str(path),
            }
        )
    return objects


def _read_ply_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]
    points = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    finite = np.isfinite(points).all(axis=1)
    return points[finite]


def _feature_index(path: Path, ordinal: int) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return ordinal


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (values / norms).astype(np.float32)


def _snapshot_text(obj: dict[str, Any], scene_id: str) -> str:
    bbox = obj.get("bbox_3d") or []
    pos = obj.get("position_3d") or [0.0, 0.0, 0.0]
    return "\n".join(
        [
            f"Object: {obj['label']}",
            f"Scene: {scene_id}",
            "Source: HOV-SG object feature map",
            f"Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
            f"BBox: {json.dumps(bbox)}",
            f"Object ID: {obj.get('object_id')}",
            f"Label score: {obj.get('label_score')}",
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
