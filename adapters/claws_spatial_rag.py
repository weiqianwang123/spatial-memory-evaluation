from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np

from spatial_memory_evaluation import ObjectPrediction, RGBDSequence


DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_MEMORY_DB = (
    DEFAULT_CLAWS_ROOT / "outputs" / "scannetpp_memory_036bce3393_ollama_vlm.db"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClawSAdapterConfig:
    spatial_rag_root: Path = DEFAULT_CLAWS_ROOT
    memory_db: Path = DEFAULT_MEMORY_DB
    scene_id: str = DEFAULT_SCENE_ID
    build_from_sequence: bool = False
    overwrite_memory_db: bool = False
    embedding_provider: str = "mock"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_endpoint: str = "http://localhost:11434"
    embedding_dim: int = 1024
    top_k: int = 5
    object_limit: int = 500
    max_build_frames: Optional[int] = None
    build_frame_stride: int = 1
    yolo_model: Optional[str] = None
    track_confidence: float = 0.45
    min_store_confidence: float = 0.45
    min_confirm_frames: int = 2
    fusion_radius_m: float = 0.5
    semantic_similarity_threshold: float = 0.8
    lexical_fallback: bool = True


class ClawSSpatialRAGMethod:
    """Adapter that exposes ClawS SpatialRAG through the evaluation interface."""

    def __init__(self, sequence: RGBDSequence, config: ClawSAdapterConfig) -> None:
        self.sequence = sequence
        self.config = config
        self._ready = False
        self._service = None
        self._embedder = None
        self._objects: Optional[List[dict[str, Any]]] = None

    def get_memory_text(self, question: str) -> str:
        """Return retrieved SpatialRAG memory context for an OpenEQA question."""

        self._ensure_ready()
        if self._objects is not None:
            hits = self._lexical_query_rows(question, top_k=self.config.top_k)
        else:
            hits = self._run_async(self._query_memories(question, top_k=self.config.top_k))
        if not hits:
            return "No relevant spatial memories found."
        return _format_memory_context(question, hits)

    def get_object(self, query: str):
        """Return objects from the current SpatialRAG memory scene."""

        self._ensure_ready()
        if self._objects is not None:
            objects = self._matching_object_rows(query)
        else:
            objects = self._run_async(self._get_matching_objects(query))
        return [
            ObjectPrediction(
                label=obj["label"],
                score=float(obj.get("score", 1.0)),
                object_id=str(obj["id"]),
                position_3d=[
                    float(obj["pos_x"]),
                    float(obj["pos_y"]),
                    float(obj["pos_z"]),
                ],
                attributes={
                    "snapshot_text": obj["snapshot_text"],
                    "scene_id": self.config.scene_id,
                    "timestamp": obj.get("timestamp"),
                    "source": "claws_spatial_rag",
                },
            )
            for obj in objects
        ]

    def close(self) -> None:
        if self._service is not None:
            self._run_async(self._service.close())
            self._service = None
            self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        _add_repo_to_path(self.config.spatial_rag_root)
        if self.config.build_from_sequence:
            self.sequence.require_depth()
            self.sequence.require_poses()
            self._run_async(self._build_memory_from_sequence())

        if self.config.memory_db.exists():
            self._objects = _load_memory_rows(self.config.memory_db)
        else:
            self._service = self._make_service()
        self._ready = True

    def _make_service(self):
        from spatial_rag.clawspine_adapter import SpatialRAGService, SpatialRAGServiceConfig

        return SpatialRAGService(
            config=SpatialRAGServiceConfig(
                db_path=self.config.memory_db,
                embedding_dim=self.config.embedding_dim,
                top_k_default=self.config.top_k,
                fusion_radius_m=self.config.fusion_radius_m,
                enable_global_fusion=True,
                semantic_similarity_threshold=self.config.semantic_similarity_threshold,
            ),
            embedder=self._make_embedder(),
        )

    def _make_embedder(self):
        if self._embedder is not None:
            return self._embedder

        from spatial_rag.embedding import (
            MockEmbeddingProvider,
            OllamaEmbeddingProvider,
            VLLMEmbeddingProvider,
        )

        provider = self.config.embedding_provider.lower()
        if provider == "ollama":
            self._embedder = OllamaEmbeddingProvider(
                dim=self.config.embedding_dim,
                model=self.config.embedding_model,
                base_url=self.config.embedding_endpoint,
            )
        elif provider == "vllm":
            self._embedder = VLLMEmbeddingProvider(
                dim=self.config.embedding_dim,
                model=self.config.embedding_model,
                base_url=self.config.embedding_endpoint,
            )
        elif provider == "mock":
            self._embedder = MockEmbeddingProvider(dim=self.config.embedding_dim)
        else:
            raise ValueError(
                "embedding_provider must be one of: mock, ollama, vllm; "
                f"got {self.config.embedding_provider!r}"
            )
        return self._embedder

    async def _query_memories(self, question: str, top_k: int) -> List[dict[str, Any]]:
        assert self._service is not None

        if self.config.embedding_provider.lower() != "mock":
            try:
                query_embedding = await self._service.embedder.embed(question)
                hits = await self._service.storage.retrieve_memory(
                    query_embedding, top_k=top_k
                )
                if hits:
                    return _normalize_hits(hits)
            except Exception as exc:
                if not self.config.lexical_fallback:
                    raise
                logger.warning("Semantic SpatialRAG query failed; using lexical fallback: %s", exc)

        if self.config.lexical_fallback:
            return await self._lexical_query(question, top_k=top_k)
        return []

    async def _lexical_query(self, question: str, top_k: int) -> List[dict[str, Any]]:
        assert self._service is not None
        all_objects = await self._service.storage.get_all_objects(limit=self.config.object_limit)
        query_tokens = _content_tokens(question)
        scored = []
        for obj in all_objects:
            label = _extract_label(obj.get("snapshot_text", ""))
            text = f"{label} {obj.get('snapshot_text', '')}"
            tokens = _content_tokens(text)
            overlap = len(query_tokens & tokens)
            label_bonus = 2 if label.lower() in question.lower() else 0
            score = overlap + label_bonus
            scored.append((_object_sort_key(score, obj), _object_hit(obj, label, score)))
        scored.sort(key=lambda item: item[0])
        return [item[1] for item in scored[:top_k]]

    def _lexical_query_rows(self, question: str, top_k: int) -> List[dict[str, Any]]:
        assert self._objects is not None
        query_tokens = _content_tokens(question)
        scored = []
        for obj in self._objects:
            label = str(obj["label"])
            text = f"{label} {obj.get('snapshot_text', '')}"
            tokens = _content_tokens(text)
            overlap = len(query_tokens & tokens)
            label_bonus = 2 if label.lower() in question.lower() else 0
            score = overlap + label_bonus
            scored.append((_object_sort_key(score, obj), {**obj, "score": float(score)}))
        scored.sort(key=lambda item: item[0])
        return [item[1] for item in scored[:top_k]]

    async def _get_matching_objects(self, query: str) -> List[dict[str, Any]]:
        assert self._service is not None
        all_objects = await self._service.storage.get_all_objects(limit=self.config.object_limit)
        query_tokens = _content_tokens(query)
        label_matches = []
        other_matches = []
        for obj in all_objects:
            label = _extract_label(obj.get("snapshot_text", ""))
            haystack = f"{label} {obj.get('snapshot_text', '')}".lower()
            label_text = label.lower()
            token_overlap = len(query_tokens & _content_tokens(haystack))
            label_match = query.strip().lower() in label_text
            substring = query.strip().lower() in haystack
            if label_match or substring or token_overlap:
                if label_match:
                    score = 1.0
                elif substring:
                    score = 0.6
                else:
                    score = min(0.59, 0.25 + 0.15 * token_overlap)
                target = label_matches if label_match else other_matches
                target.append(_object_hit(obj, label, score))

        matches = label_matches or other_matches
        if matches:
            matches.sort(key=lambda obj: (-float(obj["score"]), str(obj["label"]), int(obj["id"])))
            return matches

        return await self._lexical_query(query, top_k=self.config.top_k)

    def _matching_object_rows(self, query: str) -> List[dict[str, Any]]:
        assert self._objects is not None
        query_tokens = _content_tokens(query)
        label_matches = []
        other_matches = []
        for obj in self._objects:
            haystack = f"{obj['label']} {obj.get('snapshot_text', '')}".lower()
            label_text = str(obj["label"]).lower()
            token_overlap = len(query_tokens & _content_tokens(haystack))
            label_match = query.strip().lower() in label_text
            substring = query.strip().lower() in haystack
            if label_match or substring or token_overlap:
                if label_match:
                    score = 1.0
                elif substring:
                    score = 0.6
                else:
                    score = min(0.59, 0.25 + 0.15 * token_overlap)
                target = label_matches if label_match else other_matches
                target.append({**obj, "score": float(score)})

        matches = label_matches or other_matches
        if matches:
            matches.sort(key=lambda obj: (-float(obj["score"]), str(obj["label"]), int(obj["id"])))
            return matches

        return self._lexical_query_rows(query, top_k=self.config.top_k)

    async def _build_memory_from_sequence(self) -> None:
        if self.config.memory_db.exists():
            if not self.config.overwrite_memory_db:
                return
            self.config.memory_db.unlink()
        self.config.memory_db.parent.mkdir(parents=True, exist_ok=True)

        from scipy.spatial.transform import Rotation
        from spatial_rag.clawspine_adapter import SpatialRAGService, SpatialRAGServiceConfig
        from spatial_rag.depth_utils import DetectionProjector
        from spatial_rag.pipeline import PipelineConfig, SpatialPipeline
        from spatial_rag.projection import CameraIntrinsics, RobotPose
        from spatial_rag.visual_trigger import (
            TrackEventType,
            UltralyticsBackend,
            VisualTrigger,
            VisualTriggerConfig,
        )

        import cv2

        rag_service = SpatialRAGService(
            config=SpatialRAGServiceConfig(
                db_path=self.config.memory_db,
                embedding_dim=self.config.embedding_dim,
                top_k_default=self.config.top_k,
                fusion_radius_m=self.config.fusion_radius_m,
                enable_global_fusion=True,
                semantic_similarity_threshold=self.config.semantic_similarity_threshold,
            ),
            embedder=self._make_embedder(),
        )
        yolo_model = self.config.yolo_model or str(
            self.config.spatial_rag_root / "yolo11n.pt"
        )
        backend = UltralyticsBackend(
            VisualTriggerConfig(
                model_path=yolo_model,
                tracker_config="bytetrack.yaml",
                confidence_threshold=self.config.track_confidence,
            )
        )
        trigger = VisualTrigger(backend)
        projector = DetectionProjector()
        pipeline = SpatialPipeline(
            backend,
            trigger,
            projector,
            config=PipelineConfig(
                store_on_events=frozenset({TrackEventType.NEW_TRACK}),
                min_confirm_frames=self.config.min_confirm_frames,
                min_store_confidence=self.config.min_store_confidence,
            ),
            rag_service=rag_service,
        )

        try:
            intrinsics = _load_intrinsics(self.sequence)
            if intrinsics is not None:
                projector.update_intrinsics(intrinsics)

            frames = list(self.sequence.frames)
            if self.config.build_frame_stride > 1:
                frames = frames[:: self.config.build_frame_stride]
            if self.config.max_build_frames is not None:
                frames = frames[: self.config.max_build_frames]

            for frame in frames:
                rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise ValueError(f"Could not read RGB frame: {frame.rgb_path}")

                depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    raise ValueError(f"Could not read depth frame: {frame.depth_path}")
                depth_m = _depth_to_meters(depth)

                pose_matrix = np.loadtxt(frame.pose_path)
                rotation = Rotation.from_matrix(pose_matrix[:3, :3])
                qx, qy, qz, qw = rotation.as_quat()
                pose = RobotPose(
                    x=float(pose_matrix[0, 3]),
                    y=float(pose_matrix[1, 3]),
                    z=float(pose_matrix[2, 3]),
                    qx=float(qx),
                    qy=float(qy),
                    qz=float(qz),
                    qw=float(qw),
                )

                await pipeline.process_frame(
                    rgb_bgr=rgb,
                    depth_m=depth_m,
                    robot_pose=pose,
                    timestamp=float(frame.index),
                )
        finally:
            await rag_service.close()

    def _run_async(self, awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("ClawSSpatialRAGMethod sync API cannot run inside an active loop")


def create_method(
    sequence: RGBDSequence,
    spatial_rag_root: str = str(DEFAULT_CLAWS_ROOT),
    memory_db: str = str(DEFAULT_MEMORY_DB),
    scene_id: str = DEFAULT_SCENE_ID,
    build_from_sequence: bool = False,
    overwrite_memory_db: bool = False,
    embedding_provider: str = "mock",
    embedding_model: str = "qwen3-embedding:0.6b",
    embedding_endpoint: str = "http://localhost:11434",
    embedding_dim: int = 1024,
    top_k: int = 5,
    object_limit: int = 500,
    max_build_frames: Optional[int] = None,
    build_frame_stride: int = 1,
    yolo_model: Optional[str] = None,
    track_confidence: float = 0.45,
    min_store_confidence: float = 0.45,
    min_confirm_frames: int = 2,
    fusion_radius_m: float = 0.5,
    semantic_similarity_threshold: float = 0.8,
    lexical_fallback: bool = True,
) -> ClawSSpatialRAGMethod:
    resolved_scene_id = _resolve_scene_id(scene_id, sequence)
    resolved_memory_db = _resolve_memory_db(memory_db, sequence, resolved_scene_id)
    config = ClawSAdapterConfig(
        spatial_rag_root=Path(spatial_rag_root).expanduser(),
        memory_db=resolved_memory_db,
        scene_id=resolved_scene_id,
        build_from_sequence=build_from_sequence,
        overwrite_memory_db=overwrite_memory_db,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_endpoint=embedding_endpoint,
        embedding_dim=embedding_dim,
        top_k=top_k,
        object_limit=object_limit,
        max_build_frames=max_build_frames,
        build_frame_stride=build_frame_stride,
        yolo_model=yolo_model,
        track_confidence=track_confidence,
        min_store_confidence=min_store_confidence,
        min_confirm_frames=min_confirm_frames,
        fusion_radius_m=fusion_radius_m,
        semantic_similarity_threshold=semantic_similarity_threshold,
        lexical_fallback=lexical_fallback,
    )
    return ClawSSpatialRAGMethod(sequence=sequence, config=config)


def _add_repo_to_path(path: Path) -> None:
    root = str(path.expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)


def _resolve_scene_id(scene_id: str, sequence: RGBDSequence) -> str:
    if scene_id and scene_id.lower() != "auto":
        return scene_id
    episode = sequence.episode_history.split("/")[-1]
    if "scannet-" in episode:
        return episode.split("scannet-")[-1]
    return episode


def _resolve_memory_db(memory_db: str, sequence: RGBDSequence, scene_id: str) -> Path:
    episode_history = sequence.episode_history
    episode = _safe_path_component(episode_history)
    raw = str(memory_db)
    if "{" in raw and "}" in raw:
        raw = raw.format(
            episode_history=episode_history,
            episode=episode,
            scene_id=scene_id,
        )
    return Path(raw).expanduser()


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value.strip("/"))


def _load_intrinsics(sequence: RGBDSequence):
    if sequence.intrinsic_depth_path is None and sequence.intrinsic_color_path is None:
        return None
    from spatial_rag.projection import CameraIntrinsics

    path = sequence.intrinsic_depth_path or sequence.intrinsic_color_path
    matrix = np.loadtxt(path)
    return CameraIntrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )


def _load_memory_rows(memory_db: Path, limit: int = 10000) -> List[dict[str, Any]]:
    import sqlite3

    conn = sqlite3.connect(str(memory_db))
    conn.row_factory = sqlite3.Row
    try:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass

        table = _choose_memory_table(conn)
        rows = _fetch_memory_rows(conn, table=table, limit=limit)
        return [
            {
                "id": int(row["id"]),
                "label": _extract_label(row["snapshot_text"]),
                "snapshot_text": row["snapshot_text"],
                "pos_x": float(row["pos_x"]),
                "pos_y": float(row["pos_y"]),
                "pos_z": float(row["pos_z"]),
                "timestamp": float(row["timestamp"] or 0.0),
            }
            for row in rows
        ]
    finally:
        conn.close()


def _choose_memory_table(conn) -> str:
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
    }
    if "spatial_memories" in tables:
        return "spatial_memories"
    for table in sorted(tables):
        cols = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info([{table}])").fetchall()
        }
        if {"snapshot_text", "pos_x", "pos_y", "pos_z"}.issubset(cols):
            return table
    raise ValueError("No SpatialRAG memory table found")


def _fetch_memory_rows(conn, table: str, limit: int):
    sql = (
        f"SELECT rowid AS id, snapshot_text, pos_x, pos_y, pos_z, timestamp "
        f"FROM [{table}] ORDER BY timestamp DESC LIMIT ?"
    )
    try:
        return conn.execute(sql, (limit,)).fetchall()
    except Exception as exc:
        if "vec" not in str(exc).lower() and "no such module" not in str(exc).lower():
            raise

    aux_table = f"{table}_auxiliary"
    sql = (
        f"SELECT rowid AS id, value00 AS snapshot_text, value01 AS pos_x, "
        f"value02 AS pos_y, value03 AS pos_z, value04 AS timestamp "
        f"FROM [{aux_table}] ORDER BY value04 DESC LIMIT ?"
    )
    return conn.execute(sql, (limit,)).fetchall()


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    if depth.dtype == np.float32 and np.nanmax(depth) < 100.0:
        return depth
    return depth / 1000.0


def _extract_label(snapshot_text: str) -> str:
    text = str(snapshot_text or "")
    bold = re.search(r"\*\*([^*]+)\*\*", text)
    if bold:
        return bold.group(1).strip()
    object_line = re.search(r"^\s*object\s*:\s*(.+?)\s*$", text, re.I | re.M)
    if object_line:
        return object_line.group(1).strip()
    first_line = text.splitlines()[0].strip() if text.strip() else "object"
    return first_line[:80]


def _content_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "there",
        "to",
        "what",
        "where",
        "which",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in stop
    }


def _object_sort_key(score: float, obj: Mapping[str, Any]) -> tuple[float, float]:
    return (-float(score), -float(obj.get("timestamp") or 0.0))


def _object_hit(obj: Mapping[str, Any], label: str, score: float) -> dict[str, Any]:
    return {
        "id": obj["id"],
        "label": label,
        "snapshot_text": obj["snapshot_text"],
        "pos_x": obj["pos_x"],
        "pos_y": obj["pos_y"],
        "pos_z": obj["pos_z"],
        "timestamp": obj.get("timestamp"),
        "score": float(score),
    }


def _normalize_hits(hits: Iterable[Mapping[str, Any]]) -> List[dict[str, Any]]:
    out = []
    for hit in hits:
        label = str(hit.get("label") or _extract_label(str(hit.get("snapshot_text", ""))))
        out.append(
            {
                "id": hit.get("memory_id", hit.get("id")),
                "label": label,
                "snapshot_text": hit.get("snapshot_text", ""),
                "pos_x": hit.get("pos_x"),
                "pos_y": hit.get("pos_y"),
                "pos_z": hit.get("pos_z"),
                "timestamp": hit.get("timestamp"),
                "score": hit.get("score", 1.0),
                "distance": hit.get("distance"),
            }
        )
    return out


def _format_memory_context(question: str, hits: List[Mapping[str, Any]]) -> str:
    lines = [f"Question: {question}", "Relevant spatial memories:"]
    for idx, hit in enumerate(hits, 1):
        pos = "unknown"
        if hit.get("pos_x") is not None:
            pos = f"({hit['pos_x']:.2f}, {hit['pos_y']:.2f}, {hit['pos_z']:.2f})"
        lines.append(f"{idx}. [{pos}] {hit.get('snapshot_text', '')}")
    return "\n".join(lines)
