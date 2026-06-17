"""Build a ClawS SpatialRAG minimal memory package from a native ClawS DB.

ClawS SpatialRAG stores its spatial memory in a SQLite / sqlite-vec database
(the vec0 virtual table ``spatial_memories`` plus a ``crop_images`` table).
This exporter reads that native DB through ClawS's own ``SpatialStorage`` API
(which loads the sqlite-vec extension), then packages an object inventory as a
minimal memory package for Track 1 / Track 2 fixed-API evaluation.

It does not re-run the ClawS perception pipeline; it consumes the already-built
``outputs/scannetpp_memory_<scene>_*.db`` produced by
``scripts/build_scannetpp_spatial_rag_memory.py`` in the ClawS repo.

Object positions are in the ScanNet++ world frame (the same frame as the
Track 1 GT object inventory), which is how the native ClawS evaluator scores
them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.memory_package_validator import validate_package

DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_EMBEDDING_DIM = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ClawS SpatialRAG memory package.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Native ClawS sqlite-vec DB. Default: "
        "<claws-root>/outputs/scannetpp_memory_<scene>_ollama_vlm.db",
    )
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--package-root", type=Path, default=Path("memories"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="skip exporting object crop JPEGs into evidence/crops/",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    started = time.time()
    run_id = args.run_id or f"claws-fromdb-{_run_timestamp()}"
    db_path = args.db_path or (
        args.claws_root / "outputs" / f"scannetpp_memory_{args.scene_id}_ollama_vlm.db"
    )
    if not db_path.exists():
        raise FileNotFoundError(f"ClawS native DB not found: {db_path}")
    package_dir = args.package_root / "claws" / "scannetpp" / args.scene_id / run_id

    objects, crops = _load_claws_objects(
        claws_root=args.claws_root,
        db_path=db_path,
        embedding_dim=args.embedding_dim,
        limit=args.limit,
        export_crops=not args.no_crops,
    )

    export_summary = export_minimal_package(
        package_dir=package_dir,
        objects=objects,
        crops=crops,
        db_path=db_path,
        args=args,
        run_id=run_id,
        started_at=started,
    )
    print(json.dumps(export_summary, indent=2))
    return 0


def _load_claws_objects(
    *,
    claws_root: Path,
    db_path: Path,
    embedding_dim: int,
    limit: int,
    export_crops: bool,
) -> tuple[list[dict[str, Any]], dict[int, bytes]]:
    sys.path.insert(0, str(claws_root))
    try:
        from spatial_rag.config import SpatialRAGConfig
        from spatial_rag.storage import SpatialStorage, _extract_memory_label
    finally:
        try:
            sys.path.remove(str(claws_root))
        except ValueError:
            pass

    cfg = SpatialRAGConfig(db_path=str(db_path), embedding_dim=embedding_dim)
    storage = SpatialStorage(cfg)

    async def _collect() -> tuple[list[dict[str, Any]], dict[int, bytes]]:
        raw = await storage.get_all_objects(limit=limit)
        crops: dict[int, bytes] = {}
        if export_crops:
            for obj in raw:
                if obj.get("has_crop"):
                    crop = await storage.get_crop(int(obj["id"]))
                    if crop:
                        crops[int(obj["id"])] = crop
        await storage.close()
        return raw, crops

    raw, crops = asyncio.run(_collect())

    objects: list[dict[str, Any]] = []
    for obj in raw:
        memory_id = int(obj["id"])
        snapshot = str(obj.get("snapshot_text") or "")
        label = _extract_memory_label(snapshot) or "object"
        position = [float(obj["pos_x"]), float(obj["pos_y"]), float(obj["pos_z"])]
        evidence: list[dict[str, Any]] = []
        if memory_id in crops:
            evidence.append(
                {
                    "source_type": "crop",
                    "source_path": f"evidence/crops/{memory_id}.jpg",
                    "notes": "Object crop captured by the ClawS visual trigger.",
                }
            )
        objects.append(
            {
                "object_id": f"claws_{memory_id}",
                "memory_id": memory_id,
                "label": label,
                "aliases": [],
                "position_3d": position,
                "bbox_3d": None,
                "confidence": None,
                "snapshot_text": snapshot,
                "timestamp": obj.get("timestamp"),
                "source_artifacts": ["memory/object_table.jsonl"],
                "evidence": evidence,
            }
        )
    if not objects:
        raise ValueError(f"no ClawS objects found in {db_path}")
    return objects, crops


def export_minimal_package(
    *,
    package_dir: Path,
    objects: list[dict[str, Any]],
    crops: dict[int, bytes],
    db_path: Path,
    args: argparse.Namespace,
    run_id: str,
    started_at: float,
) -> dict[str, Any]:
    _reset_dir(package_dir)
    for directory in ("memory", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / directory).mkdir(parents=True, exist_ok=True)

    _write_jsonl(package_dir / "memory" / "object_table.jsonl", objects)

    crop_count = 0
    if crops:
        crops_dir = package_dir / "evidence" / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for memory_id, blob in crops.items():
            (crops_dir / f"{memory_id}.jpg").write_bytes(blob)
            crop_count += 1
    _write_jsonl(
        package_dir / "evidence" / "object_snapshots.jsonl",
        [
            {
                "object_id": obj["object_id"],
                "memory_id": obj["memory_id"],
                "label": obj["label"],
                "snapshot_text": obj["snapshot_text"],
                "crop_path": (
                    f"evidence/crops/{obj['memory_id']}.jpg"
                    if obj["memory_id"] in crops
                    else None
                ),
            }
            for obj in objects
        ],
    )
    _write_json(
        package_dir / "raw_links" / "native_sources.json",
        {
            "claws_root": str(args.claws_root),
            "native_db_path": str(db_path),
            "scannetpp_root": str(args.scannetpp_root),
            "source_scene": args.scene_id,
            "embedding_dim": args.embedding_dim,
        },
    )
    _write_tool_files(package_dir)
    _write_package_schemas(package_dir)
    _write_schema_md(package_dir, db_path, crop_count)
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        db_path=db_path,
        object_count=len(objects),
        crop_count=crop_count,
    )
    _write_capabilities(package_dir)
    _write_build_log(
        package_dir=package_dir,
        args=args,
        started_at=started_at,
        object_count=len(objects),
        crop_count=crop_count,
        db_path=db_path,
    )

    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))
    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "native_db_path": str(db_path),
        "object_count": len(objects),
        "crop_count": crop_count,
        "validation": report.to_json(),
    }


def _write_tool_files(package_dir: Path) -> None:
    (package_dir / "tools" / "list_objects.py").write_text(
        '''from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_objects(package_dir: str, query: dict[str, Any]) -> dict[str, Any]:
    path = Path(package_dir) / "memory" / "object_table.jsonl"
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return {"status": "ok", "objects": objects}
''',
        encoding="utf-8",
    )
    (package_dir / "tools" / "query_object.py").write_text(
        '''from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def query_object(package_dir: str, query: dict[str, Any]) -> dict[str, Any]:
    target_label = _normalize_label(
        query.get("target_label") or query.get("canonical_label") or query.get("object") or ""
    )
    query_text = _normalize_label(query.get("query") or "")
    top_k = int(query.get("top_k") or 5)
    objects = _load_objects(Path(package_dir))
    predictions = _rank(objects, target_label, query_text)[:top_k]
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


def _normalize_label(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\\s+", " ", text).strip()


def _rank(objects, target_label, query_text):
    ranked = []
    target_tokens = set(target_label.split())
    query_tokens = set(query_text.split())
    for obj in objects:
        label = _normalize_label(obj.get("label") or "object")
        snapshot = _normalize_label(obj.get("snapshot_text") or "")
        label_tokens = set(label.split())
        if target_label and target_label == label:
            score = 1.0
        elif target_label and (target_label in label or label in target_label):
            score = 0.9
        elif target_tokens and target_tokens & label_tokens:
            score = 0.75
        elif target_label and target_label in snapshot:
            score = 0.6
        elif query_text and query_text == label:
            score = 0.7
        elif query_text and (query_text in label or label in query_text):
            score = 0.65
        elif query_tokens and query_tokens & label_tokens:
            score = 0.55
        elif query_text in ("", "object", "objects"):
            score = 0.5
        else:
            score = 0.05
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
''',
        encoding="utf-8",
    )


def _write_package_schemas(package_dir: Path) -> None:
    _write_json(
        package_dir / "schemas" / "track1_input.schema.json",
        {"type": "object", "additionalProperties": True},
    )
    _write_json(
        package_dir / "schemas" / "object_table.schema.json",
        {
            "type": "object",
            "required": ["status", "objects"],
            "properties": {"status": {"const": "ok"}, "objects": {"type": "array"}},
        },
    )
    _write_json(
        package_dir / "schemas" / "track2_input.schema.json",
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "target_label": {"type": "string"},
                "canonical_label": {"type": "string"},
                "top_k": {"type": "integer"},
            },
        },
    )
    _write_json(
        package_dir / "schemas" / "object_query_result.schema.json",
        {
            "type": "object",
            "required": ["status", "predictions"],
            "properties": {"status": {"const": "ok"}, "predictions": {"type": "array"}},
        },
    )


def _write_schema_md(package_dir: Path, db_path: Path, crop_count: int) -> None:
    (package_dir / "schema.md").write_text(
        f"""# ClawS SpatialRAG Minimal Memory Schema

Coordinate frame and units: object positions use the ScanNet++ world frame
produced from aligned iPhone camera poses (same frame as the Track 1 GT object
inventory). Units are meters. Positions come from the ClawS spatial memory
record `pos_x/pos_y/pos_z`.

Object id format: `claws_<memory_id>`, where `<memory_id>` is the native
`spatial_memories` rowid in the ClawS sqlite-vec database.

Label meaning: `label` is extracted from the memory `snapshot_text` (the leading
`**<object>**` token or `object:` line), which is the YOLO11/COCO class assigned
by the ClawS visual trigger and optionally refined by the VLM describer. The
full `snapshot_text` description is preserved per object for evidence/agentic use.

Timestamp format: `timestamp` is the ClawS memory timestamp (float seconds).

Relation representation: this package exports object-level memory only. ClawS
does not export an explicit relation graph; relations live implicitly inside the
free-text `snapshot_text`.

Confidence: `confidence` is null because the saved ClawS memory record does not
expose a single scalar object confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `evidence/object_snapshots.jsonl` keeps each object's full
snapshot text and crop link. `evidence/crops/<memory_id>.jpg` holds
{crop_count} exported object crops. The native ClawS sqlite-vec database is
`{db_path}` (vec0 table `spatial_memories` + `crop_images`).

Known limitations and unsupported tracks: Track 1 object inventory and Track 2
basic object query are supported. Track 3 ScanRefer and Track 4 OpenEQA are
invalid for this package (no native referring-expression resolver or general QA
API is exported here).
""",
        encoding="utf-8",
    )


def _write_manifest(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    db_path: Path,
    object_count: int,
    crop_count: int,
) -> None:
    _write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.1",
            "package_id": f"claws/scannetpp/{args.scene_id}/{run_id}",
            "method": {
                "name": "claws",
                "display_name": "ClawS SpatialRAG",
                "family": "object_map",
                "repo_path": str(args.claws_root),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": "scannetpp",
                "split": "current-scene",
                "scene_id": args.scene_id,
                "episode_id": None,
            },
            "input": {
                "modality": ["rgb", "depth", "pose", "intrinsics", "timestamp"],
                "frame_count": 0,
                "rgbd_root": str(args.scannetpp_root / "data" / args.scene_id),
                "poses_path": None,
                "intrinsics_path": None,
                "timestamp_path": None,
                "coordinate_frame": "ScanNet++ world frame from aligned iPhone poses; meters",
            },
            "explicit_memory": True,
            "memory_artifacts": [
                {
                    "name": "object_table",
                    "type": "jsonl",
                    "path": "memory/object_table.jsonl",
                    "description": f"ClawS spatial-memory object inventory with {object_count} objects.",
                    "required_for": ["track1_memory_construction", "track2_object_location"],
                }
            ],
            "evidence_artifacts": [
                {
                    "name": "object_snapshots",
                    "type": "jsonl",
                    "path": "evidence/object_snapshots.jsonl",
                    "description": "Per-object snapshot text and crop link.",
                    "required_for": [],
                },
                {
                    "name": "object_crops",
                    "type": "directory",
                    "path": "evidence/crops",
                    "description": f"{crop_count} object crop JPEGs captured by ClawS.",
                    "required_for": [],
                },
            ],
            "raw_links": [
                {
                    "name": "native_claws_db",
                    "type": "file",
                    "path": str(db_path),
                    "description": "Native ClawS sqlite-vec spatial-memory database.",
                    "required_for": [],
                }
            ],
            "tools": [
                {
                    "name": "list_objects",
                    "type": "python",
                    "path": "tools/list_objects.py",
                    "description": "Return the exported ClawS object table.",
                    "required_for": ["track1_memory_construction"],
                },
                {
                    "name": "query_object",
                    "type": "python",
                    "path": "tools/query_object.py",
                    "description": "Label/snapshot object query over exported ClawS memory.",
                    "required_for": ["track2_object_location"],
                },
            ],
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [],
                "environment": None,
                "started_at": None,
                "finished_at": None,
                "runtime_seconds": None,
                "memory_size_bytes": None,
            },
            "allowed_access": {
                "contains_gt_annotations": False,
                "contains_benchmark_answers": False,
                "contains_test_labels": False,
                "contains_question_specific_rules": False,
            },
            "notes": "Package built from a pre-built ClawS SpatialRAG sqlite-vec DB for one scene.",
        },
    )


def _write_capabilities(package_dir: Path) -> None:
    _write_json(
        package_dir / "capabilities.json",
        {
            "schema_version": "0.1",
            "fixed_api": {
                "track1_memory_construction": {
                    "status": "supported",
                    "entrypoint": "tools/list_objects.py:list_objects",
                    "reason": "",
                    "input_schema": "schemas/track1_input.schema.json",
                    "output_schema": "schemas/object_table.schema.json",
                },
                "track2_object_location": {
                    "status": "supported",
                    "entrypoint": "tools/query_object.py:query_object",
                    "reason": "",
                    "input_schema": "schemas/track2_input.schema.json",
                    "output_schema": "schemas/object_query_result.schema.json",
                },
                "track3_scanrefer": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": "No ScanRefer referring-expression resolver is exported.",
                },
                "track4_openeqa": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": "No native OpenEQA QA or retrieval API is exported in this package.",
                },
            },
            "agent_access": {
                "mode": "memory_only",
                "read_manifest": True,
                "read_schema": True,
                "read_memory_artifacts": True,
                "read_evidence": True,
                "read_raw_links": False,
                "read_raw_frames": False,
                "read_source_keyframes_or_crops": False,
                "run_package_tools": False,
                "write_package": False,
            },
        },
    )


def _write_build_log(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    started_at: float,
    object_count: int,
    crop_count: int,
    db_path: Path,
) -> None:
    finished_at = time.time()
    _write_json(
        package_dir / "build_log.json",
        {
            "status": "ok",
            "started_at": _iso_time(started_at),
            "finished_at": _iso_time(finished_at),
            "runtime_seconds": finished_at - started_at,
            "command": " ".join(sys.argv),
            "config_paths": [],
            "source_outputs": [str(db_path)],
            "object_count": object_count,
            "crop_count": crop_count,
            "claws_runtime": {
                "native_db_path": str(db_path),
                "embedding_dim": args.embedding_dim,
                "note": "Exported from a pre-built ClawS DB; perception pipeline not re-run here.",
            },
            "warnings": [],
        },
    )


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _iso_time(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
