from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.hovsg.build_memory_smoke import (
    DEFAULT_CLAWS_ROOT,
    DEFAULT_SCANNETPP_ROOT,
    DEFAULT_SCENE_ID,
    export_scannetpp_iphone_layout,
    _iso_time,
    _run_timestamp,
)
from spatial_memory_evaluation.common.labels import (
    DEFAULT_DETECTOR_CLASS_LIST_PATH,
    validate_detector_class_list,
)
from spatial_memory_evaluation.memory_package_validator import validate_package


DEFAULT_DUALMAP_ROOT = Path("/home/robin_wang/DualMap")
DEFAULT_DUALMAP_PYTHON = Path("/home/robin_wang/miniforge3/envs/spatial-rag/bin/python")
DEFAULT_YOLO_CHECKPOINT = DEFAULT_DUALMAP_ROOT / "yolov8s-world.pt"
DEFAULT_SAM_CHECKPOINT = DEFAULT_DUALMAP_ROOT / "sam_b.pt"
DEFAULT_FASTSAM_CHECKPOINT = DEFAULT_DUALMAP_ROOT / "model" / "FastSAM-s.pt"
DEFAULT_CLASS_NAMES = DEFAULT_DETECTOR_CLASS_LIST_PATH
DUALMAP_IMAGE_HEIGHT = 192
DUALMAP_IMAGE_WIDTH = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DualMap memory smoke package from a ScanNet++ scene. "
            "The script exports a DualMap ScanNet-style RGB-D layout, runs "
            "DualMap applications/runner_dataset.py, and packages map/*.pkl "
            "as a minimal memory package."
        )
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--dualmap-root", type=Path, default=DEFAULT_DUALMAP_ROOT)
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path("data/dualmap_layouts"),
        help="where sampled DualMap ScanNet-style RGB-D layout is written",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help=(
            "prepared DualMap layout root. It must contain "
            "exported/scannetpp_<scene-id>/{color,depth,pose,intrinsic}."
        ),
    )
    parser.add_argument(
        "--native-output-root",
        type=Path,
        default=Path("data/dualmap_native"),
        help="where DualMap native output_path is written",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=Path("memories"),
        help="root for exported memory packages",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="sample every Nth ScanNet++ iPhone frame before DualMap",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="maximum sampled frames for smoke; use 0 for all frames selected by stride",
    )
    parser.add_argument(
        "--dualmap-python",
        type=Path,
        default=DEFAULT_DUALMAP_PYTHON if DEFAULT_DUALMAP_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="optional CUDA_VISIBLE_DEVICES value for the DualMap subprocess",
    )
    parser.add_argument(
        "--skip-cuda-preflight",
        action="store_true",
        help="skip a small CUDA/cuDNN test before launching DualMap",
    )
    parser.add_argument("--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip-length", type=int, default=512)
    parser.add_argument(
        "--enable-fastsam",
        action="store_true",
        help="enable DualMap FastSAM supplementation; disabled by default for smoke",
    )
    parser.add_argument("--fastsam-checkpoint", type=Path, default=DEFAULT_FASTSAM_CHECKPOINT)
    parser.add_argument(
        "--use-parallel",
        action="store_true",
        help="use DualMap parallel processing; smoke defaults to sequential processing",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="pass skip_refinement=true to DualMap for faster smoke runs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="only export the DualMap RGB-D layout and dataset yaml",
    )
    parser.add_argument(
        "--skip-dualmap-run",
        action="store_true",
        help="skip native DualMap run and package an existing --native-map-dir",
    )
    parser.add_argument(
        "--native-map-dir",
        type=Path,
        default=None,
        help="existing DualMap map directory containing *.pkl objects",
    )
    parser.add_argument(
        "--skip-layout-export",
        action="store_true",
        help="reuse the computed layout directory for this run-id",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    started = time.time()
    run_id = args.run_id or (args.layout_dir.name if args.layout_dir is not None else _run_timestamp())
    dualmap_scene_id = f"scannetpp_{args.scene_id}"
    layout_dir = args.layout_dir or (args.layout_root / dualmap_scene_id / run_id)
    dataset_config_path = layout_dir / f"{dualmap_scene_id}_dataset.yaml"
    native_run_root = args.native_output_root / dualmap_scene_id / run_id
    native_map_dir = args.native_map_dir or (native_run_root / f"scannet_{dualmap_scene_id}" / "map")
    package_dir = args.package_root / "dualmap" / "scannetpp" / args.scene_id / run_id

    if args.layout_dir is not None or args.skip_layout_export:
        layout_summary = _read_prepared_layout_summary(layout_dir, dualmap_scene_id)
        dataset_config_path = Path(layout_summary.get("dataset_config_path") or dataset_config_path)
    else:
        layout_summary = export_dualmap_scannetpp_layout(
            scannetpp_root=args.scannetpp_root,
            claws_root=args.claws_root,
            scene_id=args.scene_id,
            dualmap_scene_id=dualmap_scene_id,
            layout_dir=layout_dir,
            frame_stride=args.frame_stride,
            max_frames=None if args.max_frames == 0 else args.max_frames,
        )

    if args.prepare_only:
        print(json.dumps({"status": "prepared", "layout": layout_summary}, indent=2))
        return 0

    if not args.skip_dualmap_run:
        _preflight_dualmap(args)
        run_dualmap_native(
            args,
            layout_dir=layout_dir,
            dataset_config_path=dataset_config_path,
            dualmap_scene_id=dualmap_scene_id,
            native_run_root=native_run_root,
        )

    export_summary = export_minimal_package(
        package_dir=package_dir,
        native_map_dir=native_map_dir,
        layout_dir=layout_dir,
        dataset_config_path=dataset_config_path,
        args=args,
        run_id=run_id,
        dualmap_scene_id=dualmap_scene_id,
        layout_summary=layout_summary,
        started_at=started,
    )
    print(json.dumps(export_summary, indent=2))
    return 0


def export_dualmap_scannetpp_layout(
    *,
    scannetpp_root: Path,
    claws_root: Path,
    scene_id: str,
    dualmap_scene_id: str,
    layout_dir: Path,
    frame_stride: int,
    max_frames: Optional[int],
) -> dict[str, Any]:
    scene_layout_dir = layout_dir / "exported" / dualmap_scene_id
    native_summary = export_scannetpp_iphone_layout(
        scannetpp_root=scannetpp_root,
        claws_root=claws_root,
        scene_id=scene_id,
        output_dir=scene_layout_dir,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    dataset_config_path = layout_dir / f"{dualmap_scene_id}_dataset.yaml"
    _write_dualmap_dataset_config(dataset_config_path)
    summary = {
        **native_summary,
        "layout_dir": str(layout_dir),
        "dualmap_scene_id": dualmap_scene_id,
        "scene_layout_dir": str(scene_layout_dir),
        "dataset_config_path": str(dataset_config_path),
        "image_height": DUALMAP_IMAGE_HEIGHT,
        "image_width": DUALMAP_IMAGE_WIDTH,
        "png_depth_scale": 1000.0,
    }
    _write_json(layout_dir / "layout_summary.json", summary)
    return summary


def run_dualmap_native(
    args: argparse.Namespace,
    *,
    layout_dir: Path,
    dataset_config_path: Path,
    dualmap_scene_id: str,
    native_run_root: Path,
) -> None:
    native_run_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.dualmap_python),
        "applications/runner_dataset.py",
        "dataset_name=scannet",
        f"scene_id={dualmap_scene_id}",
        f"dataset_path={layout_dir.resolve()}",
        f"dataset_conf_path={dataset_config_path.resolve()}",
        f"output_path={native_run_root.resolve()}",
        "use_stride=true",
        "stride=1",
        "start=0",
        "end=-1",
        "run_detection=true",
        f"device={args.device}",
        f"yolo.model_path={args.yolo_checkpoint.resolve()}",
        f"yolo.given_classes_path={args.class_names.resolve()}",
        f"sam.model_path={args.sam_checkpoint.resolve()}",
        f"clip.model_name={args.clip_model}",
        f"clip.pretrained={args.clip_pretrained}",
        f"clip.clip_length={args.clip_length}",
        f"use_fastsam={str(args.enable_fastsam).lower()}",
        f"fastsam.model_path={args.fastsam_checkpoint.resolve()}",
        f"use_parallel={str(args.use_parallel).lower()}",
        f"skip_refinement={str(args.skip_refinement).lower()}",
        "use_rerun=false",
        "visualize_detection=false",
        "show_fastsam_debug=false",
        "show_3d_bbox_overlapped=false",
        "run_local_mapping_only=true",
        "save_local_map=true",
        "save_global_map=false",
        "merge_local_map=false",
        "save_layout=false",
        "save_detection=false",
        "use_end_process=true",
    ]
    env = _dualmap_subprocess_env(args)
    print("running DualMap:")
    print(" ".join(command))
    subprocess.run(command, cwd=args.dualmap_root, env=env, check=True)


def export_minimal_package(
    *,
    package_dir: Path,
    native_map_dir: Path,
    layout_dir: Path,
    dataset_config_path: Path,
    args: argparse.Namespace,
    run_id: str,
    dualmap_scene_id: str,
    layout_summary: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    _require_dualmap_map(native_map_dir)
    _reset_dir(package_dir)
    for directory in ("memory", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / directory).mkdir(parents=True, exist_ok=True)

    objects, features, feature_warning = _load_dualmap_objects(
        native_map_dir=native_map_dir,
        dualmap_root=args.dualmap_root,
        class_names_path=args.class_names,
    )
    has_features = features is not None
    if features is not None:
        np.save(package_dir / "memory" / "object_features.npy", features)

    _write_jsonl(package_dir / "memory" / "object_table.jsonl", objects)
    _write_jsonl(
        package_dir / "evidence" / "object_native_paths.jsonl",
        [
            {
                "object_id": obj["object_id"],
                "native_pkl_path": obj["native_pkl_path"],
                "feature_index": obj["feature_index"],
            }
            for obj in objects
        ],
    )
    _write_json(
        package_dir / "raw_links" / "native_sources.json",
        {
            "scannetpp_root": str(args.scannetpp_root),
            "dualmap_root": str(args.dualmap_root),
            "layout_dir": str(layout_dir),
            "dataset_config_path": str(dataset_config_path),
            "native_map_dir": str(native_map_dir),
            "source_scene": args.scene_id,
            "dualmap_scene_id": dualmap_scene_id,
        },
    )
    _write_tool_files(package_dir)
    _write_package_schemas(package_dir)
    _write_schema_md(package_dir, args, native_map_dir, has_features)
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        native_map_dir=native_map_dir,
        layout_dir=layout_dir,
        dataset_config_path=dataset_config_path,
        dualmap_scene_id=dualmap_scene_id,
        object_count=len(objects),
        has_features=has_features,
        layout_summary=layout_summary,
    )
    _write_capabilities(package_dir)
    warnings = [feature_warning] if feature_warning else []
    _write_build_log(
        package_dir=package_dir,
        args=args,
        started_at=started_at,
        object_count=len(objects),
        native_map_dir=native_map_dir,
        warnings=warnings,
    )

    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))
    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "native_map_dir": str(native_map_dir),
        "object_count": len(objects),
        "has_features": has_features,
        "validation": report.to_json(),
    }


def _preflight_dualmap(args: argparse.Namespace) -> None:
    if not args.dualmap_root.exists():
        raise FileNotFoundError(f"DualMap repo not found: {args.dualmap_root}")
    if not args.dualmap_python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.dualmap_python}")
    if not args.yolo_checkpoint.exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.yolo_checkpoint}")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    if not args.class_names.exists():
        raise FileNotFoundError(f"class list not found: {args.class_names}")
    validate_detector_class_list(args.class_names)
    if args.enable_fastsam and not args.fastsam_checkpoint.exists():
        raise FileNotFoundError(f"FastSAM checkpoint not found: {args.fastsam_checkpoint}")
    if args.device == "cuda" and not args.skip_cuda_preflight:
        _preflight_cuda(args)


def _preflight_cuda(args: argparse.Namespace) -> None:
    code = r"""
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"cudnn_version={torch.backends.cudnn.version()}")
print(f"device_count={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")

device = torch.device("cuda")
conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).to(device)
x = torch.randn(1, 3, 64, 64, device=device)
with torch.no_grad():
    y = conv(x)
torch.cuda.synchronize()
print(f"cuda_cudnn_smoke=ok shape={tuple(y.shape)}")
"""
    result = subprocess.run(
        [str(args.dualmap_python), "-c", code],
        cwd=args.dualmap_root,
        env=_dualmap_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    details = "\n".join(
        part
        for part in ("stdout:", result.stdout.strip(), "stderr:", result.stderr.strip())
        if part
    )
    raise RuntimeError(
        "CUDA/cuDNN preflight failed before launching DualMap. "
        "Run on a GPU node with a healthy NVIDIA driver, set "
        "--cuda-visible-devices to a valid GPU if needed, or pass "
        "--skip-cuda-preflight if you intentionally want DualMap to fail inside "
        f"its own runtime instead.\n{details}"
    )


def _dualmap_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.dualmap_root.resolve()),
            str(REPO_ROOT),
            str(args.claws_root.resolve()),
            env.get("PYTHONPATH", ""),
        ]
    )
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def _read_prepared_layout_summary(layout_dir: Path, dualmap_scene_id: str) -> dict[str, Any]:
    if not layout_dir.exists():
        raise FileNotFoundError(f"prepared DualMap layout does not exist: {layout_dir}")
    summary_path = layout_dir / "layout_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"prepared DualMap layout is missing {summary_path}. "
            "Run build_memory_smoke.py with --prepare-only first, or omit "
            "--skip-layout-export."
        )
    summary = _read_json(summary_path)
    if summary.get("dualmap_scene_id") not in (None, dualmap_scene_id):
        raise ValueError(
            f"prepared layout dualmap_scene_id mismatch: expected {dualmap_scene_id}, "
            f"found {summary.get('dualmap_scene_id')}"
        )
    _require_prepared_layout_files(layout_dir, dualmap_scene_id)
    summary["layout_dir"] = str(layout_dir)
    return summary


def _require_prepared_layout_files(layout_dir: Path, dualmap_scene_id: str) -> None:
    scene_layout_dir = layout_dir / "exported" / dualmap_scene_id
    missing = [
        path
        for path in (
            scene_layout_dir / "color",
            scene_layout_dir / "depth",
            scene_layout_dir / "pose",
            scene_layout_dir / "intrinsic" / "intrinsic_color.txt",
            scene_layout_dir / "intrinsic" / "intrinsic_depth.txt",
            layout_dir / f"{dualmap_scene_id}_dataset.yaml",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("prepared DualMap layout is incomplete:\n" + "\n".join(str(path) for path in missing))
    color_count = sum(1 for _ in (scene_layout_dir / "color").glob("*.jpg"))
    depth_count = sum(1 for _ in (scene_layout_dir / "depth").glob("*.png"))
    pose_count = sum(1 for _ in (scene_layout_dir / "pose").glob("*.txt"))
    if color_count == 0 or depth_count == 0 or pose_count == 0:
        raise ValueError(f"prepared DualMap layout has empty color/depth/pose folders: {scene_layout_dir}")
    if len({color_count, depth_count, pose_count}) != 1:
        raise ValueError(
            "prepared DualMap layout has mismatched frame counts: "
            f"color={color_count}, depth={depth_count}, pose={pose_count}"
        )


def _require_dualmap_map(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"DualMap native map directory does not exist: {path}")
    pkl_paths = sorted(path.glob("*.pkl"))
    if not pkl_paths:
        raise ValueError(f"no DualMap object pickle files found under {path}")


def _load_dualmap_objects(
    *,
    native_map_dir: Path,
    dualmap_root: Path,
    class_names_path: Path,
) -> tuple[list[dict[str, Any]], Optional[np.ndarray], Optional[str]]:
    sys.path.insert(0, str(dualmap_root))
    try:
        from utils.object import BaseObject
    finally:
        try:
            sys.path.remove(str(dualmap_root))
        except ValueError:
            pass

    labels = _read_labels(class_names_path)
    objects: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    feature_dim: Optional[int] = None
    feature_warning: Optional[str] = None

    for ordinal, path in enumerate(sorted(native_map_dir.glob("*.pkl"), key=_natural_key)):
        obj = BaseObject.load_from_disk(str(path))
        points = _object_points(obj)
        bbox, center, num_points = _bounds_from_points(points)
        class_id = _safe_int(getattr(obj, "class_id", None))
        label = labels[class_id] if class_id is not None and 0 <= class_id < len(labels) else "object"
        feature = _object_feature(obj)
        feature_index: Optional[int] = None
        if feature is not None:
            if feature_dim is None:
                feature_dim = int(feature.shape[0])
            if feature.shape[0] == feature_dim:
                feature_index = len(features)
                features.append(feature)
            elif feature_warning is None:
                feature_warning = f"skipped at least one feature with unexpected dim {feature.shape[0]}"

        objects.append(
            {
                "object_id": path.stem,
                "label": label,
                "aliases": [],
                "position_3d": center,
                "bbox_3d": bbox,
                "confidence": None,
                "label_score": None,
                "num_points": num_points,
                "class_id": class_id,
                "feature_index": feature_index,
                "source_artifacts": ["memory/object_table.jsonl"],
                "native_pkl_path": str(path),
                "evidence": [
                    {
                        "source_type": "native_dualmap_object",
                        "source_path": "evidence/object_native_paths.jsonl",
                        "notes": "Package-local index points to the native DualMap object pickle.",
                    }
                ],
            }
        )

    if not objects:
        raise ValueError(f"no DualMap objects found under {native_map_dir}")
    feature_array = None
    if features:
        feature_array = np.stack(features).astype(np.float32)
        norms = np.linalg.norm(feature_array, axis=1, keepdims=True)
        feature_array = feature_array / np.maximum(norms, 1e-12)
    return objects, feature_array, feature_warning


def _object_points(obj: Any) -> Optional[np.ndarray]:
    pcd = getattr(obj, "pcd", None)
    if pcd is None or not hasattr(pcd, "points"):
        return None
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return None
    finite = np.isfinite(points[:, :3]).all(axis=1)
    points = points[finite, :3]
    return points if len(points) else None


def _bounds_from_points(points: Optional[np.ndarray]) -> tuple[Optional[list[float]], Optional[list[float]], int]:
    if points is None or len(points) == 0:
        return None, None, 0
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    return min_bound.tolist() + max_bound.tolist(), center.tolist(), int(len(points))


def _object_feature(obj: Any) -> Optional[np.ndarray]:
    feature = getattr(obj, "clip_ft", None)
    if feature is None:
        return None
    array = np.asarray(feature, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return array


def _write_tool_files(package_dir: Path) -> None:
    (package_dir / "tools" / "list_objects.py").write_text(
        """from __future__ import annotations

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
""",
        encoding="utf-8",
    )
    (package_dir / "tools" / "query_object.py").write_text(
        """from __future__ import annotations

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
""",
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
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
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


def _write_schema_md(package_dir: Path, args: argparse.Namespace, native_map_dir: Path, has_features: bool) -> None:
    (package_dir / "schema.md").write_text(
        f"""# DualMap Minimal Memory Schema

Coordinate frame and units: object positions and bounding boxes use the DualMap
local map frame produced from ScanNet++ aligned camera poses. Units are meters.

Object id format: object ids follow the native DualMap pickle stem, usually the
UUID assigned by DualMap.

Timestamp format: this smoke package exports static object memory, so object
timestamps are not present. Source frame indices are recorded in
`raw_links/native_sources.json`.

Relation representation: this smoke package does not export DualMap global
relations or navigation graph state. It only exports concrete object memory from
DualMap local map pickles.

Confidence or score meaning: `label` is derived from DualMap's configured YOLO
class list and the object `class_id`. `confidence` and `label_score` are null
because the saved local map pickle does not expose one scalar confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `memory/object_features.npy` is present only when object CLIP
features are readable and shape-consistent; present={has_features}. The native
DualMap map directory is `{native_map_dir}`.

Known limitations and unsupported tracks: Track 1 object inventory and Track 2
basic object query are exported for smoke testing. Track 3 ScanRefer and Track 4
OpenEQA are invalid for this package.
""",
        encoding="utf-8",
    )


def _write_manifest(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    native_map_dir: Path,
    layout_dir: Path,
    dataset_config_path: Path,
    dualmap_scene_id: str,
    object_count: int,
    has_features: bool,
    layout_summary: dict[str, Any],
) -> None:
    _write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.1",
            "package_id": f"dualmap/scannetpp/{args.scene_id}/{run_id}",
            "method": {
                "name": "dualmap",
                "display_name": "DualMap",
                "family": "object_map",
                "repo_path": str(args.dualmap_root),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": "scannetpp",
                "split": "smoke",
                "scene_id": args.scene_id,
                "episode_id": None,
                "dualmap_scene_id": dualmap_scene_id,
            },
            "input": {
                "modality": ["rgb", "depth", "pose", "intrinsics"],
                "frame_count": int(layout_summary.get("frame_count") or 0),
                "rgbd_root": str(layout_dir / "exported" / dualmap_scene_id),
                "poses_path": str(layout_dir / "exported" / dualmap_scene_id / "pose"),
                "intrinsics_path": str(layout_dir / "exported" / dualmap_scene_id / "intrinsic"),
                "timestamp_path": None,
                "coordinate_frame": "DualMap local map frame from ScanNet++ aligned poses; meters",
            },
            "explicit_memory": True,
            "memory_artifacts": [
                {
                    "name": "object_table",
                    "type": "jsonl",
                    "path": "memory/object_table.jsonl",
                    "description": f"DualMap object inventory with {object_count} objects.",
                    "required_for": ["track1_memory_construction", "track2_object_location"],
                }
            ]
            + (
                [
                    {
                        "name": "object_features",
                        "type": "npy",
                        "path": "memory/object_features.npy",
                        "description": "Normalized DualMap object CLIP features.",
                        "required_for": [],
                    }
                ]
                if has_features
                else []
            ),
            "evidence_artifacts": [
                {
                    "name": "native_object_paths",
                    "type": "jsonl",
                    "path": "evidence/object_native_paths.jsonl",
                    "description": "Mapping from package object ids to native DualMap object pickle paths.",
                    "required_for": [],
                }
            ],
            "raw_links": [
                {
                    "name": "native_dualmap_map",
                    "type": "directory",
                    "path": str(native_map_dir),
                    "description": "Native DualMap map directory containing object pickles.",
                    "required_for": [],
                },
                {
                    "name": "prepared_dualmap_layout",
                    "type": "directory",
                    "path": str(layout_dir),
                    "description": "Sampled ScanNet-style RGB-D layout used by DualMap.",
                    "required_for": [],
                },
                {
                    "name": "dualmap_dataset_config",
                    "type": "yaml",
                    "path": str(dataset_config_path),
                    "description": "Generated DualMap ScanNet dataset yaml.",
                    "required_for": [],
                },
            ],
            "tools": [
                {
                    "name": "list_objects",
                    "type": "python",
                    "path": "tools/list_objects.py",
                    "description": "Return the exported object table.",
                    "required_for": ["track1_memory_construction"],
                },
                {
                    "name": "query_object",
                    "type": "python",
                    "path": "tools/query_object.py",
                    "description": "Smoke object query over exported DualMap labels and object sizes.",
                    "required_for": ["track2_object_location"],
                },
            ],
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [str(dataset_config_path)],
                "environment": str(args.dualmap_python),
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
            "notes": "Smoke package built from DualMap native local map object pickles.",
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
                    "reason": "No native OpenEQA QA or retrieval API is exported.",
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
    native_map_dir: Path,
    warnings: list[str],
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
            "source_outputs": [str(native_map_dir)],
            "object_count": object_count,
            "dualmap_runtime": {
                "yolo_checkpoint": str(args.yolo_checkpoint),
                "sam_checkpoint": str(args.sam_checkpoint),
                "class_names": str(args.class_names),
                "fastsam_enabled": args.enable_fastsam,
                "fastsam_checkpoint": str(args.fastsam_checkpoint),
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
            },
            "warnings": warnings,
        },
    )


def _write_dualmap_dataset_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""dataset_name: scannet
camera_params:
  image_height: {DUALMAP_IMAGE_HEIGHT}
  image_width: {DUALMAP_IMAGE_WIDTH}
  png_depth_scale: 1000.0
  distortion: [0, 0, 0, 0, 0]
""",
        encoding="utf-8",
    )


def _read_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _natural_key(path: Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


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
