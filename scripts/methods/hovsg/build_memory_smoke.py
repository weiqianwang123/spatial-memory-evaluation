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

from spatial_memory_evaluation.common.labels import (
    validate_detector_class_list,
)
from spatial_memory_evaluation.memory_package_validator import validate_package
from scripts.methods.shared_modules import (
    add_shared_module_args,
    apply_hovsg_shared_modules,
    shared_modules_metadata,
)


DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_HOVSG_ROOT = Path("/home/robin_wang/HOV-SG")
DEFAULT_SPATIAL_RAG_PYTHON = Path("/home/robin_wang/miniforge3/envs/spatial-rag/bin/python")
IPHONE_RGB_SHAPE = (1440, 1920)
IPHONE_DEPTH_SHAPE = (192, 256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a HOV-SG memory smoke package from a ScanNet++ scene. "
            "The script starts from ScanNet++ iPhone RGB-D files, prepares a "
            "HOV-SG/ScanNet-style layout, runs HOV-SG semantic_segmentation.py, "
            "and exports a minimal memory package."
        )
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--hovsg-root", type=Path, default=DEFAULT_HOVSG_ROOT)
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path("data/hovsg_layouts"),
        help="where sampled HOV-SG RGB-D layout is written",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help=(
            "prepared HOV-SG RGB-D layout directory. If set, this script will "
            "not export ScanNet++ frames; run prepare_eval_layout.py first."
        ),
    )
    parser.add_argument(
        "--native-output-root",
        type=Path,
        default=Path("data/hovsg_native"),
        help="where HOV-SG native output is written",
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
        default=120,
        help="sample every Nth ScanNet++ iPhone frame before HOV-SG",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="maximum sampled frames for the smoke run; use 0 for all frames",
    )
    parser.add_argument(
        "--hovsg-python",
        type=Path,
        default=DEFAULT_SPATIAL_RAG_PYTHON if DEFAULT_SPATIAL_RAG_PYTHON.exists() else Path(sys.executable),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "device passed to HOV-SG. The current HOV-SG extractor still "
            "hard-codes CUDA in several places, so native smoke builds require "
            "a working CUDA/cuDNN setup."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="optional CUDA_VISIBLE_DEVICES value for the HOV-SG subprocess",
    )
    parser.add_argument(
        "--skip-cuda-preflight",
        action="store_true",
        help="skip the CUDA/cuDNN smoke test before launching HOV-SG",
    )
    parser.add_argument(
        "--disable-safe-crop-patch",
        action="store_true",
        help=(
            "call HOV-SG application/semantic_segmentation.py directly. By "
            "default the wrapper installs a small crop guard that replaces "
            "empty SAM crops with blank crops to avoid OpenCV resize crashes."
        ),
    )
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    parser.add_argument("--sam-type", default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-points-per-side", type=int, default=8)
    parser.add_argument("--sam-points-per-batch", type=int, default=64)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--merge-type", default="sequential")
    parser.add_argument("--class-names", type=Path, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "legacy helper: only export the HOV-SG RGB-D layout; prefer "
            "scripts/methods/hovsg/prepare_eval_layout.py for new runs"
        ),
    )
    parser.add_argument(
        "--skip-hovsg-run",
        action="store_true",
        help="skip native HOV-SG run and package an existing --native-result-path",
    )
    parser.add_argument(
        "--native-result-path",
        type=Path,
        default=None,
        help="existing HOV-SG result directory containing mask_feats.pt and objects/",
    )
    parser.add_argument(
        "--skip-layout-export",
        action="store_true",
        help="reuse the computed layout directory for this run-id; prefer --layout-dir",
    )
    parser.add_argument(
        "--no-classify-labels",
        action="store_true",
        help="do not classify HOV-SG object features into the configured class list",
    )
    add_shared_module_args(parser)
    args = parser.parse_args()
    apply_hovsg_shared_modules(args)
    return args


def main(args: argparse.Namespace) -> int:
    started = time.time()
    run_id = args.run_id or (args.layout_dir.name if args.layout_dir is not None else _run_timestamp())
    layout_dir = args.layout_dir or (args.layout_root / f"scannetpp_{args.scene_id}" / run_id)
    native_save_path = args.native_output_root / f"scannetpp_{args.scene_id}" / run_id
    native_result_path = args.native_result_path or (native_save_path / "scannet")
    package_dir = args.package_root / "hovsg" / "scannetpp" / args.scene_id / run_id

    layout_summary: dict[str, Any] = {
        "layout_dir": str(layout_dir),
        "frame_count": None,
        "source_frame_indices": [],
    }
    if args.layout_dir is not None:
        layout_summary = _read_prepared_layout_summary(layout_dir, args.scene_id)
    elif not args.skip_layout_export:
        layout_summary = export_scannetpp_iphone_layout(
            scannetpp_root=args.scannetpp_root,
            claws_root=args.claws_root,
            scene_id=args.scene_id,
            output_dir=layout_dir,
            frame_stride=args.frame_stride,
            max_frames=None if args.max_frames == 0 else args.max_frames,
        )
    else:
        layout_summary = _read_prepared_layout_summary(layout_dir, args.scene_id)

    if args.prepare_only:
        print(json.dumps({"status": "prepared", "layout": layout_summary}, indent=2))
        return 0

    if not args.skip_hovsg_run:
        _preflight_hovsg(args)
        run_hovsg_native(args, layout_dir=layout_dir, native_save_path=native_save_path)

    export_summary = export_minimal_package(
        package_dir=package_dir,
        native_result_path=native_result_path,
        layout_dir=layout_dir,
        args=args,
        run_id=run_id,
        layout_summary=layout_summary,
        started_at=started,
    )
    print(json.dumps(export_summary, indent=2))
    return 0


def export_scannetpp_iphone_layout(
    *,
    scannetpp_root: Path,
    claws_root: Path,
    scene_id: str,
    output_dir: Path,
    frame_stride: int,
    max_frames: Optional[int],
) -> dict[str, Any]:
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")

    sys.path.insert(0, str(claws_root))
    try:
        from spatial_rag.datasets.scannetpp_adapter import (
            IPHONE_DEPTH_SHAPE as CLAWS_DEPTH_SHAPE,
            ScanNetPPDepthReader,
            detect_scene_layout,
        )
    finally:
        try:
            sys.path.remove(str(claws_root))
        except ValueError:
            pass

    layout = detect_scene_layout(scannetpp_root, scene_id)
    pose_data = json.loads(layout.pose_intrinsic_path.read_text(encoding="utf-8"))
    frame_ids = sorted(pose_data, key=_frame_sort_key)
    source_indices = list(range(0, len(frame_ids), frame_stride))
    if max_frames is not None:
        source_indices = source_indices[:max_frames]
    if not source_indices:
        raise ValueError("no ScanNet++ frames selected")

    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    intrinsic_dir = output_dir / "intrinsic"
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import cv2

    cap = cv2.VideoCapture(str(layout.rgb_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open ScanNet++ RGB video: {layout.rgb_path}")
    depth_reader = ScanNetPPDepthReader(layout.depth_path)

    first_frame_id = frame_ids[source_indices[0]]
    first_intrinsic = np.asarray(pose_data[first_frame_id]["intrinsic"], dtype=np.float64)
    rgb_shape = _video_shape(cap) or IPHONE_RGB_SHAPE
    depth_shape = tuple(CLAWS_DEPTH_SHAPE)
    depth_intrinsic = _scale_intrinsic(first_intrinsic, rgb_shape, depth_shape)
    _write_matrix(intrinsic_dir / "intrinsic_color.txt", depth_intrinsic)
    _write_matrix(intrinsic_dir / "intrinsic_depth.txt", depth_intrinsic)

    exported = []
    for out_index, source_index in enumerate(source_indices):
        frame_id = frame_ids[source_index]
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read RGB frame {source_index} from {layout.rgb_path}")
        depth_m = depth_reader.read(source_index)
        if tuple(depth_m.shape) != depth_shape:
            raise ValueError(f"unexpected depth shape {depth_m.shape}; expected {depth_shape}")

        bgr = cv2.resize(bgr, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_AREA)
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        pose = np.asarray(
            pose_data[frame_id].get("aligned_pose") or pose_data[frame_id]["pose"],
            dtype=np.float64,
        )

        stem = f"{out_index:06d}"
        cv2.imwrite(str(color_dir / f"{stem}.jpg"), bgr)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
        _write_matrix(pose_dir / f"{stem}.txt", pose)
        exported.append({"output_index": out_index, "source_index": source_index, "frame_id": frame_id})

    cap.release()
    summary = {
        "layout_dir": str(output_dir),
        "scene_id": scene_id,
        "frame_count": len(exported),
        "frame_stride": frame_stride,
        "source_frame_indices": [item["source_index"] for item in exported],
        "source_rgb": str(layout.rgb_path),
        "source_depth": str(layout.depth_path),
        "source_pose_intrinsic": str(layout.pose_intrinsic_path),
    }
    _write_json(output_dir / "layout_summary.json", summary)
    return summary


def run_hovsg_native(args: argparse.Namespace, *, layout_dir: Path, native_save_path: Path) -> None:
    native_save_path.mkdir(parents=True, exist_ok=True)
    entrypoint = (
        "application/semantic_segmentation.py"
        if args.disable_safe_crop_patch
        else str((REPO_ROOT / "scripts/methods/hovsg/run_semantic_segmentation_patched.py").resolve())
    )
    command = [
        str(args.hovsg_python),
        entrypoint,
        "main.dataset=scannet",
        f"main.dataset_path={layout_dir.resolve()}",
        f"main.save_path={native_save_path.resolve()}",
        f"main.device={args.device}",
        f"models.clip.type={args.clip_model}",
        f"models.clip.checkpoint={args.clip_pretrained}",
        f"models.sam.type={args.sam_type}",
        f"models.sam.checkpoint={args.sam_checkpoint.resolve()}",
        f"models.sam.points_per_side={args.sam_points_per_side}",
        f"models.sam.points_per_batch={args.sam_points_per_batch}",
        f"pipeline.skip_frames=1",
        f"pipeline.voxel_size={args.voxel_size}",
        f"pipeline.merge_type={args.merge_type}",
        "pipeline.save_intermediate_results=false",
    ]
    env = _hovsg_subprocess_env(args)
    print("running HOV-SG:")
    print(" ".join(command))
    subprocess.run(command, cwd=args.hovsg_root, env=env, check=True)


def export_minimal_package(
    *,
    package_dir: Path,
    native_result_path: Path,
    layout_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    layout_summary: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    _require_hovsg_result(native_result_path)
    _reset_dir(package_dir)
    for directory in ("memory", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / directory).mkdir(parents=True, exist_ok=True)

    objects = _load_native_objects(native_result_path)
    features, feature_warning = _load_mask_features(native_result_path / "mask_feats.pt")
    if features is not None:
        np.save(package_dir / "memory" / "object_features.npy", features)
        labels = _classify_labels(features, args) if not args.no_classify_labels else None
        if labels:
            for obj in objects:
                feature_index = int(obj["feature_index"])
                if feature_index < len(labels):
                    obj["label"] = labels[feature_index]["label"]
                    obj["label_score"] = labels[feature_index]["score"]

    _write_jsonl(package_dir / "memory" / "object_table.jsonl", objects)
    _write_jsonl(
        package_dir / "evidence" / "object_native_paths.jsonl",
        [
            {
                "object_id": obj["object_id"],
                "native_ply_path": obj["native_ply_path"],
                "feature_index": obj["feature_index"],
            }
            for obj in objects
        ],
    )
    _write_json(
        package_dir / "raw_links" / "native_sources.json",
        {
            "scannetpp_root": str(args.scannetpp_root),
            "hovsg_root": str(args.hovsg_root),
            "layout_dir": str(layout_dir),
            "native_result_path": str(native_result_path),
            "source_scene": args.scene_id,
        },
    )
    _write_tool_files(package_dir)
    _write_package_schemas(package_dir)
    _write_schema_md(package_dir, args, native_result_path, features is not None)
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        native_result_path=native_result_path,
        layout_dir=layout_dir,
        object_count=len(objects),
        has_features=features is not None,
        layout_summary=layout_summary,
    )
    _write_capabilities(package_dir)
    warnings = []
    if feature_warning:
        warnings.append(feature_warning)
    _write_build_log(
        package_dir=package_dir,
        args=args,
        started_at=started_at,
        object_count=len(objects),
        warnings=warnings,
    )

    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))
    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "native_result_path": str(native_result_path),
        "object_count": len(objects),
        "has_features": features is not None,
        "validation": report.to_json(),
    }


def _preflight_hovsg(args: argparse.Namespace) -> None:
    if not args.hovsg_root.exists():
        raise FileNotFoundError(f"HOV-SG repo not found: {args.hovsg_root}")
    if not args.hovsg_python.exists():
        raise FileNotFoundError(f"Python executable not found: {args.hovsg_python}")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {args.sam_checkpoint}. "
            "Pass --sam-checkpoint or download the checkpoint before running HOV-SG."
        )
    if not args.no_classify_labels:
        validate_detector_class_list(args.class_names)
    if args.device != "cuda":
        raise ValueError(
            "This HOV-SG smoke runner currently requires --device cuda because "
            "the upstream HOV-SG feature extraction code still contains direct "
            ".cuda() calls. Use --skip-hovsg-run with an existing native result, "
            "or run on a machine with working CUDA/cuDNN."
        )
    if not args.skip_cuda_preflight:
        _preflight_cuda(args)


def _read_prepared_layout_summary(layout_dir: Path, scene_id: str) -> dict[str, Any]:
    if not layout_dir.exists():
        raise FileNotFoundError(f"prepared HOV-SG layout directory does not exist: {layout_dir}")
    summary_path = layout_dir / "layout_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"prepared HOV-SG layout is missing {summary_path}. "
            "Run scripts/methods/hovsg/prepare_eval_layout.py first and only "
            "start the HOV-SG build after that script finishes."
        )
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    if not isinstance(summary, dict):
        raise ValueError(f"expected JSON object in {summary_path}")
    if summary.get("scene_id") not in (None, scene_id):
        raise ValueError(
            f"prepared layout scene_id mismatch: expected {scene_id}, "
            f"found {summary.get('scene_id')}"
        )
    _require_prepared_layout_files(layout_dir)
    summary["layout_dir"] = str(layout_dir)
    return summary


def _require_prepared_layout_files(layout_dir: Path) -> None:
    missing = [
        path
        for path in (
            layout_dir / "color",
            layout_dir / "depth",
            layout_dir / "pose",
            layout_dir / "intrinsic" / "intrinsic_color.txt",
            layout_dir / "intrinsic" / "intrinsic_depth.txt",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("prepared HOV-SG layout is incomplete:\n" + "\n".join(str(path) for path in missing))
    color_count = sum(1 for _ in (layout_dir / "color").glob("*.jpg"))
    depth_count = sum(1 for _ in (layout_dir / "depth").glob("*.png"))
    pose_count = sum(1 for _ in (layout_dir / "pose").glob("*.txt"))
    if color_count == 0 or depth_count == 0 or pose_count == 0:
        raise ValueError(f"prepared HOV-SG layout has empty color/depth/pose folders: {layout_dir}")
    if len({color_count, depth_count, pose_count}) != 1:
        raise ValueError(
            "prepared HOV-SG layout has mismatched frame counts: "
            f"color={color_count}, depth={depth_count}, pose={pose_count}"
        )


def _hovsg_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.hovsg_root.resolve()),
            str(REPO_ROOT),
            str(args.claws_root.resolve()),
            env.get("PYTHONPATH", ""),
        ]
    )
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


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
        [str(args.hovsg_python), "-c", code],
        cwd=args.hovsg_root,
        env=_hovsg_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    details = "\n".join(
        part
        for part in (
            "stdout:",
            result.stdout.strip(),
            "stderr:",
            result.stderr.strip(),
        )
        if part
    )
    raise RuntimeError(
        "CUDA/cuDNN preflight failed before launching HOV-SG. "
        "The native HOV-SG smoke build needs a working CUDA device because "
        "HOV-SG currently hard-codes .cuda() in its feature extraction path. "
        "Run on a GPU node with a healthy NVIDIA driver, set "
        "--cuda-visible-devices to a valid GPU if needed, or use "
        "--skip-hovsg-run with --native-result-path to package an existing "
        f"HOV-SG output.\n{details}"
    )


def _require_hovsg_result(path: Path) -> None:
    missing = [
        item
        for item in (path / "mask_feats.pt", path / "objects", path / "full_pcd.ply", path / "masked_pcd.ply")
        if not item.exists()
    ]
    if missing:
        raise FileNotFoundError("HOV-SG result is incomplete:\n" + "\n".join(str(item) for item in missing))


def _load_native_objects(native_result_path: Path) -> list[dict[str, Any]]:
    objects = []
    for ordinal, path in enumerate(sorted((native_result_path / "objects").glob("*.ply"), key=_natural_key)):
        feature_index = _feature_index(path, ordinal)
        bbox, center, num_points = _read_ply_bounds(path)
        objects.append(
            {
                "object_id": path.stem,
                "label": "object",
                "aliases": [],
                "position_3d": center,
                "bbox_3d": bbox,
                "confidence": None,
                "label_score": None,
                "num_points": num_points,
                "feature_index": feature_index,
                "source_artifacts": ["memory/object_table.jsonl"],
                "native_ply_path": str(path),
                "evidence": [
                    {
                        "source_type": "native_ply_index",
                        "source_path": "evidence/object_native_paths.jsonl",
                        "notes": "Package-local index points to the native HOV-SG object PLY.",
                    }
                ],
            }
        )
    if not objects:
        raise ValueError(f"no HOV-SG objects found under {native_result_path / 'objects'}")
    return objects


def _load_mask_features(path: Path) -> tuple[Optional[np.ndarray], Optional[str]]:
    try:
        import torch

        loaded = torch.load(path, map_location="cpu")
        if hasattr(loaded, "detach"):
            features = loaded.detach().cpu().float().numpy()
        else:
            features = np.asarray(loaded, dtype=np.float32)
        if features.ndim != 2:
            return None, f"mask_feats.pt has unexpected shape {features.shape}"
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return features.astype(np.float32), None
    except Exception as exc:
        return None, f"could not export mask features: {exc}"


def _classify_labels(features: np.ndarray, args: argparse.Namespace) -> Optional[list[dict[str, Any]]]:
    if not args.class_names.exists():
        return None
    try:
        import open_clip
        import torch
    except Exception:
        return None

    labels = [line.strip() for line in args.class_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        return None
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
        device=device,
    )
    model.eval()
    prompts = [f"a photo of a {label}" for label in labels]
    with torch.no_grad():
        tokens = open_clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    text_np = text_features.detach().cpu().numpy().astype(np.float32)
    if text_np.shape[1] != features.shape[1]:
        return None
    scores = features @ text_np.T
    best = np.argmax(scores, axis=1)
    return [
        {"label": labels[index], "score": float(scores[row, index])}
        for row, index in enumerate(best.tolist())
    ]


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
    predictions = _rank_by_label_and_size(objects, target_label, query_text)[:top_k]
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
    return re.sub(r"\s+", " ", text).strip()


def _rank_by_label_and_size(
    objects: list[dict[str, Any]],
    target_label: str,
    query_text: str,
) -> list[dict[str, Any]]:
    ranked = []
    target_tokens = set(target_label.split())
    query_tokens = set(query_text.split())
    for obj in objects:
        label = _normalize_label(obj.get("label") or "object")
        label_tokens = set(label.split())
        if target_label and target_label == label:
            score = 1.0
        elif target_label and (target_label in label or label in target_label):
            score = 0.9
        elif target_tokens and target_tokens & label_tokens:
            score = 0.75
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


def _write_schema_md(package_dir: Path, args: argparse.Namespace, native_result_path: Path, has_features: bool) -> None:
    (package_dir / "schema.md").write_text(
        f"""# HOV-SG Minimal Memory Schema

Coordinate frame and units: object positions and bounding boxes use the HOV-SG
world coordinate frame produced from ScanNet++ aligned camera poses. Units are
meters.

Object id format: object ids follow the native HOV-SG object PLY stem, such as
`pcd_0` or `pcd_143`.

Timestamp format: this smoke package exports static object memory, so object
timestamps are not present. Source frame indices are recorded in
`raw_links/native_sources.json`.

Relation representation: this smoke package does not export room or relation
edges. It only exports object-level geometry from HOV-SG semantic segmentation.

Confidence or score meaning: `label_score` is a CLIP text similarity when label
classification succeeds. It can be null when labels are not classified.
`confidence` is null because HOV-SG object PLYs do not expose one scalar object
confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `memory/object_features.npy` is present only when
`mask_feats.pt` was readable during export; present={has_features}. The native
HOV-SG result directory is `{native_result_path}`.

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
    native_result_path: Path,
    layout_dir: Path,
    object_count: int,
    has_features: bool,
    layout_summary: dict[str, Any],
) -> None:
    _write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.1",
            "package_id": f"hovsg/scannetpp/{args.scene_id}/{run_id}",
            "method": {
                "name": "hovsg",
                "display_name": "HOV-SG",
                "family": "object_map",
                "repo_path": str(args.hovsg_root),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": "scannetpp",
                "split": "smoke",
                "scene_id": args.scene_id,
                "episode_id": None,
            },
            "input": {
                "modality": ["rgb", "depth", "pose", "intrinsics"],
                "frame_count": int(layout_summary.get("frame_count") or 0),
                "rgbd_root": str(layout_dir),
                "poses_path": str(layout_dir / "pose"),
                "intrinsics_path": str(layout_dir / "intrinsic"),
                "timestamp_path": None,
                "coordinate_frame": "HOV-SG world frame from ScanNet++ aligned poses; meters",
            },
            "vocabulary": {
                "vocabulary_mode": "closed",
                "class_list_path": str(args.class_names),
                "source": "shared_modules",
                "profile": args.shared_module_profile,
            },
            "modules": shared_modules_metadata(args),
            "explicit_memory": True,
            "memory_artifacts": [
                {
                    "name": "object_table",
                    "type": "jsonl",
                    "path": "memory/object_table.jsonl",
                    "description": f"HOV-SG object inventory with {object_count} objects.",
                    "required_for": ["track1_memory_construction", "track2_object_location"],
                }
            ]
            + (
                [
                    {
                        "name": "object_features",
                        "type": "npy",
                        "path": "memory/object_features.npy",
                        "description": "Normalized HOV-SG mask features exported from mask_feats.pt.",
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
                    "description": "Mapping from package object ids to native HOV-SG object PLY paths.",
                    "required_for": [],
                }
            ],
            "raw_links": [
                {
                    "name": "native_hovsg_result",
                    "type": "directory",
                    "path": str(native_result_path),
                    "description": "Native HOV-SG output directory.",
                    "required_for": [],
                },
                {
                    "name": "prepared_hovsg_layout",
                    "type": "directory",
                    "path": str(layout_dir),
                    "description": "Sampled ScanNet-style RGB-D layout used by HOV-SG.",
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
                    "description": "Smoke object query over exported labels and object sizes.",
                    "required_for": ["track2_object_location"],
                },
            ],
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [],
                "environment": str(args.hovsg_python),
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
            "notes": "Smoke package built from HOV-SG native semantic segmentation output.",
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
                "mode": "agentic_full_access",
                "read_manifest": True,
                "read_schema": True,
                "read_memory_artifacts": True,
                "read_evidence": True,
                "read_adapter_code": True,
                "read_shared_module_code": True,
                "read_method_root_source_code": True,
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
            "source_outputs": [],
            "object_count": object_count,
            "shared_modules": shared_modules_metadata(args),
            "hovsg_runtime": {
                "class_names": str(args.class_names),
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "sam_type": args.sam_type,
                "sam_checkpoint": str(args.sam_checkpoint),
                "sam_points_per_side": args.sam_points_per_side,
                "sam_points_per_batch": args.sam_points_per_batch,
                "voxel_size": args.voxel_size,
                "merge_type": args.merge_type,
                "no_classify_labels": args.no_classify_labels,
                "safe_crop_patch": not args.disable_safe_crop_patch,
                "safe_crop_patch_entrypoint": (
                    None
                    if args.disable_safe_crop_patch
                    else "scripts/methods/hovsg/run_semantic_segmentation_patched.py"
                ),
            },
            "warnings": warnings,
        },
    )


def _read_ply_bounds(path: Path) -> tuple[list[float], list[float], int]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        fmt = _ply_format(header_lines)
        vertex_count, properties = _ply_vertex_layout(header_lines)
        if vertex_count <= 0:
            raise ValueError(f"PLY has no vertices: {path}")
        if fmt == "binary_little_endian":
            dtype = np.dtype([(_safe_prop_name(name), _ply_dtype(kind)) for kind, name in properties])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            points = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
        elif fmt == "ascii":
            rows = []
            for _ in range(vertex_count):
                rows.append([float(value) for value in f.readline().decode("ascii").split()[: len(properties)]])
            array = np.asarray(rows, dtype=np.float64)
            prop_names = [_safe_prop_name(name) for _, name in properties]
            points = array[:, [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]]
        else:
            raise ValueError(f"unsupported PLY format {fmt!r}: {path}")

    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) == 0:
        raise ValueError(f"PLY has no finite XYZ points: {path}")
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2.0
    return min_bound.tolist() + max_bound.tolist(), center.tolist(), int(len(points))


def _ply_format(header_lines: Iterable[str]) -> str:
    for line in header_lines:
        if line.startswith("format "):
            return line.split()[1]
    raise ValueError("PLY format is missing")


def _ply_vertex_layout(header_lines: list[str]) -> tuple[int, list[tuple[str, str]]]:
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and parts[0] == "property" and len(parts) >= 3:
            properties.append((parts[1], parts[2]))
    prop_names = {name for _, name in properties}
    if not {"x", "y", "z"} <= prop_names:
        raise ValueError("PLY vertex properties must include x, y, z")
    return vertex_count, properties


def _ply_dtype(kind: str) -> str:
    return {
        "double": "<f8",
        "float": "<f4",
        "float32": "<f4",
        "float64": "<f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int8": "i1",
        "ushort": "<u2",
        "uint16": "<u2",
        "short": "<i2",
        "int16": "<i2",
        "uint": "<u4",
        "uint32": "<u4",
        "int": "<i4",
        "int32": "<i4",
    }[kind]


def _safe_prop_name(name: str) -> str:
    return name if name not in {"class"} else "class_"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in matrix:
            f.write(" ".join(f"{float(value):.10g}" for value in row) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _iso_time(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat()


def _frame_sort_key(frame_id: str) -> int:
    try:
        return int(frame_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _natural_key(path: Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _feature_index(path: Path, ordinal: int) -> int:
    import re

    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else ordinal


def _video_shape(cap: Any) -> Optional[tuple[int, int]]:
    width = int(cap.get(3))
    height = int(cap.get(4))
    if width > 0 and height > 0:
        return height, width
    return None


def _scale_intrinsic(matrix: np.ndarray, source_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    scaled = matrix.astype(np.float64).copy()
    scaled[0, :] *= dst_w / src_w
    scaled[1, :] *= dst_h / src_h
    return scaled


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
