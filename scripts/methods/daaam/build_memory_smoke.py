"""Build a DAAAM minimal memory package for Track 1/2 smoke tests.

Runbook (local DAAAM smoke):

  # Prepare ScanNet++ RGB-D layout only (no DAAAM run):
  python scripts/methods/daaam/build_memory_smoke.py --prepare-only \
      --max-frames 5 --frame-stride 50

  # Raw build route (needs a DAAAM runtime that can finish segmentation +
  # DAM grounding). Point --daaam-python at the DAAAM conda env:
  DAAAM_PYTHON=/home/robin_wang/miniforge3/envs/daaam/bin/python \
  python scripts/methods/daaam/build_memory_smoke.py \
      --daaam-python /home/robin_wang/miniforge3/envs/daaam/bin/python

  # Package an existing native DAAAM output (DSG already on disk):
  python scripts/methods/daaam/build_memory_smoke.py \
      --skip-daaam-run --native-output-dir /path/to/daaam/output

Required local env for a native build:
  - --daaam-python: a DAAAM conda env that imports spark_dsg, daaam, open_clip,
    sentence_transformers, torch, ultralytics, segment_anything, boxmot, and
    daaam.grounding.workers.dam_grounding. PYTHONPATH=$DAAAM/src is set for the
    subprocess automatically. These are runtime deps, not NAS checkpoints.
  - --hydra-config-path: defaults to the colcon_ws clio_dataset_khronos.yaml
    when present.
  - SAM/YOLO/ReID/DAM model weights are shared-module/NAS artifacts.

Adapter-applied DAAAM config overrides (DAAAM repo is never edited):
  - segmentation.imgsz cleared (SAM-ViT route would otherwise crash on an
    injected fastsam_imgsz kwarg); pass --keep-segmenter-imgsz for FastSAM.
  - tracking.with_reid=false by default (native ReID .engine is usually absent);
    pass --with-reid with --reid-weights to re-enable.

Known native blocker (2026-06-17): the DAM grounding worker imports
gradio/fastapi and loads multi-GB nvidia/DAM-3B; until those exist in the DAAAM
env, no DSG (hence no Track 1/2 memory) can be produced. See
.codex/baseline_registry.md (DAAAM smoke finish status) for details.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.shared_modules import (  # noqa: E402
    add_shared_module_args,
    apply_daaam_shared_modules,
    shared_modules_metadata,
)
from spatial_memory_evaluation.common.build_accounting import (  # noqa: E402
    write_build_log_with_accounting,
)
from spatial_memory_evaluation.common.labels import (  # noqa: E402
    DEFAULT_LABEL_ALIASES,
    normalize_label,
    read_detector_class_list,
    validate_detector_class_list,
)
from spatial_memory_evaluation.memory_package_validator import validate_package  # noqa: E402


DEFAULT_DAAAM_ROOT = Path("/home/robin_wang/DAAAM")
DEFAULT_DAAAM_PYTHON = Path(os.environ.get("DAAAM_PYTHON", sys.executable))
DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SENTENCE_EMBEDDING_MODEL = None
DEFAULT_DAAAM_CONFIG = DEFAULT_DAAAM_ROOT / "config" / "pipeline_config.yaml"
DEFAULT_SAM_MODEL_CONFIG = DEFAULT_DAAAM_ROOT / "config" / "sam" / "sam_vit_config.yaml"
DEFAULT_HYDRA_CONFIG = Path(
    "/home/robin_wang/daaam_colcon_ws/src/daaam_ros/config/hydra_config/clio_dataset_khronos.yaml"
)
DEFAULT_SHARED_MODULES_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/modules")
DEFAULT_NATIVE_FASTSAM_MODEL = "fastsam/FastSAM-x-640x480.engine"
DEFAULT_NATIVE_FASTSAM_CONFIG = "fastsam/fastsam_config.yaml"
ADAPTER_WARNINGS_NAME = "adapter_warnings.json"
DAAAM_ENV_HINT = (
    "DAAAM Python imports such as spark_dsg, daaam, open_clip, "
    "sentence_transformers, torch, ultralytics, segment_anything, and cvxpy are "
    "environment dependencies. Install them in the conda env used by "
    "--daaam-python; do not treat them as NAS/model-checkpoint files."
)
DAAAM_ARTIFACT_HINT = (
    "Model artifacts/checkpoints such as SAM, YOLO-World, FastSAM, ReID "
    "weights/engines, and optional HF/OpenCLIP caches should be centralized "
    f"under shared modules/NAS, e.g. {DEFAULT_SHARED_MODULES_ROOT}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DAAAM minimal memory package for Track 1/2 smoke tests. "
            "The script can package an existing DAAAM DSG output or prepare "
            "ScanNet++ RGB-D data and launch DAAAM without modifying the DAAAM repo."
        )
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--daaam-root", type=Path, default=DEFAULT_DAAAM_ROOT)
    parser.add_argument(
        "--daaam-python",
        type=Path,
        default=DEFAULT_DAAAM_PYTHON,
        help="Python executable for the DAAAM runtime environment.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_DAAAM_CONFIG)
    parser.add_argument(
        "--hydra-config-path",
        type=Path,
        default=DEFAULT_HYDRA_CONFIG if DEFAULT_HYDRA_CONFIG.exists() else None,
        help=(
            "Hydra config YAML passed to DAAAM run_pipeline.py for raw builds. "
            f"Defaults to {DEFAULT_HYDRA_CONFIG} when present."
        ),
    )
    parser.add_argument("--sam-model-config-path", type=Path, default=DEFAULT_SAM_MODEL_CONFIG)
    parser.add_argument("--sentence-embedding-model", default=DEFAULT_SENTENCE_EMBEDDING_MODEL)
    parser.add_argument(
        "--dam-model-path",
        type=Path,
        default=None,
        help="DAAAM DAM model snapshot path; defaults to shared_modules NAS snapshot.",
    )
    parser.add_argument(
        "--daaam-segmenter",
        choices=("shared_sam", "native_fastsam_trt"),
        default="shared_sam",
        help=(
            "DAAAM segmentation route. shared_sam uses the shared SAM checkpoint "
            "for benchmark consistency; native_fastsam_trt uses DAAAM's native "
            "FastSAM/TensorRT path for realtime/async smoke experiments."
        ),
    )
    parser.add_argument(
        "--native-fastsam-model",
        type=Path,
        default=None,
        help=(
            "DAAAM-native FastSAM checkpoint/engine, relative to "
            "$DAAAM_ROOT/checkpoints or absolute. Used only with "
            "--daaam-segmenter native_fastsam_trt."
        ),
    )
    parser.add_argument(
        "--native-fastsam-config-path",
        type=Path,
        default=DEFAULT_NATIVE_FASTSAM_CONFIG,
        help=(
            "DAAAM-native FastSAM config, relative to $DAAAM_ROOT/config or "
            "absolute. Used only with --daaam-segmenter native_fastsam_trt."
        ),
    )
    parser.add_argument("--semantic-config", type=Path, default=None)
    parser.add_argument("--labelspace-colors", type=Path, default=None)
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path("data/daaam_layouts"),
        help="where prepared DAAAM RGB-D layouts are written",
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help="existing DAAAM layout containing rgb/, depth/, pose/, camera_info.json",
    )
    parser.add_argument(
        "--native-output-root",
        type=Path,
        default=Path("data/daaam_native"),
        help="where DAAAM native output is written for raw builds",
    )
    parser.add_argument(
        "--native-output-dir",
        type=Path,
        default=None,
        help="existing DAAAM output directory containing dsg_updated.json or dsg.json",
    )
    parser.add_argument("--package-root", type=Path, default=Path("memories"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="maximum sampled frames for smoke; use 0 for all frames selected by stride",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help=(
            "optional DAAAM processing FPS. If unset, the adapter uses "
            "--no-throttle. Use a low value when DAM grounding worker model "
            "loading needs more time than the short smoke frame sequence."
        ),
    )
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--skip-cuda-preflight", action="store_true")
    parser.add_argument("--skip-dependency-preflight", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-layout-export", action="store_true")
    parser.add_argument(
        "--skip-daaam-run",
        action="store_true",
        help="skip native DAAAM run and package --native-output-dir or the computed native dir",
    )
    parser.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="do not run DAAAM postprocess_scene_graph.py after a native run",
    )
    parser.add_argument(
        "--strict-native-run",
        action="store_true",
        help=(
            "fail immediately if DAAAM run_pipeline.py exits nonzero. By default "
            "the smoke adapter can package recoverable partial native output when "
            "a current-run DSG exists."
        ),
    )
    parser.add_argument(
        "--native-verbose",
        action="store_true",
        help="pass --verbose to DAAAM run_pipeline.py so native exceptions print tracebacks",
    )
    parser.add_argument(
        "--skip-track2-index",
        action="store_true",
        help="do not try to build the deterministic DAAAM semantic query index",
    )
    parser.add_argument(
        "--config-overrides",
        action="append",
        default=None,
        help="extra DAAAM config override key=value; repeatable",
    )
    parser.add_argument(
        "--reid-weights",
        default=None,
        help=(
            "DAAAM BotSort ReID weights, relative to the DAAAM repo or absolute. "
            "ReID is disabled by default for smoke because the native TensorRT "
            "engine (checkpoints/reid_weights/clip_general.engine) is a "
            "machine-specific artifact that is usually absent."
        ),
    )
    parser.add_argument(
        "--with-reid",
        action="store_true",
        help=(
            "enable DAAAM BotSort ReID; requires a real --reid-weights file. "
            "Off by default so the smoke build does not crash on a missing "
            "ReID engine."
        ),
    )
    parser.add_argument(
        "--keep-segmenter-imgsz",
        action="store_true",
        help=(
            "do not clear DAAAM segmentation.imgsz. By default the adapter "
            "clears it because the shared SAM-ViT checkpoint path triggers "
            "DAAAM to inject an unsupported 'fastsam_imgsz' kwarg into "
            "SamAutomaticMaskGenerator. Only keep it for a FastSAM engine run."
        ),
    )
    parser.add_argument(
        "--dataset-tag",
        default="scannetpp",
        help="Dataset path segment for outputs (memories/daaam/<tag>/<scene>, "
        "native key <tag>_<scene>). Use 'scannet' for ScanNet .sens scenes.",
    )
    parser.add_argument("--class-names", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-type", default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--clip-pretrained", default=None)
    add_shared_module_args(parser)
    args = parser.parse_args()
    apply_daaam_shared_modules(args)
    _normalize_repo_relative_paths(args)
    return args


def _repo_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _normalize_repo_relative_paths(args: argparse.Namespace) -> None:
    for name in (
        "layout_root",
        "layout_dir",
        "native_output_root",
        "native_output_dir",
        "package_root",
        "semantic_config",
        "labelspace_colors",
        "dam_model_path",
    ):
        setattr(args, name, _repo_path(getattr(args, name)))


def main(args: argparse.Namespace) -> int:
    started = time.time()
    if args.prepare_only and args.native_output_dir is not None:
        raise ValueError("--prepare-only is only valid for raw ScanNet++ layout preparation")

    run_id = args.run_id or (
        args.native_output_dir.name if args.native_output_dir is not None else f"daaam-smoke-{_run_timestamp()}"
    )
    dataset_tag = getattr(args, "dataset_tag", None) or "scannetpp"
    scene_key = f"{dataset_tag}_{args.scene_id}"
    layout_dir = args.layout_dir or (args.layout_root / scene_key / run_id)
    native_output_dir = args.native_output_dir or (args.native_output_root / scene_key / run_id)
    package_dir = args.package_root / "daaam" / dataset_tag / args.scene_id / run_id

    layout_summary: dict[str, Any] = {}
    if args.native_output_dir is None:
        if args.layout_dir is not None or args.skip_layout_export:
            layout_summary = _read_prepared_layout_summary(layout_dir)
        else:
            layout_summary = export_daaam_scannetpp_layout(
                scannetpp_root=args.scannetpp_root,
                claws_root=args.claws_root,
                scene_id=args.scene_id,
                layout_dir=layout_dir,
                frame_stride=args.frame_stride,
                max_frames=None if args.max_frames == 0 else args.max_frames,
            )
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "layout": layout_summary}, indent=2))
            return 0
        if not args.skip_daaam_run:
            _preflight_daaam(args, require_native_run=True)
            run_daaam_native(args, layout_dir=layout_dir, native_output_dir=native_output_dir)
            if not args.skip_postprocess:
                _run_daaam_postprocess(args, native_output_dir)
        else:
            _preflight_daaam(args, require_native_run=False)
    else:
        _preflight_daaam(args, require_native_run=False)

    export_summary = export_minimal_package(
        package_dir=package_dir,
        native_output_dir=native_output_dir,
        layout_dir=layout_dir if layout_dir.exists() else None,
        args=args,
        run_id=run_id,
        layout_summary=layout_summary,
        started_at=started,
    )
    print(json.dumps(export_summary, indent=2))
    return 0


def export_daaam_scannetpp_layout(
    *,
    scannetpp_root: Path,
    claws_root: Path,
    scene_id: str,
    layout_dir: Path,
    frame_stride: int,
    max_frames: Optional[int],
) -> dict[str, Any]:
    from scripts.methods.hovsg.build_memory_smoke import export_scannetpp_iphone_layout

    _reset_dir(layout_dir)
    scannet_layout = layout_dir / "_scannet_export"
    summary = export_scannetpp_iphone_layout(
        scannetpp_root=scannetpp_root,
        claws_root=claws_root,
        scene_id=scene_id,
        output_dir=scannet_layout,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    shutil.copytree(scannet_layout / "color", layout_dir / "rgb")
    shutil.copytree(scannet_layout / "depth", layout_dir / "depth")
    shutil.copytree(scannet_layout / "pose", layout_dir / "pose")
    shutil.copytree(scannet_layout / "intrinsic", layout_dir / "intrinsic")
    _write_camera_info(
        layout_dir / "camera_info.json",
        intrinsic_path=layout_dir / "intrinsic" / "intrinsic_color.txt",
        rgb_dir=layout_dir / "rgb",
    )
    layout_summary = {
        **summary,
        "layout_dir": str(layout_dir),
        "daaam_layout_dir": str(layout_dir),
        "rgb_dir": str(layout_dir / "rgb"),
        "depth_dir": str(layout_dir / "depth"),
        "pose_dir": str(layout_dir / "pose"),
        "camera_info_path": str(layout_dir / "camera_info.json"),
        "depth_scale": 1000.0,
    }
    _write_json(layout_dir / "layout_summary.json", layout_summary)
    return layout_summary


def run_daaam_native(args: argparse.Namespace, *, layout_dir: Path, native_output_dir: Path) -> None:
    native_output_dir.mkdir(parents=True, exist_ok=True)
    semantic_config = args.semantic_config or (native_output_dir / "labels_pseudo.yaml")
    labelspace_colors = args.labelspace_colors or (native_output_dir / "labels_pseudo.csv")
    semantic_config.parent.mkdir(parents=True, exist_ok=True)
    labelspace_colors.parent.mkdir(parents=True, exist_ok=True)
    segmenter_model_path, segmenter_config_path = _daaam_segmenter_paths(args, native_output_dir)

    command = [
        str(args.daaam_python),
        str(REPO_ROOT / "scripts" / "methods" / "daaam" / "run_pipeline_patched.py"),
        str(layout_dir),
        "--config",
        str(args.config),
        "--dataset-type",
        "ImageSequenceDataset",
        "--output-dir",
        str(native_output_dir),
        "--no-logging",
        "--depth-scale",
        "1000.0",
        "--sam-model",
        str(segmenter_model_path),
        "--sam-model-config-path",
        str(segmenter_config_path),
        "--sentence-embedding-model",
        str(args.sentence_embedding_model),
        "--semantic-config",
        str(semantic_config),
        "--labelspace-path",
        str(semantic_config),
        "--labelspace-colors",
        str(labelspace_colors),
        "--hydra-config-path",
        str(args.hydra_config_path),
    ]
    if args.target_fps is None:
        command.append("--no-throttle")
    else:
        command.extend(["--target-fps", str(args.target_fps)])
    if args.native_verbose:
        command.append("--verbose")
    if args.max_frames and args.max_frames > 0:
        command.extend(["--max-frames", str(args.max_frames)])
    overrides = [
        f"workers.dam_grounding_config.selectframe_clip_model_name={args.clip_model}",
        f"workers.dam_grounding_config.selectframe_clip_model_dataset={args.clip_pretrained}",
        "workers.dam_grounding_config.selectframe_clip_backend=openclip",
        f"workers.dam_grounding_config.dam_model_path={args.dam_model_path}",
        f"grounding.perframe_clip_model_name={args.clip_model}",
        f"grounding.perframe_clip_model_dataset={args.clip_pretrained}",
    ]
    overrides.extend(_segmenter_config_overrides(args))
    overrides.extend(_tracking_config_overrides(args))
    for override in args.config_overrides or []:
        overrides.append(override)
    command.extend(["--config-overrides", *overrides])

    print("running DAAAM:")
    print(" ".join(command))
    run_started_at = time.time()
    result = subprocess.run(command, cwd=args.daaam_root, env=_daaam_subprocess_env(args), check=False)
    _collect_daaam_native_outputs(args, native_output_dir, run_started_at=run_started_at)
    if result.returncode != 0:
        warning = (
            f"DAAAM native run exited with code {result.returncode}. "
            "The smoke adapter normalized current-run artifacts and will recover "
            "only if a DSG JSON exists. Re-run with --native-verbose for a native traceback."
        )
        if args.strict_native_run:
            raise RuntimeError(warning + " --strict-native-run was set.")
        try:
            dsg_path = _find_dsg_path(native_output_dir)
        except FileNotFoundError as exc:
            raise RuntimeError(
                warning
                + "\nNo recoverable current-run DSG JSON was found after the native failure."
            ) from exc
        warning = warning + f" Recovering from partial native output using {dsg_path}."
        print(f"warning: {warning}", file=sys.stderr)
        _append_adapter_warning(native_output_dir, warning)


def _daaam_segmenter_paths(args: argparse.Namespace, native_output_dir: Path) -> tuple[Path, Path]:
    if args.daaam_segmenter == "native_fastsam_trt":
        return _resolve_native_fastsam_model(args), _resolve_daaam_config_path(
            args.daaam_root,
            args.native_fastsam_config_path,
        )
    return _materialize_daaam_sam_checkpoint(args, native_output_dir), Path(args.sam_model_config_path)


def _resolve_native_fastsam_model(args: argparse.Namespace) -> Path:
    value = args.native_fastsam_model or Path(DEFAULT_NATIVE_FASTSAM_MODEL)
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(args.daaam_root) / "checkpoints" / path


def _resolve_daaam_config_path(daaam_root: Path, value: Path | str | None) -> Path:
    path = Path(value or DEFAULT_NATIVE_FASTSAM_CONFIG)
    if path.is_absolute():
        return path
    return Path(daaam_root) / "config" / path


def _segmenter_config_overrides(args: argparse.Namespace) -> list[str]:
    """Adapter-side DAAAM segmentation overrides for the shared SAM-ViT route.

    DAAAM's pipeline_config.yaml ships ``segmentation.imgsz: [480, 640]`` for a
    FastSAM TensorRT engine. ``UniversalSegmenter`` then unconditionally injects
    ``fastsam_imgsz`` into ``SamAutomaticMaskGenerator(**model_config)``
    (``src/daaam/utils/segmentation.py``), which the SAM-ViT mask generator does
    not accept and raises ``unexpected keyword argument 'fastsam_imgsz'``. Since
    the shared SAM route supplies a ``sam_vit`` checkpoint, clear ``imgsz`` so
    the SAM-ViT path receives only valid kwargs. The native FastSAM/TensorRT
    route keeps ``imgsz`` from DAAAM's config.
    """
    if args.daaam_segmenter == "native_fastsam_trt" or args.keep_segmenter_imgsz:
        return []
    return ["segmentation.imgsz="]


def _tracking_config_overrides(args: argparse.Namespace) -> list[str]:
    """Adapter-side DAAAM tracking overrides.

    DAAAM defaults to ``tracking.with_reid: true`` with a machine-specific
    TensorRT ReID engine (``checkpoints/reid_weights/clip_general.engine``) that
    is typically absent. Disable ReID for smoke unless the caller explicitly
    opts in with ``--with-reid`` and provides ``--reid-weights``.
    """
    overrides: list[str] = []
    if args.reid_weights:
        overrides.append(f"tracking.reid_weights={args.reid_weights}")
    overrides.append(f"tracking.with_reid={'true' if args.with_reid else 'false'}")
    return overrides


def _resolve_reid_weights(args: argparse.Namespace) -> Path | None:
    """Resolve --reid-weights against the DAAAM repo root like DAAAM does."""
    if not args.reid_weights:
        return None
    path = Path(args.reid_weights)
    if path.is_absolute():
        return path
    return Path(args.daaam_root) / path


def export_minimal_package(
    *,
    package_dir: Path,
    native_output_dir: Path,
    layout_dir: Path | None,
    args: argparse.Namespace,
    run_id: str,
    layout_summary: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    validate_detector_class_list(args.class_names)
    dsg_path = _find_dsg_path(native_output_dir)
    _reset_dir(package_dir)
    for directory in ("memory", "memory/native", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / directory).mkdir(parents=True, exist_ok=True)

    native_dsg_path = package_dir / "memory" / "native" / dsg_path.name
    shutil.copy2(dsg_path, native_dsg_path)
    native_artifacts = [dsg_path]
    for name in (
        "background_objects.yaml",
        "corrections.yaml",
        "region_summaries.yaml",
        "object_positions.json",
        "correction_stats.json",
        ADAPTER_WARNINGS_NAME,
    ):
        source = _find_native_artifact(native_output_dir, name)
        if source.exists():
            shutil.copy2(source, package_dir / "memory" / "native" / name)
            native_artifacts.append(source)

    extraction = _extract_daaam_memory(
        args=args,
        dsg_path=native_dsg_path,
        output_path=package_dir / "memory" / "native" / "daaam_extraction.json",
    )
    class_labels = read_detector_class_list(args.class_names)
    objects, canonicalization_warnings = _canonicalize_objects(
        extraction,
        class_labels,
        source_key="objects",
        source_artifact="memory/object_table.jsonl",
        evidence_note="Merged object exported from DAAAM/Hydra OBJECTS layer.",
    )
    background_objects, background_warnings = _canonicalize_objects(
        extraction,
        class_labels,
        source_key="background_objects",
        source_artifact="memory/background_object_table.jsonl",
        evidence_note=(
            "Filtered/background object exported from DAAAM BACKGROUND_OBJECTS layer; "
            "kept for debug and agentic evidence, not Track 1/2 fixed API."
        ),
    )
    if not objects and not background_objects:
        raise ValueError(f"DAAAM DSG exported no objects or background objects: {dsg_path}")

    _write_jsonl(package_dir / "memory" / "object_table.jsonl", objects)
    _write_jsonl(package_dir / "memory" / "background_object_table.jsonl", background_objects)
    _write_jsonl(
        package_dir / "evidence" / "object_sources.jsonl",
        [
            {
                "object_id": obj["object_id"],
                "raw_label": obj.get("raw_label"),
                "label": obj.get("label"),
                "semantic_label": obj.get("semantic_label"),
                "is_background": obj.get("is_background"),
                "source_artifacts": obj.get("source_artifacts", []),
                "provenance": obj.get("provenance"),
            }
            for obj in [*objects, *background_objects]
        ],
    )

    track2_status, track2_reason = _write_track2_index(
        package_dir=package_dir,
        extraction=extraction,
        objects=objects,
        class_labels=class_labels,
    )
    _write_tool_files(package_dir, track2_supported=track2_status == "supported")
    _write_package_schemas(package_dir, track2_supported=track2_status == "supported")
    _write_schema_md(package_dir, dsg_path=dsg_path, track2_status=track2_status, track2_reason=track2_reason)
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        native_output_dir=native_output_dir,
        dsg_path=dsg_path,
        native_dsg_path=native_dsg_path,
        object_count=len(objects),
        layout_dir=layout_dir,
        layout_summary=layout_summary,
        track2_status=track2_status,
        background_object_count=len(background_objects),
    )
    _write_capabilities(package_dir, track2_status=track2_status, track2_reason=track2_reason)
    _write_raw_links(package_dir, args=args, native_output_dir=native_output_dir, layout_dir=layout_dir, dsg_path=dsg_path)
    warnings = (
        _read_adapter_warnings(native_output_dir)
        + list(extraction.get("warnings") or [])
        + canonicalization_warnings
        + background_warnings
    )
    _write_build_log(
        package_dir=package_dir,
        args=args,
        started_at=started_at,
        native_output_dir=native_output_dir,
        native_artifacts=native_artifacts,
        object_count=len(objects),
        background_object_count=len(background_objects),
        layout_summary=layout_summary,
        track2_status=track2_status,
        warnings=warnings,
    )

    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))
    return {
        "status": "ok",
        "package_dir": str(package_dir),
        "native_output_dir": str(native_output_dir),
        "native_dsg_path": str(dsg_path),
        "object_count": len(objects),
        "background_object_count": len(background_objects),
        "track2_fixed_api": track2_status,
        "track2_reason": track2_reason,
        "validation": report.to_json(),
        "warnings": warnings,
    }


def _preflight_daaam(args: argparse.Namespace, *, require_native_run: bool) -> None:
    missing = []
    for path, label, kind in (
        (args.daaam_root, "DAAAM root", "repo_or_config"),
        (args.daaam_root / "scripts" / "run_pipeline.py", "DAAAM run_pipeline.py", "repo_or_config"),
        (
            REPO_ROOT / "scripts" / "methods" / "daaam" / "run_pipeline_patched.py",
            "DAAAM adapter run_pipeline wrapper",
            "repo_or_config",
        ),
        (
            args.daaam_root / "scripts" / "postprocess_scene_graph.py",
            "DAAAM postprocess_scene_graph.py",
            "repo_or_config",
        ),
        (args.class_names, "shared OV class list", "shared_module_artifact"),
    ):
        if path is not None and not Path(path).exists():
            missing.append({"label": label, "path": str(path), "kind": kind})
    if require_native_run:
        required_paths = [
            (args.config, "DAAAM pipeline config", "repo_or_config"),
            (args.hydra_config_path, "Hydra config", "repo_or_config"),
        ]
        if args.daaam_segmenter == "native_fastsam_trt":
            required_paths.extend(
                [
                    (
                        _resolve_native_fastsam_model(args),
                        "shared FastSAM TensorRT engine/checkpoint",
                        "shared_module_artifact",
                    ),
                    (
                        _resolve_daaam_config_path(args.daaam_root, args.native_fastsam_config_path),
                        "DAAAM FastSAM model config",
                        "repo_or_config",
                    ),
                ]
            )
        else:
            required_paths.extend(
                [
                    (args.sam_checkpoint, "shared SAM checkpoint", "shared_module_artifact"),
                    (args.sam_model_config_path, "DAAAM SAM model config", "repo_or_config"),
                ]
            )
        for path, label, kind in required_paths:
            if path is None or not Path(path).exists():
                missing.append({"label": label, "path": str(path), "kind": kind})
        if args.with_reid:
            reid_path = _resolve_reid_weights(args)
            if reid_path is None or not reid_path.exists():
                missing.append(
                    {
                        "label": "DAAAM BotSort ReID weights",
                        "path": str(reid_path) if reid_path is not None else "(none provided)",
                        "kind": "shared_module_artifact",
                    }
                )
    if missing:
        raise FileNotFoundError(_format_missing_path_error(missing))
    if not args.skip_dependency_preflight:
        _preflight_python_deps(args, require_native_run=require_native_run)
    if require_native_run and not args.skip_cuda_preflight:
        _preflight_cuda(args)


def _preflight_python_deps(args: argparse.Namespace, *, require_native_run: bool) -> None:
    imports = ["spark_dsg", "daaam", "open_clip", "sentence_transformers"]
    if require_native_run:
        # The native build also drives the DAM grounding worker (object
        # description), the BotSort tracker, and the assignment worker.
        # Importing these modules surfaces heavy DAM/gradio/langchain/cvxpy
        # dependency issues during preflight instead of crashing several frames
        # into a long run.
        imports.extend(
            [
                "torch",
                "ultralytics",
                "segment_anything",
                "boxmot",
                "cvxpy",
                "daaam.grounding.workers.dam_grounding",
            ]
        )
    code = f"""
import importlib
import json
import sys
from pathlib import Path
sys.path.insert(0, {str(args.daaam_root / "src")!r})
imports = {imports!r}
status = {{}}
for name in imports:
    try:
        module = importlib.import_module(name)
        status[name] = {{"ok": True, "file": getattr(module, "__file__", None)}}
    except Exception as exc:
        status[name] = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}
print(json.dumps(status, indent=2))
if any(not item["ok"] for item in status.values()):
    raise SystemExit(2)
"""
    result = subprocess.run(
        [str(args.daaam_python), "-c", code],
        cwd=args.daaam_root,
        env=_daaam_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    grounding_hint = ""
    if "dam_grounding" in result.stdout and "fastapi" in result.stdout:
        grounding_hint = (
            "\nKnown blocker: the DAM grounding worker "
            "(daaam.grounding.workers.dam_grounding) imports gradio/fastapi and "
            "loads the multi-GB nvidia/DAM-3B model. If your --daaam-python env "
            "lacks fastapi or cannot fetch DAM-3B, the native object-description "
            "stage cannot run, so no DSG (and therefore no Track 1/2 memory) is "
            "produced. Install the grounding deps and cache DAM-3B in the DAAAM "
            "env, or package an existing native --native-output-dir instead."
        )
    raise RuntimeError(
        "DAAAM dependency preflight failed.\n"
        f"{DAAAM_ENV_HINT}\n"
        f"{DAAAM_ARTIFACT_HINT}\n"
        "Use --daaam-python pointing to a DAAAM conda/runtime env with the "
        "failed imports installed. For package-from-output, spark_dsg and "
        "daaam are still required to parse the DSG."
        f"{grounding_hint}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


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
x = torch.randn(1, 3, 32, 32, device=device)
conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).to(device)
with torch.no_grad():
    y = conv(x)
torch.cuda.synchronize()
print(f"cuda_cudnn_smoke=ok shape={tuple(y.shape)}")
"""
    result = subprocess.run(
        [str(args.daaam_python), "-c", code],
        cwd=args.daaam_root,
        env=_daaam_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    raise RuntimeError(
        "CUDA/cuDNN preflight failed before launching DAAAM. Run on a GPU node "
        "with a healthy NVIDIA driver, set --cuda-visible-devices, or pass "
        "--skip-cuda-preflight only if you intentionally want DAAAM to fail "
        f"inside its own runtime.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _run_daaam_postprocess(args: argparse.Namespace, native_output_dir: Path) -> None:
    if not (native_output_dir / "dsg.json").exists():
        warning = f"skipping DAAAM postprocess: no dsg.json in {native_output_dir}"
        print(warning, file=sys.stderr)
        _append_adapter_warning(native_output_dir, warning)
        return
    if not (native_output_dir / "corrections.yaml").exists():
        warning = f"skipping DAAAM postprocess: no corrections.yaml in {native_output_dir}"
        print(warning, file=sys.stderr)
        _append_adapter_warning(native_output_dir, warning)
        return
    command = [
        str(args.daaam_python),
        str(args.daaam_root / "scripts" / "postprocess_scene_graph.py"),
        "--data-dir",
        str(native_output_dir),
        "--sentence-model-name",
        str(args.sentence_embedding_model),
    ]
    print("postprocessing DAAAM DSG:")
    print(" ".join(command))
    result = subprocess.run(
        command,
        cwd=args.daaam_root,
        env=_daaam_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        return
    combined = result.stdout + "\n" + result.stderr
    if "KeyError: 'features'" in combined or "no DAAAM object embeddings found" in combined:
        warning = (
            "DAAAM postprocess failed because the DSG does not contain semantic "
            "feature metadata yet. Continuing with the raw DSG; Track 1 can still "
            "export objects, while Track 2 fixed API may be invalid."
        )
        print(f"warning: {warning}", file=sys.stderr)
        _append_adapter_warning(native_output_dir, warning)
        return
    raise RuntimeError(
        "DAAAM postprocess_scene_graph.py failed.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _collect_daaam_native_outputs(
    args: argparse.Namespace,
    native_output_dir: Path,
    *,
    run_started_at: float,
) -> None:
    """Normalize DAAAM's actual output layout into the adapter native dir.

    DAAAM's runner can write Hydra DSG artifacts under ``hydra_output/`` and
    DAAAM correction artifacts under a timestamped ``out_*`` directory. Keep the
    original tree, but copy the files needed by postprocess/package to the
    native root.
    """
    native_output_dir.mkdir(parents=True, exist_ok=True)
    hydra_roots = [
        native_output_dir / "hydra_output",
        args.daaam_root / "output" / "hydra_output",
    ]
    for hydra_root in hydra_roots:
        for relative in (
            Path("backend/dsg.json"),
            Path("backend/dsg_with_mesh.json"),
            Path("frontend/dsg.json"),
            Path("frontend/dsg_with_mesh.json"),
            Path("backend/mesh.ply"),
        ):
            source = hydra_root / relative
            if source.exists() and _is_from_current_run(source, run_started_at):
                target = native_output_dir / "hydra_output" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)

    dsg_candidates = [
        native_output_dir / "hydra_output" / "backend" / "dsg.json",
        args.daaam_root / "output" / "hydra_output" / "backend" / "dsg.json",
    ]
    for source in dsg_candidates:
        if source.exists() and _is_from_current_run(source, run_started_at):
            target = native_output_dir / "dsg.json"
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            break

    timestamped_outputs = sorted(
        list(native_output_dir.glob("out_*")) + list((args.daaam_root / "output").glob("out_*")),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for run_dir in timestamped_outputs:
        if not run_dir.is_dir() or not _is_from_current_run(run_dir, run_started_at):
            continue
        archive_target = native_output_dir / "daaam_output" / run_dir.name
        if run_dir.resolve() != archive_target.resolve() and not archive_target.exists():
            shutil.copytree(run_dir, archive_target)
        for name in (
            "dsg.json",
            "adapter_corrected_dsg_status.json",
            "corrections.yaml",
            "object_positions.json",
            "correction_stats.json",
            "background_objects.yaml",
            "keyframe_annotations.yaml",
            "clip_features.pkl",
            "pipeline_config.yaml",
            "performance_statistics.csv",
        ):
            source = run_dir / name
            target = native_output_dir / name
            if source.exists() and (name == "dsg.json" or not target.exists()):
                shutil.copy2(source, target)
        if (native_output_dir / "corrections.yaml").exists():
            break


def _is_from_current_run(path: Path, run_started_at: float) -> bool:
    # Allow a small clock/logging slack for files created at process startup.
    return path.stat().st_mtime >= run_started_at - 5.0


def _extract_daaam_memory(args: argparse.Namespace, *, dsg_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = output_path.with_suffix(".input.json")
    payload = {
        "daaam_root": str(args.daaam_root),
        "dsg_path": str(dsg_path),
        "native_dir": str(dsg_path.parent),
        "class_names": str(args.class_names),
        "build_track2_index": not args.skip_track2_index,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "sentence_embedding_model": str(args.sentence_embedding_model),
    }
    _write_json(input_path, payload)
    result = subprocess.run(
        [str(args.daaam_python), "-c", _DAAAM_EXTRACTION_CODE, str(input_path), str(output_path)],
        cwd=args.daaam_root,
        env=_daaam_subprocess_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "DAAAM DSG extraction failed. The --daaam-python env must import "
            "daaam and spark_dsg and be able to load the DSG. This is an "
            "environment issue, not a model-checkpoint issue.\n"
            f"{DAAAM_ENV_HINT}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    with output_path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected extraction JSON object: {output_path}")
    return value


def _canonicalize_objects(
    extraction: dict[str, Any],
    class_labels: list[str],
    *,
    source_key: str,
    source_artifact: str,
    evidence_note: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    import numpy as np

    warnings: list[str] = []
    raw_objects = extraction.get(source_key) or []
    track2 = extraction.get("track2_index") if isinstance(extraction.get("track2_index"), dict) else {}
    object_embeddings = {
        str(item.get("object_id")): np.asarray(item.get("embedding"), dtype=np.float32)
        for item in track2.get("objects", []) or []
        if item.get("object_id") and item.get("embedding")
    }
    label_embeddings = np.asarray(track2.get("label_embeddings") or [], dtype=np.float32)
    labels_from_index = list(track2.get("labels") or [])
    label_lookup_ready = (
        len(class_labels) == len(labels_from_index)
        and label_embeddings.ndim == 2
        and len(label_embeddings) == len(class_labels)
    )
    objects: list[dict[str, Any]] = []
    for raw in raw_objects:
        object_id = str(raw["object_id"])
        raw_label = str(raw.get("raw_label") or raw.get("label") or "object")
        canonical, source, score = _canonical_label_for_object(
            object_id=object_id,
            raw_label=raw_label,
            class_labels=class_labels,
            object_embeddings=object_embeddings,
            label_embeddings=label_embeddings,
            label_lookup_ready=label_lookup_ready,
        )
        if source == "raw_unmapped":
            warnings.append(f"unmapped DAAAM label for {object_id}: {raw_label!r}")
        objects.append(
            {
                "object_id": object_id,
                "label": canonical,
                "raw_label": raw_label,
                "label_source": source,
                "label_score": score,
                "aliases": [],
                "position_3d": raw.get("position_3d"),
                "bbox_3d": raw.get("bbox_3d"),
                "dimensions_3d": raw.get("dimensions_3d"),
                "confidence": None,
                "semantic_label": raw.get("semantic_label"),
                "is_background": bool(raw.get("is_background")),
                "first_observed": raw.get("first_observed"),
                "last_observed": raw.get("last_observed"),
                "observation_timestamps": raw.get("observation_timestamps") or [],
                "source_artifacts": [source_artifact, "memory/native/" + str(raw.get("dsg_name") or "dsg.json")],
                "provenance": raw.get("provenance"),
                "evidence": [
                    {
                        "source_type": "daaam_dsg_node",
                        "source_path": "evidence/object_sources.jsonl",
                        "notes": evidence_note,
                    }
                ],
            }
        )
    return objects, warnings


def _canonical_label_for_object(
    *,
    object_id: str,
    raw_label: str,
    class_labels: list[str],
    object_embeddings: dict[str, np.ndarray],
    label_embeddings: np.ndarray,
    label_lookup_ready: bool,
) -> tuple[str, str, float | None]:
    import numpy as np

    normalized = normalize_label(raw_label, DEFAULT_LABEL_ALIASES)
    if normalized in class_labels:
        return normalized, "raw_exact_or_alias", None
    keyword = _keyword_label_for_raw_label(normalized, class_labels)
    if keyword is not None:
        return keyword, "raw_keyword_or_alias", None
    if label_lookup_ready and object_id in object_embeddings:
        embedding = object_embeddings[object_id]
        if embedding.ndim == 1 and embedding.shape[0] == label_embeddings.shape[1]:
            scores = label_embeddings @ embedding
            index = int(np.argmax(scores))
            return class_labels[index], "daaam_semantic_projection", float(scores[index])
    return normalized or "object", "raw_unmapped", None


def _keyword_label_for_raw_label(normalized_raw_label: str, class_labels: list[str]) -> str | None:
    class_set = set(class_labels)
    phrase_aliases = dict(DEFAULT_LABEL_ALIASES)
    phrase_aliases.update(
        {
            "desk": "table",
            "office chair": "chair",
            "computer monitor": "monitor",
            "flat screen monitor": "monitor",
            "flat screen computer monitor": "monitor",
            "light fixture": "lamp",
            "fluorescent light fixture": "lamp",
            "lampshade": "lamp",
        }
    )
    for phrase, label in sorted(phrase_aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if label in class_set and _contains_label_phrase(normalized_raw_label, phrase):
            return label
    for label in sorted(class_labels, key=len, reverse=True):
        if _contains_label_phrase(normalized_raw_label, label):
            return label
    return None


def _contains_label_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalize_label(phrase, {})
    if not normalized_phrase:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(normalized_phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _write_track2_index(
    *,
    package_dir: Path,
    extraction: dict[str, Any],
    objects: list[dict[str, Any]],
    class_labels: list[str],
) -> tuple[str, str]:
    track2 = extraction.get("track2_index") if isinstance(extraction.get("track2_index"), dict) else {}
    if track2.get("status") != "ok":
        reason = str(track2.get("reason") or "DAAAM semantic index was not built.")
        _write_track2_label_index(package_dir, objects, source_reason=reason)
        return "supported", f"Using deterministic object_table label query; native DAAAM semantic index unavailable: {reason}"
    object_embeddings = track2.get("objects") or []
    label_embeddings = track2.get("label_embeddings") or []
    labels = track2.get("labels") or []
    if not object_embeddings or not label_embeddings or labels != class_labels:
        reason = "DAAAM semantic index is incomplete or not aligned with the shared class list."
        _write_track2_label_index(package_dir, objects, source_reason=reason)
        return "supported", f"Using deterministic object_table label query; {reason}"
    object_by_id = {obj["object_id"]: obj for obj in objects}
    rows = []
    for item in object_embeddings:
        object_id = str(item.get("object_id"))
        if object_id not in object_by_id:
            continue
        rows.append(
            {
                "object_id": object_id,
                "embedding": item["embedding"],
                "label": object_by_id[object_id]["label"],
                "raw_label": object_by_id[object_id]["raw_label"],
            }
        )
    if not rows:
        reason = "DAAAM semantic index contains no package object ids."
        _write_track2_label_index(package_dir, objects, source_reason=reason)
        return "supported", f"Using deterministic object_table label query; {reason}"
    _write_json(
        package_dir / "memory" / "track2_semantic_index.json",
        {
            "status": "ok",
            "source": "DAAAM GetMatchingSubjects/precompute_unified_embeddings deterministic semantic index",
            "labels": labels,
            "label_embeddings": label_embeddings,
            "objects": rows,
        },
    )
    return "supported", ""


def _write_track2_label_index(package_dir: Path, objects: list[dict[str, Any]], *, source_reason: str) -> None:
    _write_json(
        package_dir / "memory" / "track2_label_index.json",
        {
            "status": "ok",
            "source": "deterministic canonical-label query over memory/object_table.jsonl",
            "source_reason": source_reason,
            "objects": [
                {
                    "object_id": obj.get("object_id"),
                    "label": obj.get("label"),
                    "raw_label": obj.get("raw_label"),
                    "label_source": obj.get("label_source"),
                    "position_3d": obj.get("position_3d"),
                }
                for obj in objects
            ],
        },
    )


def _write_tool_files(package_dir: Path, *, track2_supported: bool) -> None:
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
    if not track2_supported:
        return
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
    predictions = _rank_by_label(objects, target_label, query_text)[:top_k]
    return {"status": "ok", "predictions": predictions}


def _rank_by_label(
    objects: list[dict[str, Any]],
    target_label: str,
    query_text: str,
) -> list[dict[str, Any]]:
    ranked = []
    target_tokens = set(target_label.split())
    query_tokens = set(query_text.split())
    for obj in objects:
        label = _normalize_label(obj.get("label") or "object")
        raw_label = _normalize_label(obj.get("raw_label") or "")
        label_tokens = set(label.split())
        raw_tokens = set(raw_label.split())
        if target_label and target_label == label:
            score = 1.0
        elif target_label and (target_label in label or label in target_label):
            score = 0.9
        elif target_label and target_label in raw_label:
            score = 0.8
        elif target_tokens and target_tokens & label_tokens:
            score = 0.75
        elif target_tokens and target_tokens & raw_tokens:
            score = 0.6
        elif query_text and query_text == label:
            score = 0.55
        elif query_text and (query_text in label or label in query_text or query_text in raw_label):
            score = 0.45
        elif query_tokens and (query_tokens & label_tokens or query_tokens & raw_tokens):
            score = 0.35
        else:
            score = 0.05
        provenance = obj.get("provenance") if isinstance(obj.get("provenance"), dict) else {}
        score += min(float(provenance.get("position_observation_count") or 0) / 1000.0, 0.1)
        ranked.append(
            {
                "object_id": obj.get("object_id"),
                "label": obj.get("label"),
                "raw_label": obj.get("raw_label"),
                "position_3d": obj.get("position_3d"),
                "bbox_3d": obj.get("bbox_3d"),
                "score": score,
                "evidence": obj.get("evidence", []),
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), str(item.get("object_id"))))
    return ranked


def _load_objects(package_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (package_dir / "memory" / "object_table.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize_label(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\\s+", " ", text).strip()
""",
        encoding="utf-8",
    )


def _write_package_schemas(package_dir: Path, *, track2_supported: bool) -> None:
    _write_json(package_dir / "schemas" / "track1_input.schema.json", {"type": "object", "additionalProperties": True})
    _write_json(
        package_dir / "schemas" / "object_table.schema.json",
        {
            "type": "object",
            "required": ["status", "objects"],
            "properties": {"status": {"const": "ok"}, "objects": {"type": "array"}},
        },
    )
    if track2_supported:
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


def _write_schema_md(package_dir: Path, *, dsg_path: Path, track2_status: str, track2_reason: str) -> None:
    package_dir.joinpath("schema.md").write_text(
        f"""# DAAAM Minimal Memory Schema

Coordinate frame and units: object positions come from DAAAM/Hydra Dynamic
Scene Graph object nodes. Units are meters in the DAAAM/Hydra world frame.

Object id format: merged DSG objects use `daaam_object_<id>` and are written to
`memory/object_table.jsonl`. DAAAM BACKGROUND_OBJECTS use
`daaam_background_<id>` and are written separately to
`memory/background_object_table.jsonl`.

Track 1/2 fixed API reads only `memory/object_table.jsonl`, which represents
the native merged DAAAM/Hydra OBJECTS layer. BACKGROUND_OBJECTS are retained for
debugging, provenance, and agentic full-access evaluation, but they are not part
of the default fixed-API object universe.

Timestamp fields: rows may include `first_observed`, `last_observed`, and
`observation_timestamps` copied from DAAAM/Hydra temporal metadata. These times
are seconds in the prepared RGB-D sequence timeline.

Label meaning: `raw_label` preserves the DAAAM DAM/VLM free-text object
description. `label` is the canonical evaluator label: exact/alias mapping
first, then DAAAM semantic embedding projection to the shared OV class list
when a semantic index is available. `label_source` records which route was used.

Relations: DAAAM/Hydra DSG relation edges are kept in `memory/native/dsg.json`
for provenance, but the Track 1/2 fixed API object inventory does not expose
relations yet.

Confidence or score: DAAAM DAM corrections are free-text descriptions and do
not always expose a calibrated detector confidence. Exported rows keep native
score/confidence fields when present; otherwise confidence is null/omitted and
provenance plus observation counts should be used for debugging only.

Native artifact formats: native DSG artifacts are copied under
`memory/native/`. The source DSG was `{dsg_path}`. `memory/object_table.jsonl`
is the canonical Track 1/2 object inventory. `memory/background_object_table.jsonl`
contains DAAAM filtered/background objects for debug/evidence. `evidence/object_sources.jsonl`
maps both tables back to DAAAM node provenance.

Track 2 fixed API: {track2_status}. {track2_reason or "The package includes a deterministic DAAAM semantic index exported from native scene-understanding embeddings."}

Known limitations: the package does not use DAAAM's LLM `SceneUnderstandingAgent`
for Track 2 fixed API. Track 3 ScanRefer and Track 4 OpenEQA are invalid in this
first DAAAM Track 1/2 adapter.
""",
        encoding="utf-8",
    )


def _write_manifest(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    native_output_dir: Path,
    dsg_path: Path,
    native_dsg_path: Path,
    object_count: int,
    background_object_count: int,
    layout_dir: Path | None,
    layout_summary: dict[str, Any],
    track2_status: str,
) -> None:
    _write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.2",
            "package_id": f"daaam/{getattr(args, 'dataset_tag', None) or 'scannetpp'}/{args.scene_id}/{run_id}",
            "method": {
                "name": "daaam",
                "display_name": "DAAAM",
                "family": "scene_graph",
                "repo_path": str(args.daaam_root),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": getattr(args, "dataset_tag", None) or "scannetpp",
                "split": "smoke",
                "scene_id": args.scene_id,
                "episode_id": None,
            },
            "input": {
                "modality": ["rgb", "depth", "pose", "intrinsics", "timestamp"],
                "frame_count": int(layout_summary.get("frame_count") or 0),
                "rgbd_root": str(layout_dir) if layout_dir is not None else None,
                "poses_path": str(layout_dir / "pose") if layout_dir is not None else None,
                "intrinsics_path": str(layout_dir / "camera_info.json") if layout_dir is not None else None,
                "timestamp_path": None,
                "coordinate_frame": "DAAAM/Hydra world frame from aligned ScanNet++ poses; meters",
            },
            "vocabulary": {
                "vocabulary_mode": "open_vocabulary",
                "class_list_path": str(args.class_names),
                "source": "shared_modules + DAAAM raw DAM/VLM labels",
                "profile": args.shared_module_profile,
            },
            "modules": {
                **shared_modules_metadata(args),
                "daaam_native": {
                    "segmenter_mode": args.daaam_segmenter,
                    "native_fastsam_model": str(_resolve_native_fastsam_model(args))
                    if args.daaam_segmenter == "native_fastsam_trt"
                    else None,
                    "native_fastsam_config_path": str(
                        _resolve_daaam_config_path(args.daaam_root, args.native_fastsam_config_path)
                    )
                    if args.daaam_segmenter == "native_fastsam_trt"
                    else None,
                    "dam_model": str(args.dam_model_path),
                    "sentence_embedding_model": str(args.sentence_embedding_model),
                    "hydra_config_path": str(args.hydra_config_path) if args.hydra_config_path else None,
                    "sam_model_config_path": str(args.sam_model_config_path),
                    "sam_checkpoint": str(args.sam_checkpoint)
                    if args.daaam_segmenter == "shared_sam"
                    else None,
                    "daaam_python": str(args.daaam_python),
                },
            },
            "explicit_memory": True,
            "memory_artifacts": [
                {
                    "name": "object_table",
                    "type": "jsonl",
                    "path": "memory/object_table.jsonl",
                    "description": (
                        f"Canonical DAAAM/Hydra merged OBJECTS inventory with {object_count} objects. "
                        "This is the Track 1/2 fixed-API object universe."
                    ),
                    "required_for": ["track1_object_location"],
                },
                {
                    "name": "background_object_table",
                    "type": "jsonl",
                    "path": "memory/background_object_table.jsonl",
                    "description": (
                        f"DAAAM BACKGROUND_OBJECTS/debug inventory with {background_object_count} objects. "
                        "Not used by Track 1/2 fixed API."
                    ),
                    "required_for": [],
                },
                {
                    "name": "native_dsg",
                    "type": "json",
                    "path": f"memory/native/{native_dsg_path.name}",
                    "description": "DAAAM/Hydra Dynamic Scene Graph JSON.",
                    "required_for": [],
                },
            ]
            + (
                [
                    {
                        "name": "track2_semantic_index",
                        "type": "json",
                        "path": "memory/track2_semantic_index.json",
                        "description": "Deterministic DAAAM semantic query index for Track 2.",
                        "required_for": ["track1_object_location"],
                    }
                ]
                if (package_dir / "memory" / "track2_semantic_index.json").exists()
                else []
            )
            + (
                [
                    {
                        "name": "track2_label_index",
                        "type": "json",
                        "path": "memory/track2_label_index.json",
                        "description": "Deterministic canonical-label object index for Track 2.",
                        "required_for": ["track1_object_location"],
                    }
                ]
                if (package_dir / "memory" / "track2_label_index.json").exists()
                else []
            ),
            "evidence_artifacts": [
                {
                    "name": "object_sources",
                    "type": "jsonl",
                    "path": "evidence/object_sources.jsonl",
                    "description": "Mapping from package objects to DAAAM DSG provenance.",
                    "required_for": [],
                }
            ],
            "raw_links": [
                {
                    "name": "native_daaam_output",
                    "type": "directory",
                    "path": str(native_output_dir),
                    "description": "Native DAAAM output directory.",
                    "required_for": [],
                },
                {
                    "name": "native_dsg_source",
                    "type": "file",
                    "path": str(dsg_path),
                    "description": "Native DAAAM DSG source file.",
                    "required_for": [],
                },
            ],
            "tools": [
                {
                    "name": "list_objects",
                    "type": "python",
                    "path": "tools/list_objects.py",
                    "description": "Return the exported DAAAM object table.",
                    "required_for": ["track1_object_location"],
                }
            ]
            + (
                [
                    {
                        "name": "query_object",
                        "type": "python",
                        "path": "tools/query_object.py",
                        "description": "Query exported DAAAM object memory with a target label.",
                        "required_for": ["track1_object_location"],
                    }
                ]
                if track2_status == "supported"
                else []
            ),
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [str(args.config)],
                "environment": str(args.daaam_python),
                "started_at": None,
                "finished_at": None,
                "build_runtime_seconds": None,
                "runtime_seconds": None,
                "frame_count": int(layout_summary.get("frame_count") or 0),
                "time_per_frame_seconds": None,
                "native_memory_size_bytes": None,
                "native_memory_artifacts": [],
                "memory_artifact_size_bytes": None,
                "package_size_bytes": None,
                "peak_ram_bytes": None,
                "peak_ram_unavailable_reason": None,
                "peak_vram_bytes": None,
                "peak_vram_unavailable_reason": None,
            },
            "allowed_access": {
                "contains_gt_annotations": False,
                "contains_benchmark_answers": False,
                "contains_test_labels": False,
                "contains_question_specific_rules": False,
            },
            "notes": "DAAAM package built from native Dynamic Scene Graph output; external DAAAM repo was not modified.",
        },
    )


def _write_capabilities(package_dir: Path, *, track2_status: str, track2_reason: str) -> None:
    track2_supported = track2_status == "supported"
    _write_json(
        package_dir / "capabilities.json",
        {
            "schema_version": "0.2",
            "fixed_api": {
                "track1_object_location": {
                    "status": "supported" if track2_supported else "invalid",
                    "entrypoint": "tools/query_object.py:query_object" if track2_supported else None,
                    "reason": track2_reason if track2_reason else "",
                    "input_schema": "schemas/track2_input.schema.json" if track2_supported else None,
                    "output_schema": "schemas/object_query_result.schema.json" if track2_supported else None,
                },
                "track2_scanrefer": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": "No ScanRefer referring-expression resolver is exported.",
                },
                "track3_openeqa": {
                    "status": "invalid",
                    "entrypoint": None,
                    "reason": "DAAAM's LLM SceneUnderstandingAgent is not used as a fixed API in this Track 1/2 package.",
                },
            },
            "agent_access": {
                "mode": "tool_llm",
                "read_manifest": True,
                "read_schema": True,
                "read_native_memory": True,
                "read_fixed_api_views": False,
                "read_evidence": True,
                "read_adapter_code": False,
                "read_shared_module_code": False,
                "read_method_root_source_code": True,
                "read_build_code": False,
                "read_raw_links": False,
                "read_raw_frames": False,
                "read_source_keyframes_or_crops": False,
                "run_method_native_tools": True,
                "write_package": False,
            },
        },
    )


def _write_build_log(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    started_at: float,
    native_output_dir: Path,
    native_artifacts: list[Path],
    object_count: int,
    background_object_count: int,
    layout_summary: dict[str, Any],
    track2_status: str,
    warnings: list[str],
) -> None:
    finished_at = time.time()
    runtime_seconds = finished_at - started_at
    write_build_log_with_accounting(
        package_dir=package_dir,
        native_memory_artifact_paths=[native_output_dir, *native_artifacts],
        frame_count=int(layout_summary.get("frame_count") or 0),
        build_log={
            "status": "ok",
            "started_at": _iso_time(started_at),
            "finished_at": _iso_time(finished_at),
            "build_runtime_seconds": runtime_seconds,
            "runtime_seconds": runtime_seconds,
            "command": " ".join(sys.argv),
            "config_paths": [str(args.config)],
            "environment": str(args.daaam_python),
            "source_outputs": [str(native_output_dir)],
            "object_count": object_count,
            "background_object_count": background_object_count,
            "track2_fixed_api": track2_status,
            "shared_modules": shared_modules_metadata(args),
            "daaam_runtime": {
                "daaam_root": str(args.daaam_root),
                "segmenter_mode": args.daaam_segmenter,
                "hydra_config_path": str(args.hydra_config_path) if args.hydra_config_path else None,
                "sam_checkpoint": str(args.sam_checkpoint) if args.daaam_segmenter == "shared_sam" else None,
                "sam_model_config_path": str(args.sam_model_config_path)
                if args.daaam_segmenter == "shared_sam"
                else None,
                "native_fastsam_model": str(_resolve_native_fastsam_model(args))
                if args.daaam_segmenter == "native_fastsam_trt"
                else None,
                "native_fastsam_config_path": str(
                    _resolve_daaam_config_path(args.daaam_root, args.native_fastsam_config_path)
                )
                if args.daaam_segmenter == "native_fastsam_trt"
                else None,
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "dam_model_path": str(args.dam_model_path),
                "sentence_embedding_model": str(args.sentence_embedding_model),
                "frame_stride": args.frame_stride,
                "max_frames": args.max_frames,
                "target_fps": args.target_fps,
                "with_reid": bool(args.with_reid),
                "reid_weights": args.reid_weights,
                "applied_native_overrides": _segmenter_config_overrides(args)
                + _tracking_config_overrides(args),
            },
            "warnings": warnings,
        },
    )


def _write_raw_links(
    package_dir: Path,
    *,
    args: argparse.Namespace,
    native_output_dir: Path,
    layout_dir: Path | None,
    dsg_path: Path,
) -> None:
    _write_json(
        package_dir / "raw_links" / "native_sources.json",
        {
            "daaam_root": str(args.daaam_root),
            "native_output_dir": str(native_output_dir),
            "native_dsg_path": str(dsg_path),
            "layout_dir": str(layout_dir) if layout_dir is not None else None,
            "scannetpp_root": str(args.scannetpp_root),
            "source_scene": args.scene_id,
        },
    )


def _find_dsg_path(native_output_dir: Path) -> Path:
    corrected_candidates = sorted(
        [
            status_path.parent / "dsg.json"
            for status_path in native_output_dir.rglob("adapter_corrected_dsg_status.json")
        ],
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for path in corrected_candidates:
        if path.exists():
            return path

    candidates = [
        native_output_dir / "dsg_updated.json",
        native_output_dir / "clustered_dsg.json",
        native_output_dir / "dsg.json",
        native_output_dir / "backend" / "dsg_updated.json",
        native_output_dir / "backend" / "dsg.json",
        native_output_dir / "hydra_output" / "backend" / "dsg.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    recursive = sorted(
        list(native_output_dir.rglob("dsg_updated.json"))
        + list(native_output_dir.rglob("clustered_dsg.json"))
        + list(native_output_dir.rglob("dsg.json")),
        key=lambda path: (len(path.parts), str(path)),
    )
    if recursive:
        return recursive[0]
    exists = native_output_dir.exists()
    listing = (
        sorted(p.name for p in native_output_dir.iterdir())[:20]
        if exists
        else []
    )
    raise FileNotFoundError(
        "no DAAAM Dynamic Scene Graph JSON found for package-from-output.\n"
        f"Looked under: {native_output_dir} "
        f"({'directory exists' if exists else 'directory does not exist'}).\n"
        f"Searched names: dsg_updated.json, clustered_dsg.json, dsg.json "
        "(also under backend/ and hydra_output/backend/).\n"
        + (f"Found instead: {listing}\n" if exists else "")
        + "Next actions:\n"
        "- If you meant to package an existing native run, pass "
        "--native-output-dir pointing at the DAAAM output directory that "
        "contains a DSG JSON.\n"
        "- To produce a DSG, run the raw build route (drop --skip-daaam-run) "
        "with a DAAAM runtime that can complete segmentation + DAM grounding; "
        "the DSG is written on pipeline shutdown.\n"
        "- A DSG is required: Track 1 object inventory and the Track 2 "
        "semantic index are both exported from it."
    )


def _find_native_artifact(native_output_dir: Path, name: str) -> Path:
    direct = native_output_dir / name
    if direct.exists():
        return direct
    candidates = sorted(
        list(native_output_dir.glob(f"out_*/{name}"))
        + list(native_output_dir.glob(f"daaam_output/out_*/{name}")),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return direct


def _format_missing_path_error(missing: list[dict[str, str]]) -> str:
    repo_lines = [
        f"- {item['label']}: {item['path']}"
        for item in missing
        if item.get("kind") == "repo_or_config"
    ]
    artifact_lines = [
        f"- {item['label']}: {item['path']}"
        for item in missing
        if item.get("kind") == "shared_module_artifact"
    ]
    sections = ["DAAAM preflight missing required paths:"]
    if repo_lines:
        sections.append("Repo/config paths to fix in the DAAAM or eval setup:\n" + "\n".join(repo_lines))
    if artifact_lines:
        sections.append(
            "Shared module artifacts/checkpoints to put on NAS/shared modules:\n"
            + "\n".join(artifact_lines)
        )
    sections.append("Policy:\n- " + DAAAM_ENV_HINT + "\n- " + DAAAM_ARTIFACT_HINT)
    return "\n\n".join(sections)


def _materialize_daaam_sam_checkpoint(args: argparse.Namespace, native_output_dir: Path) -> Path:
    checkpoint = Path(args.sam_checkpoint)
    if "sam_vit" in checkpoint.name.lower():
        return checkpoint
    link_dir = native_output_dir / "shared_module_links"
    link_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"sam_{args.sam_type}.pt" if str(args.sam_type).startswith("vit_") else checkpoint.name
    if "sam_vit" not in target_name:
        target_name = f"sam_{str(args.sam_type or 'vit_b')}.pt"
    target = link_dir / target_name
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, target)
    return target


def _read_prepared_layout_summary(layout_dir: Path) -> dict[str, Any]:
    missing = [item for item in (layout_dir / "rgb", layout_dir / "depth", layout_dir / "pose", layout_dir / "camera_info.json") if not item.exists()]
    if missing:
        raise FileNotFoundError("prepared DAAAM layout is incomplete:\n" + "\n".join(str(item) for item in missing))
    summary_path = layout_dir / "layout_summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    frame_count = len(list((layout_dir / "rgb").glob("*")))
    return {"layout_dir": str(layout_dir), "frame_count": frame_count, "depth_scale": 1000.0}


def _write_camera_info(path: Path, *, intrinsic_path: Path, rgb_dir: Path) -> None:
    matrix = _read_matrix(intrinsic_path)
    first_image = sorted(rgb_dir.glob("*"))[0]
    import cv2

    image = cv2.imread(str(first_image))
    if image is None:
        raise RuntimeError(f"failed to read image for camera_info: {first_image}")
    height, width = image.shape[:2]
    _write_json(
        path,
        {
            "width": width,
            "height": height,
            "intrinsics": matrix.tolist(),
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )


def _read_matrix(path: Path) -> np.ndarray:
    import numpy as np

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append([float(value) for value in text.split()])
    return np.asarray(rows, dtype=np.float64)


def _daaam_subprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(args.daaam_root / "src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    openclip_cache = _openclip_hf_cache_dir(args)
    if openclip_cache is not None:
        env.setdefault("HF_HUB_CACHE", str(openclip_cache))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(openclip_cache))
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _openclip_hf_cache_dir(args: argparse.Namespace) -> Path | None:
    model = getattr(args, "clip_model", None)
    pretrained = getattr(args, "clip_pretrained", None)
    if not model or not pretrained:
        return None
    candidate = DEFAULT_SHARED_MODULES_ROOT / "openclip" / str(model) / str(pretrained) / "hf_cache"
    if candidate.exists():
        return candidate
    return None


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _append_adapter_warning(native_output_dir: Path, warning: str) -> None:
    path = native_output_dir / ADAPTER_WARNINGS_NAME
    warnings = _read_adapter_warnings(native_output_dir)
    if warning not in warnings:
        warnings.append(warning)
    _write_json(path, warnings)


def _read_adapter_warnings(native_output_dir: Path) -> list[str]:
    path = native_output_dir / ADAPTER_WARNINGS_NAME
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat()


_DAAAM_EXTRACTION_CODE = r'''
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output_path = Path(sys.argv[2])
daaam_root = Path(payload["daaam_root"])
native_dir = Path(payload["native_dir"])
sys.path.insert(0, str(daaam_root / "src"))

warnings = []

import spark_dsg as sdsg
from spark_dsg import DsgLayers, NodeSymbol
from daaam.scene_understanding.utils import retrieve_objects_from_scene_graph
from daaam.scene_understanding.tools.utils import (
    extract_background_objects,
    get_unified_objects_list,
    precompute_unified_embeddings,
)
from daaam.scene_understanding.config import ToolConfig


def as_list(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.size == 0:
        return None
    return array.reshape(-1).astype(float).tolist()


def bbox_from_center_dims(center, dims):
    if center is None or dims is None:
        return None
    center_arr = np.asarray(center, dtype=float).reshape(-1)
    dims_arr = np.asarray(dims, dtype=float).reshape(-1)
    if center_arr.size != 3 or dims_arr.size != 3 or not np.all(dims_arr > 0):
        return None
    half = dims_arr / 2.0
    return (center_arr - half).tolist() + (center_arr + half).tolist()


def node_for_regular(objects_layer, object_id):
    for symbol in (NodeSymbol("O", int(object_id)), NodeSymbol("o", int(object_id))):
        try:
            if objects_layer.has_node(symbol):
                return objects_layer.get_node(symbol)
        except Exception:
            continue
    return None


def load_yaml(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return value if isinstance(value, dict) else {}


def position_from_observations(observations):
    positions = []
    for obs in observations or []:
        value = obs.get("position_world") if isinstance(obs, dict) else None
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 3 and np.all(np.isfinite(arr)):
            positions.append(arr)
    if not positions:
        return None
    return np.mean(positions, axis=0).astype(float).tolist()


def fallback_objects_from_corrections(native_dir, dsg_name):
    corrections = load_yaml(native_dir / "corrections.yaml")
    positions_by_semantic = {}
    positions_path = native_dir / "object_positions.json"
    if positions_path.exists():
        try:
            positions_by_semantic = json.loads(positions_path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"could not read DAAAM object_positions.json: {type(exc).__name__}: {exc}")
            positions_by_semantic = {}
    labels = corrections.get("label_names") or []
    if not labels:
        return []
    fallback = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        semantic_id = label.get("label")
        if semantic_id is None:
            continue
        try:
            semantic_int = int(semantic_id)
        except Exception:
            continue
        raw_label = str(label.get("name") or "object")
        observations = positions_by_semantic.get(str(semantic_int), [])
        position = position_from_observations(observations)
        if position is None:
            continue
        temporal = label.get("temporal_history") if isinstance(label.get("temporal_history"), dict) else {}
        fallback.append(
            {
                "object_id": f"daaam_track_{semantic_int}",
                "native_id": semantic_int,
                "raw_label": raw_label,
                "position_3d": position,
                "dimensions_3d": None,
                "bbox_3d": None,
                "semantic_label": semantic_int,
                "is_background": True,
                "first_observed": temporal.get("first_observed"),
                "last_observed": temporal.get("last_observed"),
                "observation_timestamps": temporal.get("timestamps") or [],
                "dsg_name": dsg_name,
                "provenance": {
                    "layer": "DAAAM_CORRECTIONS_WITH_TRACK_POSITIONS",
                    "native_id": semantic_int,
                    "position_observation_count": len(observations),
                },
            }
        )
    if fallback:
        warnings.append(
            "DAAAM DSG had no native OBJECTS/BACKGROUND_OBJECTS; exported "
            f"{len(fallback)} objects from corrections.yaml + object_positions.json."
        )
    elif positions_by_semantic:
        warnings.append("DAAAM corrections existed but no correction labels matched saved object positions.")
    else:
        warnings.append("DAAAM DSG had no objects and no object_positions.json fallback was available.")
    return fallback


scene_graph = sdsg.DynamicSceneGraph.load(str(payload["dsg_path"]))
dsg_name = Path(payload["dsg_path"]).name
regular_objects = retrieve_objects_from_scene_graph(scene_graph)
objects_layer = scene_graph.get_layer(DsgLayers.OBJECTS)

objects = []
background_object_rows = []
for object_id, object_data in regular_objects.items():
    info = object_data.object_info
    node = node_for_regular(objects_layer, object_id)
    semantic_label = None
    if node is not None:
        try:
            semantic_label = int(node.attributes.semantic_label)
        except Exception:
            semantic_label = None
    position = as_list(info.position)
    dimensions = as_list(info.dimensions)
    objects.append(
        {
            "object_id": f"daaam_object_{object_id}",
            "native_id": int(object_id),
            "raw_label": str(info.description or "object"),
            "position_3d": position,
            "dimensions_3d": dimensions,
            "bbox_3d": bbox_from_center_dims(position, dimensions),
            "semantic_label": semantic_label,
            "is_background": False,
            "first_observed": float(info.first_observed) if info.first_observed is not None else None,
            "last_observed": float(info.last_observed) if info.last_observed is not None else None,
            "observation_timestamps": as_list(object_data.observation_timestamps) or [],
            "dsg_name": dsg_name,
            "provenance": {"layer": "OBJECTS", "native_id": int(object_id)},
        }
    )

try:
    background_objects = extract_background_objects(scene_graph)
except Exception as exc:
    background_objects = {}
    warnings.append(f"could not extract DAAAM background objects: {type(exc).__name__}: {exc}")

for object_id, bg in background_objects.items():
    position = as_list(bg.position)
    background_object_rows.append(
        {
            "object_id": f"daaam_background_{object_id}",
            "native_id": int(object_id),
            "raw_label": str(bg.name or "object"),
            "position_3d": position,
            "dimensions_3d": None,
            "bbox_3d": None,
            "semantic_label": int(bg.semantic_label) if bg.semantic_label is not None else None,
            "is_background": True,
            "first_observed": float(bg.first_observed) if bg.first_observed is not None else None,
            "last_observed": float(bg.last_observed) if bg.last_observed is not None else None,
            "observation_timestamps": list(bg.observation_timestamps or []),
            "dsg_name": dsg_name,
            "provenance": {"layer": "BACKGROUND_OBJECTS", "native_id": int(object_id)},
        }
    )

if not objects and not background_object_rows:
    objects = fallback_objects_from_corrections(native_dir, dsg_name)

track2_index = {"status": "invalid", "reason": "Track2 semantic index was not requested."}
if payload.get("build_track2_index"):
    try:
        config = ToolConfig(
            sentence_embedding_model_name=payload["sentence_embedding_model"],
            clip_model_name=payload["clip_model"],
            clip_backend="openclip",
        )
        unified = get_unified_objects_list(scene_graph, regular_objects)
        features_metadata = dict(scene_graph.metadata.get())
        features_dict = features_metadata.get("features", {}) or {}
        object_embeddings, obj_id_to_idx = precompute_unified_embeddings(unified, features_dict, config)
        if object_embeddings.size == 0:
            raise RuntimeError("no DAAAM object embeddings found in DSG metadata")

        import torch
        from daaam.utils.embedding import CLIPHandler, SentenceEmbeddingHandler, get_combined_embedding

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_handler = CLIPHandler(
            model_name=payload["clip_model"],
            pretrained=payload["clip_pretrained"],
            backend="openclip",
            device=device,
        )
        sentence_handler = SentenceEmbeddingHandler(
            model_name=payload["sentence_embedding_model"],
            device=device,
        )
        labels = [
            line.strip()
            for line in Path(payload["class_names"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        label_clip = clip_handler.extract_text_features(labels)
        label_sentence = sentence_handler.extract_text_embeddings(labels, show_progress=False)
        label_embeddings = []
        for clip_feature, sentence_feature in zip(label_clip, label_sentence):
            embedding = get_combined_embedding(
                clip_embedding=np.asarray(clip_feature),
                sentence_embedding=np.asarray(sentence_feature),
                clip_weight=config.clip_weight,
                sentence_weight=config.sentence_weight,
            )
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            label_embeddings.append(embedding.astype(float).tolist())

        object_rows = []
        native_to_package = {}
        for row in objects:
            native_key = int(row["native_id"])
            native_to_package.setdefault(native_key, []).append(row["object_id"])
        for native_id, emb_idx in obj_id_to_idx.items():
            for package_id in native_to_package.get(int(native_id), []):
                embedding = object_embeddings[int(emb_idx)]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                object_rows.append({"object_id": package_id, "embedding": embedding.astype(float).tolist()})
        if not object_rows:
            raise RuntimeError("DAAAM object embeddings did not map to exported object ids")
        track2_index = {
            "status": "ok",
            "labels": labels,
            "label_embeddings": label_embeddings,
            "objects": object_rows,
        }
    except Exception as exc:
        track2_index = {
            "status": "invalid",
            "reason": f"could not build DAAAM deterministic semantic index: {type(exc).__name__}: {exc}",
        }
        warnings.append(track2_index["reason"])

output = {
    "objects": objects,
    "background_objects": background_object_rows,
    "track2_index": track2_index,
    "warnings": warnings,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
'''


def _cli() -> int:
    try:
        return main(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Adapter-raised errors carry actionable guidance; print the message
        # cleanly rather than dumping a Python traceback. Subprocess/native
        # failures (CalledProcessError) keep their traceback for debugging.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
