"""Build a ClawS SpatialRAG sqlite-vec memory DB for a plain ScanNet scene.

ClawS ships a ScanNet++-only build script (`build_scannetpp_spatial_rag_memory.py`
via `ScanNetPPAdapter`). Track 2 (`scene0207_00`) and Track 3 (`scene0709_00`)
are plain ScanNet scenes, so this driver reuses ClawS's generic pipeline
(`build_pipeline` + `SpatialPipeline.process_frame`, the same API the native
build uses) but feeds frames from a prepared DAAAM-style RGB-D layout
(`rgb/<idx>.jpg`, `depth/<idx>.png` uint16 mm, `pose/<idx>.txt` 4x4 cam->world,
`camera_info.json`). No ClawS repo code is modified.

The resulting DB is the same native artifact
(`outputs/scannet_memory_<scene>.db`) that
`scripts/methods/claws/build_memory_package.py` packages for tool_llm eval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
logger = logging.getLogger("build_claws_scannet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ClawS DB for a ScanNet scene from an RGB-D layout.")
    parser.add_argument("--layout-dir", type=Path, required=True, help="rgb/ depth/ pose/ camera_info.json")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--rag-config", type=Path, default=None, help="ClawS RAG config YAML (default: repo default).")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames in layout.")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="uint16 depth -> meters divisor.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-vlm", action="store_true", help="Disable the VLM describer (faster; label from YOLO).")
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=Path("/data/mondo-training-dataset/semantic_mapping/modules/yolo/yolo_world/yolov8l-world.pt"),
        help="Shared OV detector (default: YOLO-World-L from shared modules). "
        "set_classes is applied with the shared class list below.",
    )
    parser.add_argument(
        "--class-list",
        type=Path,
        default=Path("spatial_memory_evaluation/assets/class_lists/detector_coverable.txt"),
        help="Shared OV prompt/eval class list (the Track 1 detector_coverable list); "
        "applied to the OV detector via set_classes so all methods share one vocabulary.",
    )
    parser.add_argument(
        "--vlm-model",
        default="qwen3.5:4b",
        help="Ollama VLM describer model (default qwen3.5:4b, locally available; the "
        "config default qwen3.5:35b is not pulled). Produces snapshot descriptions.",
    )
    return parser.parse_args()


def _read_matrix(path: Path) -> np.ndarray:
    return np.asarray(
        [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()],
        dtype=np.float64,
    )


async def run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.claws_root))
    import cv2
    from scipy.spatial.transform import Rotation

    # Reuse ClawS's own pipeline wiring + config loader.
    from spatial_rag.projection import CameraIntrinsics, RobotPose

    build_script = args.claws_root / "scripts"
    sys.path.insert(0, str(build_script))
    from build_scannetpp_spatial_rag_memory import build_pipeline, load_config  # type: ignore

    cfg = load_config(args.rag_config)
    # Shared OV detector + shared class prompt for ALL detector-based methods:
    # point ClawS's trigger at YOLO-World-L and prompt it with the Track 1 class
    # list (applied via set_classes after the model loads, below).
    class_list = [
        line.strip()
        for line in Path(args.class_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        cfg.trigger.model = str(args.detector_model)
    except Exception:
        pass
    # VLM describer: enable so ClawS stores rich `**label** description` snapshots
    # (like DAAAM's DAM grounding); off only with --no-vlm.
    if hasattr(cfg, "vlm"):
        try:
            cfg.vlm.enabled = not args.no_vlm
            if not args.no_vlm:
                cfg.vlm.model = args.vlm_model
                cfg.vlm.provider = "ollama"
                cfg.vlm.endpoint = "http://localhost:11434"
        except Exception:
            pass

    db_path = args.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    camera_info = json.loads((args.layout_dir / "camera_info.json").read_text())
    k = np.asarray(camera_info["intrinsics"], dtype=np.float64)
    intrinsics = CameraIntrinsics(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]))

    rgb_paths = sorted((args.layout_dir / "rgb").glob("*.jpg"))
    if not rgb_paths:
        rgb_paths = sorted((args.layout_dir / "rgb").glob("*.png"))
    if not rgb_paths:
        raise FileNotFoundError(f"no rgb frames in {args.layout_dir/'rgb'}")

    pipeline, rag_service = await build_pipeline(cfg, db_path)

    # Apply the shared OV class prompt to the detector (ClawS's UltralyticsBackend
    # never calls set_classes itself, so an open-vocab YOLO-World model would
    # otherwise detect nothing / its defaults). This makes ClawS use the SAME
    # detector + class list as Track 1 and the other detector-based methods.
    backend = getattr(pipeline, "_backend", None)
    model = getattr(backend, "_model", None)
    if model is not None and hasattr(model, "set_classes") and class_list:
        try:
            model.set_classes(class_list)
            print(f"set_classes OK: {len(class_list)} shared OV labels on {args.detector_model.name}")
        except Exception as exc:
            print(f"warning: set_classes failed ({exc}); detector runs with its default vocabulary")
    else:
        print(f"note: detector {args.detector_model.name} has no set_classes (closed-vocab); using its built-in classes")

    frames_seen = 0
    stored_events = 0
    start = time.monotonic()
    try:
        stride = max(1, args.stride)
        for idx in range(0, len(rgb_paths), stride):
            if args.max_frames > 0 and frames_seen >= args.max_frames:
                break
            stem = rgb_paths[idx].stem
            depth_path = args.layout_dir / "depth" / f"{stem}.png"
            pose_path = args.layout_dir / "pose" / f"{stem}.txt"
            if not (depth_path.exists() and pose_path.exists()):
                continue
            pose = _read_matrix(pose_path)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                continue

            rgb_bgr = cv2.imread(str(rgb_paths[idx]))
            depth_u16 = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth_u16 is None:
                continue
            depth_m = depth_u16.astype(np.float32) / float(args.depth_scale)

            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()  # scalar-last
            robot_pose = RobotPose(
                x=float(pose[0, 3]), y=float(pose[1, 3]), z=float(pose[2, 3]),
                qx=float(quat[0]), qy=float(quat[1]), qz=float(quat[2]), qw=float(quat[3]),
            )
            pipeline.projector.update_intrinsics(intrinsics, depth_shape=depth_m.shape[:2])
            result = await pipeline.process_frame(rgb_bgr, depth_m, robot_pose, float(frames_seen))
            frames_seen += 1
            stored_events += result.stored_count
            if getattr(cfg.vlm, "enabled", False) and hasattr(pipeline, "wait_vlm_pending"):
                from build_scannetpp_spatial_rag_memory import pending_vlm_count  # type: ignore
                if pending_vlm_count(pipeline) >= 8:
                    await pipeline.wait_vlm_pending(timeout=60.0)
            if frames_seen == 1 or frames_seen % max(1, args.log_every) == 0:
                count = await rag_service.storage.count()
                logger.info("frames=%d stored_events=%d memories=%d", frames_seen, stored_events, count)

        if getattr(cfg.vlm, "enabled", False) and hasattr(pipeline, "wait_vlm_pending"):
            await pipeline.wait_vlm_pending(timeout=120.0)
        count = await rag_service.storage.count()
        elapsed = round(time.monotonic() - start, 1)
        stats = {
            "status": "ok", "scene_id": args.scene_id, "frames_processed": frames_seen,
            "stored_events": stored_events, "memory_records": count, "db_path": str(db_path),
            "elapsed_s": elapsed,
            "time_per_frame_seconds": round(elapsed / frames_seen, 4) if frames_seen else None,
        }
        # Sidecar next to the DB so the package builder can record frame_count +
        # time_per_frame in build_log.json (the package step is a separate process).
        try:
            Path(str(db_path) + ".build_stats.json").write_text(json.dumps(stats, indent=2))
        except Exception:
            pass
        print(json.dumps(stats, indent=2))
    finally:
        await rag_service.storage.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
