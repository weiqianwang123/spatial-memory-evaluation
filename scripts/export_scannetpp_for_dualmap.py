from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/mondo-training-dataset/semantic_mapping/dualmap/scannetpp_dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ScanNet++ iPhone RGB-D frames into DualMap's ScanNet layout."
    )
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _add_claws_to_path(args.claws_root)
    from spatial_rag.datasets.scannetpp_adapter import ScanNetPPAdapter

    scene_root = args.output_root / "exported" / args.scene_id
    color_dir = scene_root / "color"
    depth_dir = scene_root / "depth"
    pose_dir = scene_root / "pose"
    intrinsic_dir = scene_root / "intrinsic"

    if scene_root.exists() and any(scene_root.rglob("*")) and not args.overwrite:
        print(f"DualMap dataset already exists: {scene_root}")
        print("Pass --overwrite to regenerate it.")
        return 0

    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    written = 0
    first_intrinsics = None
    max_frames = None if args.max_frames == 0 else args.max_frames
    with ScanNetPPAdapter(args.dataset_root, args.scene_id, auto_mount=False) as dataset:
        frame_indices = range(0, len(dataset.frame_ids), max(1, args.stride))
        if max_frames is not None:
            frame_indices = list(frame_indices)[:max_frames]
        for out_idx, frame_idx in enumerate(tqdm(frame_indices, desc="exporting")):
            frame = dataset.read_frame(frame_idx)
            if frame is None:
                break

            rgb = frame["rgb"]
            depth_m = frame["depth"]
            color = cv2.cvtColor(
                cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_RGB2BGR,
            )
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            depth_mm = cv2.resize(
                depth_mm,
                (args.width, args.height),
                interpolation=cv2.INTER_NEAREST,
            )

            cv2.imwrite(str(color_dir / f"{out_idx:06d}.jpg"), color)
            cv2.imwrite(str(depth_dir / f"{out_idx:06d}.png"), depth_mm)
            np.savetxt(pose_dir / f"{out_idx:06d}.txt", _pose_matrix(frame), fmt="%.8f")

            if first_intrinsics is None:
                first_intrinsics = _scaled_intrinsic_matrix(
                    frame["camera_intrinsics"],
                    source_shape=rgb.shape[:2],
                    target_shape=(args.height, args.width),
                )
            written += 1

    if first_intrinsics is None:
        raise RuntimeError(f"No frames exported for scene {args.scene_id}")

    np.savetxt(intrinsic_dir / "intrinsic_color.txt", first_intrinsics, fmt="%.8f")
    np.savetxt(intrinsic_dir / "intrinsic_depth.txt", first_intrinsics, fmt="%.8f")
    print(f"exported {written} frames to {scene_root}")
    return 0


def _add_claws_to_path(root: Path) -> None:
    root = root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _scaled_intrinsic_matrix(
    intrinsics: Any,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = source_shape
    dst_h, dst_w = target_shape
    sx = dst_w / src_w
    sy = dst_h / src_h
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = float(intrinsics.fx) * sx
    matrix[1, 1] = float(intrinsics.fy) * sy
    matrix[0, 2] = float(intrinsics.cx) * sx
    matrix[1, 2] = float(intrinsics.cy) * sy
    return matrix


def _pose_matrix(frame: dict[str, Any]) -> np.ndarray:
    pose = frame["camera_pose"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    matrix[:3, 3] = [pose.x, pose.y, pose.z]
    return matrix


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
