"""Convert extracted ScanNet frames into a DAAAM RGB-D layout.

The DAAAM adapter (`build_memory_smoke.py`) consumes a layout with:

```text
<layout_dir>/
  rgb/<idx:06d>.jpg          # color frames
  depth/<idx:06d>.png        # uint16 depth in millimeters
  pose/<idx:06d>.txt         # 4x4 camera->world pose
  intrinsic/intrinsic_color.txt, intrinsic_depth.txt   # 4x4
  camera_info.json           # {width,height,intrinsics 3x3,distortion}
  layout_summary.json
```

ScanNet scenes for Track 2 (`scene0207_00`) and Track 3 (`scene0709_00`) are not
ScanNet++ iPhone captures, so the existing ScanNet++ exporter does not apply.
Their frames were already extracted from the `.sens` files (OpenEQA SensorData)
into a directory of `{frame:06d}-rgb.png`, `{frame:06d}-depth.png`,
`{frame:06d}.txt` (4x4 pose), plus `intrinsic_color.txt`/`intrinsic_depth.txt`.
This script samples that directory and writes the DAAAM layout above, scaling
the color intrinsic to the depth resolution (DAAAM unprojects on the depth
frame), exactly like the ScanNet++ exporter does.

Depth is written as uint16 millimeters (DAAAM `--depth-scale 1000`). Frames with
a non-finite or all-zero pose are skipped (ScanNet marks lost-tracking frames
with `-inf`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a DAAAM RGB-D layout from extracted ScanNet frames.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of {idx:06d}-rgb.png/-depth.png/.txt.")
    parser.add_argument("--output-dir", type=Path, required=True, help="DAAAM layout output dir.")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=200)
    return parser.parse_args()


def _read_matrix(path: Path) -> np.ndarray:
    rows = [
        [float(v) for v in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.asarray(rows, dtype=np.float64)


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{value:.6f}" for value in row) for row in matrix]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scale_intrinsic(intrinsic: np.ndarray, rgb_shape: tuple[int, int], depth_shape: tuple[int, int]) -> np.ndarray:
    """Scale a color-frame 3x3/4x4 intrinsic to the depth resolution."""
    scaled = np.array(intrinsic, dtype=np.float64)
    sx = depth_shape[1] / float(rgb_shape[1])
    sy = depth_shape[0] / float(rgb_shape[0])
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _pose_is_valid(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    return not np.allclose(pose[:3, :3], 0.0)


def export_scannet_layout(
    *,
    frames_dir: Path,
    output_dir: Path,
    scene_id: str,
    frame_stride: int,
    max_frames: int | None,
) -> dict[str, Any]:
    import cv2

    frame_ids = sorted(
        p.stem.split("-")[0]
        for p in (*frames_dir.glob("*-rgb.jpg"), *frames_dir.glob("*-rgb.png"))
    )
    if not frame_ids:
        raise FileNotFoundError(f"no *-rgb.jpg/*-rgb.png frames in {frames_dir}")
    selected = frame_ids[:: max(1, frame_stride)]

    color_intrinsic = _read_matrix(frames_dir / "intrinsic_color.txt")
    if color_intrinsic.shape == (4, 4):
        color_intrinsic = color_intrinsic[:3, :3]

    color_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    pose_dir = output_dir / "pose"
    intrinsic_dir = output_dir / "intrinsic"
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    skipped = 0
    depth_intrinsic_written = False
    depth_intrinsic: np.ndarray | None = None
    rgb_shape: tuple[int, int] | None = None
    depth_shape: tuple[int, int] | None = None

    for frame_id in selected:
        if max_frames is not None and len(exported) >= max_frames:
            break
        pose_path = frames_dir / f"{frame_id}.txt"
        rgb_jpg = frames_dir / f"{frame_id}-rgb.jpg"
        rgb_path = rgb_jpg if rgb_jpg.exists() else frames_dir / f"{frame_id}-rgb.png"
        depth_path = frames_dir / f"{frame_id}-depth.png"
        if not (pose_path.exists() and rgb_path.exists() and depth_path.exists()):
            skipped += 1
            continue
        pose = _read_matrix(pose_path)
        if not _pose_is_valid(pose):
            skipped += 1
            continue
        bgr = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            skipped += 1
            continue
        if rgb_shape is None:
            rgb_shape = (bgr.shape[0], bgr.shape[1])
            depth_shape = (depth.shape[0], depth.shape[1])
            depth_intrinsic = _scale_intrinsic(color_intrinsic, rgb_shape, depth_shape)

        # DAAAM unprojects on the depth frame; resize color to depth resolution.
        bgr_resized = cv2.resize(bgr, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_AREA)
        depth_u16 = depth.astype(np.uint16) if depth.dtype != np.uint16 else depth

        stem = f"{len(exported):06d}"
        cv2.imwrite(str(color_dir / f"{stem}.jpg"), bgr_resized)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_u16)
        _write_matrix(pose_dir / f"{stem}.txt", pose)
        exported.append({"output_index": len(exported), "frame_id": frame_id})

    if not exported:
        raise RuntimeError(f"no valid frames exported from {frames_dir} (skipped {skipped})")

    intrinsic_4x4 = np.eye(4)
    intrinsic_4x4[:3, :3] = depth_intrinsic
    _write_matrix(intrinsic_dir / "intrinsic_color.txt", intrinsic_4x4)
    _write_matrix(intrinsic_dir / "intrinsic_depth.txt", intrinsic_4x4)

    camera_info = {
        "width": int(depth_shape[1]),
        "height": int(depth_shape[0]),
        "intrinsics": depth_intrinsic.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    (output_dir / "camera_info.json").write_text(json.dumps(camera_info, indent=2) + "\n", encoding="utf-8")

    summary = {
        "layout_dir": str(output_dir),
        "scene_id": scene_id,
        "frame_count": len(exported),
        "frame_stride": frame_stride,
        "skipped_frames": skipped,
        "source_frames_dir": str(frames_dir),
        "depth_scale": 1000.0,
        "rgb_dir": str(color_dir),
        "depth_dir": str(depth_dir),
        "pose_dir": str(pose_dir),
        "camera_info_path": str(output_dir / "camera_info.json"),
    }
    (output_dir / "layout_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = export_scannet_layout(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        frame_stride=args.frame_stride,
        max_frames=None if args.max_frames == 0 else args.max_frames,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
