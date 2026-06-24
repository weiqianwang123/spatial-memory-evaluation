"""Extract sampled RGB+depth+pose frames from a ScanNet .sens file.

Writes `{idx:06d}-rgb.png`, `{idx:06d}-depth.png` (uint16 mm), `{idx:06d}.txt`
(4x4 pose), plus `intrinsic_color.txt`/`intrinsic_depth.txt` into an output dir
matching the layout that `export_scannet_layout.py` consumes. Sampling keeps the
read lean for multi-GB .sens files (the full decode of scene0207_00 is ~1.1GB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

OPENEQA_SCANNET = Path("/home/robin_wang/open-eqa/data/scannet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a ScanNet .sens file.")
    parser.add_argument("--sens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(OPENEQA_SCANNET))
    import imageio
    from SensorData import SensorData

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sd = SensorData(str(args.sens))

    # Intrinsics (4x4 -> file).
    np.savetxt(args.output_dir / "intrinsic_color.txt", sd.intrinsic_color, fmt="%.6f")
    np.savetxt(args.output_dir / "intrinsic_depth.txt", sd.intrinsic_depth, fmt="%.6f")

    indices = list(range(0, len(sd.frames), max(1, args.frame_skip)))
    if args.max_frames:
        indices = indices[: args.max_frames]

    import cv2  # JPEG encode is ~25x faster than imageio PNG for the 1.3MP color frame.

    exported = 0
    for out_idx, src in enumerate(indices):
        frame = sd.frames[src]
        pose = frame.camera_to_world
        if not np.all(np.isfinite(pose)):
            continue
        color = frame.decompress_color(sd.color_compression_type)  # RGB
        depth = frame.decompress_depth(sd.depth_compression_type)
        depth = np.frombuffer(depth, dtype=np.uint16).reshape(sd.depth_height, sd.depth_width)
        stem = f"{out_idx:06d}"
        # Color -> JPEG (lossy is fine: every downstream method resizes to depth
        # resolution anyway). Depth stays PNG (uint16, must be lossless).
        cv2.imwrite(str(args.output_dir / f"{stem}-rgb.jpg"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        imageio.imwrite(args.output_dir / f"{stem}-depth.png", depth)
        np.savetxt(args.output_dir / f"{stem}.txt", pose, fmt="%.6f")
        exported += 1

    print(f"exported {exported} frames (of {len(indices)} sampled, {len(sd.frames)} total) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
