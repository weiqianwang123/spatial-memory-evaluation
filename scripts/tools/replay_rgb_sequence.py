#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay or export a ScanNet++ scene RGB sequence.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--rgb-path", type=Path, default=None, help="override RGB video path")
    parser.add_argument(
        "--pose-intrinsic-path",
        type=Path,
        default=None,
        help="override pose/intrinsic JSON path used only for frame ids",
    )
    parser.add_argument("--stride", type=int, default=1, help="replay every Nth RGB frame")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None, help="exclusive source-frame index")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means no limit after stride sampling")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=960, help="resize output/display width; 0 keeps source size")
    parser.add_argument("--no-display", action="store_true", help="do not open an OpenCV display window")
    parser.add_argument("--save-video", type=Path, default=None, help="optional output video path, e.g. /tmp/replay.mp4")
    parser.add_argument("--save-frames-dir", type=Path, default=None, help="optional directory for sampled JPEG frames")
    parser.add_argument("--no-overlay", action="store_true", help="do not draw scene/frame text on frames")
    parser.add_argument("--list-only", action="store_true", help="print selected frame indices and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("--end-index must be greater than --start-index")

    import cv2

    rgb_path = args.rgb_path or _default_rgb_path(args.scannetpp_root, args.scene_id)
    pose_path = args.pose_intrinsic_path or _default_pose_path(args.scannetpp_root, args.scene_id)
    frame_ids = _load_frame_ids(pose_path)

    cap = cv2.VideoCapture(str(rgb_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RGB video: {rgb_path}")
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_count = len(frame_ids) if frame_ids else video_frame_count
    if source_count <= 0:
        raise RuntimeError(f"could not determine frame count for {rgb_path}")

    end_index = min(args.end_index if args.end_index is not None else source_count, source_count)
    selected = list(range(args.start_index, end_index, args.stride))
    if args.max_frames > 0:
        selected = selected[: args.max_frames]
    if not selected:
        raise ValueError("no frames selected")

    summary = {
        "scene_id": args.scene_id,
        "rgb_path": str(rgb_path),
        "pose_intrinsic_path": str(pose_path) if pose_path.exists() else None,
        "video_frame_count": video_frame_count,
        "metadata_frame_count": len(frame_ids),
        "selected_count": len(selected),
        "start_index": args.start_index,
        "end_index": end_index,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "fps": args.fps,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.list_only:
        for output_index, source_index in enumerate(selected):
            frame_id = frame_ids[source_index] if source_index < len(frame_ids) else None
            print(json.dumps({"output_index": output_index, "source_index": source_index, "frame_id": frame_id}))
        cap.release()
        return 0

    if args.no_display and args.save_video is None and args.save_frames_dir is None:
        print("Nothing to display or save; use --save-video, --save-frames-dir, or omit --no-display.", file=sys.stderr)
        cap.release()
        return 0

    if args.save_frames_dir is not None:
        args.save_frames_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    delay = 1.0 / args.fps
    paused = False
    last_tick = time.perf_counter()

    try:
        for output_index, source_index in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_index} from {rgb_path}")
            frame = _resize_width(frame, args.width)
            frame_id = frame_ids[source_index] if source_index < len(frame_ids) else None
            if not args.no_overlay:
                _draw_overlay(frame, args.scene_id, output_index, len(selected), source_index, frame_id)

            if args.save_video is not None:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.save_video),
                        _fourcc_for(args.save_video),
                        args.fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"failed to open video writer: {args.save_video}")
                writer.write(frame)

            if args.save_frames_dir is not None:
                cv2.imwrite(str(args.save_frames_dir / f"{output_index:06d}_src{source_index:06d}.jpg"), frame)

            if not args.no_display:
                cv2.imshow(f"RGB replay {args.scene_id}", frame)
                while paused:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (ord("q"), 27):
                        return 0
                    if key == ord(" "):
                        paused = False
                elapsed = time.perf_counter() - last_tick
                wait_ms = max(1, int(max(0.0, delay - elapsed) * 1000.0))
                key = cv2.waitKey(wait_ms) & 0xFF
                last_tick = time.perf_counter()
                if key in (ord("q"), 27):
                    return 0
                if key == ord(" "):
                    paused = True
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def _default_rgb_path(scannetpp_root: Path, scene_id: str) -> Path:
    scene_dir = scannetpp_root / "data" / scene_id / "iphone"
    for name in ("rgb.mkv", "rgb.mp4", "rgb.mov"):
        path = scene_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"could not find rgb.mkv/rgb.mp4/rgb.mov under {scene_dir}")


def _default_pose_path(scannetpp_root: Path, scene_id: str) -> Path:
    return scannetpp_root / "data" / scene_id / "iphone" / "pose_intrinsic_imu.json"


def _load_frame_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return sorted(value, key=_frame_sort_key)


def _frame_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits) if digits else -1, text)


def _resize_width(frame: Any, width: int) -> Any:
    if width <= 0:
        return frame
    import cv2

    height, current_width = frame.shape[:2]
    if current_width == width:
        return frame
    new_height = max(1, round(height * width / current_width))
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)


def _draw_overlay(
    frame: Any,
    scene_id: str,
    output_index: int,
    selected_count: int,
    source_index: int,
    frame_id: str | None,
) -> None:
    import cv2

    text = f"{scene_id}  {output_index + 1}/{selected_count}  src={source_index}"
    if frame_id is not None:
        text += f"  id={frame_id}"
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 1, 8 + 12 * len(text)), 42), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def _fourcc_for(path: Path) -> int:
    import cv2

    if path.suffix.lower() == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if path.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"MJPG")


if __name__ == "__main__":
    raise SystemExit(main())
