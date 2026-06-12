from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCANNET_READER_ROOTS = (
    REPO_ROOT / "data" / "scannet",
    Path("/home/robin_wang/open-eqa/data/scannet"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert existing ScanNet .sens files into posed RGB-D sequences for "
            "SpatialRAG/OpenEQA and ScanNet-style layouts for HOV-SG/DualMap."
        )
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/open-eqa-v0.json"))
    parser.add_argument("--episode-prefix", default="scannet-v0")
    parser.add_argument(
        "--episode-history",
        dest="episode_histories",
        action="append",
        help="Only process this OpenEQA episode_history. Can be passed more than once.",
    )
    parser.add_argument(
        "--scene-id",
        dest="scene_ids",
        action="append",
        help="Only process this ScanNet scene id. Can be passed more than once.",
    )
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default=Path("data/raw/scannet"),
        help="Root containing scans/<scene>/<scene>.sens and/or scans_test/...",
    )
    parser.add_argument(
        "--scannet-reader-root",
        type=Path,
        default=None,
        help="Directory containing ScanNet SensorData.py. Defaults to repo data/scannet, then the old open-eqa path.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("data/frames"),
        help="Flat OpenEQA frame root. Output is <frames-root>/<episode_history>/...",
    )
    parser.add_argument(
        "--layout-root",
        type=Path,
        default=Path("data/openeqa_scannet_rgbd"),
        help=(
            "ScanNet-style output root. Output is "
            "<layout-root>/exported/<scene_id>/{color,depth,pose,intrinsic}."
        ),
    )
    parser.add_argument(
        "--layout-key",
        choices=("scene", "episode"),
        default="scene",
        help="Use scene id or sanitized episode history as the exported layout id.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to mirror depth/pose/intrinsics into the ScanNet-style layout.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Frames to export per .sens after frame skipping. Default 0 exports all frames.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Read every Nth frame from the .sens file.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be converted without opening or writing .sens output.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="report path. Default: results/_data/data-prepare/<timestamp>/report.json",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.report is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.report = Path("results") / "_data" / "data-prepare" / timestamp / "report.json"

    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0")

    targets = _load_targets(args)
    if args.limit is not None:
        targets = targets[: args.limit]

    entries: List[Dict[str, Any]] = []
    for target in tqdm(targets, desc="episodes"):
        scene_id = target["scene_id"]
        sens_path = _find_sens(args.scannet_root, scene_id)
        if sens_path is None:
            entry = _base_entry(args, target, None)
            entry.update({"status": "missing_raw", "selected_frame_count": 0})
            entries.append(entry)
            continue

        if args.dry_run:
            entry = _base_entry(args, target, sens_path)
            entry.update({"status": "raw_available", "selected_frame_count": None})
            entries.append(entry)
            continue

        entries.append(_convert_target(args, target, sens_path))

    report = _make_report(args, entries)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report, args.report)
    return 0 if report["num_raw_available"] > 0 else 1


def _load_targets(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.episode_histories:
        return [
            {
                "episode_history": episode_history,
                "scene_id": _scene_id_from_episode(episode_history),
            }
            for episode_history in sorted(set(args.episode_histories))
        ]

    if args.scene_ids:
        return [
            {
                "episode_history": f"{args.episode_prefix}/manual-scannet-{scene_id}",
                "scene_id": scene_id,
            }
            for scene_id in sorted(set(args.scene_ids))
        ]

    with args.dataset.open("r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected list dataset: {args.dataset}")

    targets = []
    seen = set()
    for item in dataset:
        episode_history = str(item.get("episode_history", ""))
        if not episode_history.startswith(args.episode_prefix + "/"):
            continue
        if episode_history in seen:
            continue
        seen.add(episode_history)
        targets.append(
            {
                "episode_history": episode_history,
                "scene_id": _scene_id_from_episode(episode_history),
            }
        )
    return sorted(targets, key=lambda item: item["episode_history"])


def _convert_target(
    args: argparse.Namespace,
    target: Dict[str, str],
    sens_path: Path,
) -> Dict[str, Any]:
    entry = _base_entry(args, target, sens_path)
    frame_dir = Path(entry["frames_dir"])
    layout_dir = Path(entry["layout_dir"])
    color_dir = layout_dir / "color"
    depth_dir = layout_dir / "depth"
    pose_dir = layout_dir / "pose"
    intrinsic_dir = layout_dir / "intrinsic"
    for directory in (frame_dir, color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _clear_managed_outputs(frame_dir, color_dir, depth_dir, pose_dir, intrinsic_dir)

    sensor_data = _load_sensor_data_class(args.scannet_reader_root)
    sensor = sensor_data(str(sens_path))
    selected_indices = list(range(0, len(sensor.frames), args.frame_skip))
    if args.max_frames:
        selected_indices = selected_indices[: args.max_frames]

    _write_intrinsics(sensor, frame_dir, intrinsic_dir, args)

    counts = {
        "flat_rgb_written": 0,
        "flat_depth_written": 0,
        "flat_pose_written": 0,
        "layout_rgb_written": 0,
        "layout_depth_linked": 0,
        "layout_pose_linked": 0,
    }
    frame_iter = tqdm(
        list(enumerate(selected_indices)),
        desc=target["scene_id"],
        leave=False,
    )
    for out_idx, frame_idx in frame_iter:
        _write_frame(sensor, frame_idx, out_idx, frame_dir, color_dir, depth_dir, pose_dir, args, counts)

    entry.update(
        {
            "status": "converted",
            "sens_total_frame_count": len(sensor.frames),
            "selected_frame_count": len(selected_indices),
            "counts": counts,
        }
    )
    return entry


def _load_sensor_data_class(reader_root: Optional[Path]):
    roots = []
    if reader_root is not None:
        roots.append(reader_root)
    env_root = os.environ.get("SPATIAL_MEMORY_SCANNET_READER_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.extend(DEFAULT_SCANNET_READER_ROOTS)

    for root in roots:
        sensor_data_py = root / "SensorData.py"
        if sensor_data_py.exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from SensorData import SensorData

            return SensorData

    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Could not find ScanNet SensorData.py. Pass --scannet-reader-root or set "
        f"SPATIAL_MEMORY_SCANNET_READER_ROOT. Searched: {searched}"
    )


def _write_intrinsics(
    sensor: SensorData,
    frame_dir: Path,
    intrinsic_dir: Path,
    args: argparse.Namespace,
) -> None:
    matrices = {
        "intrinsic_color.txt": sensor.intrinsic_color,
        "intrinsic_depth.txt": sensor.intrinsic_depth,
        "extrinsic_color.txt": sensor.extrinsic_color,
        "extrinsic_depth.txt": sensor.extrinsic_depth,
    }
    for name, matrix in matrices.items():
        flat_path = frame_dir / name
        if args.overwrite or not flat_path.exists():
            np.savetxt(flat_path, matrix, fmt="%f")
        _mirror_file(flat_path, intrinsic_dir / name, args.link_mode, args.overwrite)


def _write_frame(
    sensor: SensorData,
    frame_idx: int,
    out_idx: int,
    frame_dir: Path,
    color_dir: Path,
    depth_dir: Path,
    pose_dir: Path,
    args: argparse.Namespace,
    counts: Dict[str, int],
) -> None:
    frame = sensor.frames[frame_idx]
    flat_rgb = frame_dir / f"{frame_idx:06d}-rgb.png"
    flat_depth = frame_dir / f"{frame_idx:06d}-depth.png"
    flat_pose = frame_dir / f"{frame_idx:06d}.txt"
    layout_rgb = color_dir / f"{out_idx:06d}.jpg"
    layout_depth = depth_dir / f"{out_idx:06d}.png"
    layout_pose = pose_dir / f"{out_idx:06d}.txt"

    need_rgb = args.overwrite or not flat_rgb.exists() or not layout_rgb.exists()
    if need_rgb:
        rgb = None
        if args.overwrite or not flat_rgb.exists():
            rgb = frame.decompress_color(sensor.color_compression_type)
            imageio.imwrite(flat_rgb, rgb)
            counts["flat_rgb_written"] += 1
        if args.overwrite or not layout_rgb.exists():
            if sensor.color_compression_type == "jpeg":
                layout_rgb.write_bytes(frame.color_data)
            else:
                if rgb is None:
                    rgb = frame.decompress_color(sensor.color_compression_type)
                _write_jpeg(layout_rgb, rgb, args.jpeg_quality)
            counts["layout_rgb_written"] += 1

    if args.overwrite or not flat_depth.exists():
        depth_data = frame.decompress_depth(sensor.depth_compression_type)
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
            sensor.depth_height,
            sensor.depth_width,
        )
        if not cv2.imwrite(str(flat_depth), depth):
            raise RuntimeError(f"Failed to write depth PNG: {flat_depth}")
        counts["flat_depth_written"] += 1
    if _mirror_file(flat_depth, layout_depth, args.link_mode, args.overwrite):
        counts["layout_depth_linked"] += 1

    if args.overwrite or not flat_pose.exists():
        np.savetxt(flat_pose, frame.camera_to_world, fmt="%f")
        counts["flat_pose_written"] += 1
    if _mirror_file(flat_pose, layout_pose, args.link_mode, args.overwrite):
        counts["layout_pose_linked"] += 1


def _write_jpeg(path: Path, rgb: np.ndarray, quality: int) -> None:
    quality = max(1, min(100, int(quality)))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to write JPEG: {path}")


def _clear_managed_outputs(
    frame_dir: Path,
    color_dir: Path,
    depth_dir: Path,
    pose_dir: Path,
    intrinsic_dir: Path,
) -> None:
    for pattern in ("*-rgb.png", "*-depth.png", "*.txt"):
        for path in frame_dir.glob(pattern):
            path.unlink()
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        for path in directory.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()


def _mirror_file(src: Path, dst: Path, mode: str, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return False
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        try:
            relative_src = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(relative_src)
        except OSError:
            shutil.copy2(src, dst)
    return True


def _base_entry(
    args: argparse.Namespace,
    target: Dict[str, str],
    sens_path: Optional[Path],
) -> Dict[str, Any]:
    layout_id = _layout_id(target, args.layout_key)
    return {
        "episode_history": target["episode_history"],
        "scene_id": target["scene_id"],
        "layout_id": layout_id,
        "sens_path": str(sens_path) if sens_path else None,
        "frames_dir": str(args.frames_root / target["episode_history"]),
        "layout_dir": str(args.layout_root / "exported" / layout_id),
    }


def _make_report(args: argparse.Namespace, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "dataset": str(args.dataset),
        "episode_prefix": args.episode_prefix,
        "scannet_root": str(args.scannet_root),
        "frames_root": str(args.frames_root),
        "layout_root": str(args.layout_root),
        "layout_key": args.layout_key,
        "max_frames": args.max_frames,
        "frame_skip": args.frame_skip,
        "dry_run": args.dry_run,
        "num_targets": len(entries),
        "num_raw_available": sum(1 for entry in entries if entry["sens_path"]),
        "num_converted": sum(1 for entry in entries if entry["status"] == "converted"),
        "num_missing_raw": sum(1 for entry in entries if entry["status"] == "missing_raw"),
        "total_selected_frames": sum(
            entry.get("selected_frame_count") or 0
            for entry in entries
            if entry["status"] == "converted"
        ),
        "missing_raw_scenes": [
            entry["scene_id"] for entry in entries if entry["status"] == "missing_raw"
        ],
        "entries": entries,
    }


def _print_summary(report: Dict[str, Any], report_path: Path) -> None:
    print(f"targets:          {report['num_targets']}")
    print(f"raw available:    {report['num_raw_available']}/{report['num_targets']}")
    print(f"converted:        {report['num_converted']}/{report['num_targets']}")
    print(f"selected frames:  {report['total_selected_frames']}")
    print(f"wrote report:     {report_path}")
    if report["missing_raw_scenes"]:
        preview = ", ".join(report["missing_raw_scenes"][:10])
        print(f"missing raw:      {preview}")


def _layout_id(target: Dict[str, str], layout_key: str) -> str:
    if layout_key == "scene":
        return target["scene_id"]
    return target["episode_history"].replace("/", "__")


def _scene_id_from_episode(episode_history: str) -> str:
    episode = episode_history.split("/")[-1]
    return episode.split("scannet-")[-1]


def _find_sens(scannet_root: Path, scene_id: str) -> Optional[Path]:
    for split in ("scans", "scans_test"):
        path = scannet_root / split / scene_id / f"{scene_id}.sens"
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
