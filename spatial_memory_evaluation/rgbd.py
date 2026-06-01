from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .interfaces import RGBDFrame, RGBDSequence


def load_open_eqa_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with dataset_path.open("r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError(f"Expected a list in {dataset_path}")
    return dataset


def iter_episode_histories(
    dataset: Sequence[Dict[str, Any]],
    episode_prefix: Optional[str] = None,
) -> Iterable[str]:
    seen = set()
    for item in dataset:
        episode_history = item.get("episode_history")
        if not episode_history:
            continue
        if episode_prefix and not episode_history.startswith(episode_prefix):
            continue
        if episode_history in seen:
            continue
        seen.add(episode_history)
        yield episode_history


def questions_for_episode(
    dataset: Sequence[Dict[str, Any]],
    episode_history: str,
) -> List[Dict[str, Any]]:
    return [item for item in dataset if item.get("episode_history") == episode_history]


def load_rgbd_sequence(
    frames_root: Path,
    episode_history: str,
    max_frames: Optional[int] = None,
    frame_stride: int = 1,
) -> RGBDSequence:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    root = frames_root / episode_history
    if not root.exists():
        raise FileNotFoundError(f"Episode frames not found: {root}")

    rgb_paths = sorted(root.glob("*-rgb.png"))
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB frames matching *-rgb.png in {root}")

    rgb_paths = rgb_paths[::frame_stride]
    if max_frames is not None:
        rgb_paths = rgb_paths[:max_frames]

    frames = []
    for position, rgb_path in enumerate(rgb_paths):
        frame_id = rgb_path.name.replace("-rgb.png", "")
        index = _safe_frame_index(frame_id, fallback=position)
        depth_path = root / f"{frame_id}-depth.png"
        pose_path = root / f"{frame_id}.txt"
        frames.append(
            RGBDFrame(
                index=index,
                rgb_path=rgb_path,
                depth_path=depth_path if depth_path.exists() else None,
                pose_path=pose_path if pose_path.exists() else None,
            )
        )

    return RGBDSequence(
        episode_history=episode_history,
        root=root,
        frames=frames,
        intrinsic_color_path=_optional_file(root / "intrinsic_color.txt"),
        intrinsic_depth_path=_optional_file(root / "intrinsic_depth.txt"),
        extrinsic_color_path=_optional_file(root / "extrinsic_color.txt"),
        extrinsic_depth_path=_optional_file(root / "extrinsic_depth.txt"),
        metadata={
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "num_frames": len(frames),
        },
    )


def _optional_file(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _safe_frame_index(frame_id: str, fallback: int) -> int:
    try:
        return int(frame_id)
    except ValueError:
        return fallback
