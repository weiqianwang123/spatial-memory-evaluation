from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check raw ScanNet .sens and extracted RGB-D frames for OpenEQA ScanNet."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/open-eqa-v0.json"))
    parser.add_argument("--scannet-root", type=Path, default=Path("data/raw/scannet"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("spatial-memory-evaluation/results/openeqa-scannet-data-report.json"),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args.dataset)
    episodes = _scannet_episodes(dataset)

    entries = []
    for episode_history in episodes:
        scene_id = _scene_id_from_episode(episode_history)
        sens_path = _find_sens(args.scannet_root, scene_id)
        frame_dir = args.frames_root / episode_history
        rgb_count = len(list(frame_dir.glob("*-rgb.png"))) if frame_dir.exists() else 0
        depth_count = len(list(frame_dir.glob("*-depth.png"))) if frame_dir.exists() else 0
        pose_count = len(list(frame_dir.glob("*.txt"))) if frame_dir.exists() else 0
        has_intrinsics = (frame_dir / "intrinsic_depth.txt").exists()
        entries.append(
            {
                "episode_history": episode_history,
                "scene_id": scene_id,
                "sens_path": str(sens_path) if sens_path else None,
                "frames_dir": str(frame_dir),
                "rgb_count": rgb_count,
                "depth_count": depth_count,
                "pose_count": pose_count,
                "has_intrinsics": has_intrinsics,
                "raw_ok": sens_path is not None,
                "rgbd_ok": rgb_count > 0 and depth_count > 0 and pose_count > 0 and has_intrinsics,
            }
        )

    report = {
        "dataset": str(args.dataset),
        "scannet_root": str(args.scannet_root),
        "frames_root": str(args.frames_root),
        "num_episodes": len(entries),
        "num_raw_ok": sum(1 for entry in entries if entry["raw_ok"]),
        "num_rgbd_ok": sum(1 for entry in entries if entry["rgbd_ok"]),
        "missing_raw_scenes": [
            entry["scene_id"] for entry in entries if not entry["raw_ok"]
        ],
        "missing_rgbd_episodes": [
            entry["episode_history"] for entry in entries if not entry["rgbd_ok"]
        ],
        "entries": entries,
    }
    with args.report.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"OpenEQA ScanNet episodes: {report['num_episodes']}")
    print(f"raw .sens available:      {report['num_raw_ok']}/{report['num_episodes']}")
    print(f"extracted RGB-D ready:    {report['num_rgbd_ok']}/{report['num_episodes']}")
    print(f"wrote report:             {args.report}")

    if report["missing_raw_scenes"]:
        preview = ", ".join(report["missing_raw_scenes"][:10])
        print(f"missing raw ScanNet scenes: {preview}")
    if report["missing_rgbd_episodes"]:
        preview = ", ".join(report["missing_rgbd_episodes"][:5])
        print(f"missing RGB-D episodes: {preview}")
    return 0 if report["num_rgbd_ok"] == report["num_episodes"] else 1


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset: {path}")
    return data


def _scannet_episodes(dataset: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            item["episode_history"]
            for item in dataset
            if str(item.get("episode_history", "")).startswith("scannet-v0/")
        }
    )


def _scene_id_from_episode(episode_history: str) -> str:
    episode = episode_history.split("/")[-1]
    return episode.split("scannet-")[-1]


def _find_sens(scannet_root: Path, scene_id: str) -> Path | None:
    for split in ("scans", "scans_test"):
        path = scannet_root / split / scene_id / f"{scene_id}.sens"
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
