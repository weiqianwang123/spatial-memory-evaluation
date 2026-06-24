#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track1.data import (
    DEFAULT_SCANNETPP_ROOT,
    DEFAULT_SCENE_ID,
    DEFAULT_TOP_K,
    build_track1_data,
    build_track1_scannet_data,
)
from spatial_memory_evaluation.track2.scannet_bbox import DEFAULT_SCANNET_SCANS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Track 1 object-location benchmark data (GT inventory + queries)."
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument(
        "--dataset",
        choices=("scannetpp", "scannet"),
        default="scannetpp",
        help="ScanNet++ (iPhone capture) or ScanNet (.sens scans). Track 1 supports both.",
    )
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scannet-scans-root", type=Path, default=DEFAULT_SCANNET_SCANS_ROOT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track1/<dataset>/<scene-id>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path("benchmarks") / "track1" / args.dataset / args.scene_id
    if args.dataset == "scannet":
        summary = build_track1_scannet_data(
            scannet_scans_root=args.scannet_scans_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            top_k=args.top_k,
        )
    else:
        summary = build_track1_data(
            scannetpp_root=args.scannetpp_root,
            scene_id=args.scene_id,
            output_dir=output_dir,
            top_k=args.top_k,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
