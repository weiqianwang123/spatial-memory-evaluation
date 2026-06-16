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
    build_track1_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 1 object-inventory benchmark data.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track1/scannetpp/<scene-id>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path("benchmarks") / "track1" / "scannetpp" / args.scene_id
    summary = build_track1_data(
        scannetpp_root=args.scannetpp_root,
        scene_id=args.scene_id,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
