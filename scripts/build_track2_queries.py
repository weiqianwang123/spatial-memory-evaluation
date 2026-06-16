#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track1.data import DEFAULT_SCENE_ID
from spatial_memory_evaluation.track2.data import build_track2_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 2 object-location query data.")
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument(
        "--track1-benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track1/scannetpp/<scene-id>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track2/scannetpp/<scene-id>",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    track1_dir = args.track1_benchmark_dir or Path("benchmarks") / "track1" / "scannetpp" / args.scene_id
    output_dir = args.output_dir or Path("benchmarks") / "track2" / "scannetpp" / args.scene_id
    summary = build_track2_queries(
        track1_benchmark_dir=track1_dir,
        output_dir=output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
