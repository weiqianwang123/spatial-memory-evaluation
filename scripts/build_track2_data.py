#!/usr/bin/env python
"""Build Track 2 ScanRefer referring-query benchmark data.

Skeleton: ScanRefer annotations are not yet on NAS. This CLI either builds the
benchmark (once `build_track2_data` is implemented) or writes a `data_unavailable`
metadata stub. See `.codex/path_registry.md` (Track 2/3 datasets to acquire).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track2.data import (
    DEFAULT_SCANNET_ROOT,
    build_track2_data,
    write_unavailable_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 2 ScanRefer benchmark data.")
    parser.add_argument("--scanrefer-root", type=Path, default=None, help="ScanRefer annotation root (TBD).")
    parser.add_argument("--scannet-root", type=Path, default=DEFAULT_SCANNET_ROOT)
    parser.add_argument("--scannet-split", default="val")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path("benchmarks") / "track2" / "scanrefer" / args.scannet_split
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.scanrefer_root is None or not Path(args.scanrefer_root).exists():
        summary = write_unavailable_metadata(output_dir, scanrefer_root=args.scanrefer_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    summary = build_track2_data(
        scanrefer_root=args.scanrefer_root,
        scannet_root=args.scannet_root,
        output_dir=output_dir,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
