#!/usr/bin/env python
"""Build Track 2 referring-query benchmark data from ScanRefer/ScanEnts3D.

Primary source: ScanEnts3D val (https://scanents3d.github.io/), a public superset
of ScanRefer with GT target object_id/object_name + an `entities` array (anchor
instance ids). ScanEnts3D references objects by ScanNet instance id (no bbox in
the json), so Track 2 scores referring at the target object-name level. The
ScanEnts3D val json is staged on NAS at semantic_mapping/scanents3d/.
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
    DEFAULT_SCANENTS3D_VAL_JSON,
    build_track2_data,
    write_unavailable_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 2 ScanEnts3D/ScanRefer benchmark data.")
    parser.add_argument(
        "--referring-json",
        type=Path,
        default=DEFAULT_SCANENTS3D_VAL_JSON,
        help="ScanEnts3D/ScanRefer annotation JSON.",
    )
    parser.add_argument("--scene-id", default=None, help="Restrict to one scene (recommended).")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track2/scanents3d/<scene-id or all>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_name = args.scene_id or "all"
    output_dir = args.output_dir or Path("benchmarks") / "track2" / "scanents3d" / default_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.referring_json).exists():
        summary = write_unavailable_metadata(output_dir, scanrefer_root=args.referring_json)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    summary = build_track2_data(
        scanrefer_json=args.referring_json,
        output_dir=output_dir,
        scene_id=args.scene_id,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
