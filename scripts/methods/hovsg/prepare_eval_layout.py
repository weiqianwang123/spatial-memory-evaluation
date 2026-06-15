from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.hovsg.build_memory_smoke import (
    DEFAULT_CLAWS_ROOT,
    DEFAULT_SCANNETPP_ROOT,
    DEFAULT_SCENE_ID,
    export_scannetpp_iphone_layout,
    _run_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a ScanNet++ scene as a HOV-SG/ScanNet-style RGB-D layout. "
            "This is data preparation for HOV-SG memory/eval runs, not evaluation."
        )
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--layout-root", type=Path, default=Path("data/hovsg_layouts"))
    parser.add_argument(
        "--layout-dir",
        type=Path,
        default=None,
        help="exact output directory; defaults to layout-root/scannetpp_<scene-id>/<run-id>",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="sample every Nth ScanNet++ iPhone frame; default 1 prepares the full scene",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="maximum sampled frames; 0 means all selected frames",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing output layout directory before preparing it",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    run_id = args.run_id or _run_timestamp()
    layout_dir = args.layout_dir or (args.layout_root / f"scannetpp_{args.scene_id}" / run_id)

    if layout_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"layout directory already exists: {layout_dir}. "
                "Pass --overwrite or choose a new --run-id."
            )
        shutil.rmtree(layout_dir)

    summary = export_scannetpp_iphone_layout(
        scannetpp_root=args.scannetpp_root,
        claws_root=args.claws_root,
        scene_id=args.scene_id,
        output_dir=layout_dir,
        frame_stride=args.frame_stride,
        max_frames=None if args.max_frames == 0 else args.max_frames,
    )
    print(json.dumps({"status": "prepared", "layout_dir": str(layout_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
