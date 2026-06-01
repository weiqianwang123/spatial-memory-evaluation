from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_METHOD = "adapters.claws_spatial_rag:create_method"
DEFAULT_KWARGS = Path(
    "spatial-memory-evaluation/configs/claws_openeqa_scannet_method_kwargs.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ClawS SpatialRAG on OpenEQA ScanNet RGB-D episodes."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/open-eqa-v0.json"))
    parser.add_argument("--scannet-root", type=Path, default=Path("data/raw/scannet"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--method-kwargs", type=Path, default=DEFAULT_KWARGS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spatial-memory-evaluation/results/openeqa-scannet-memory.json"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--extract-rgbd",
        action="store_true",
        help="first run data/scannet/extract-frames.py to create RGB-D frames",
    )
    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="skip preflight RGB-D availability check",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    if args.extract_rgbd:
        _run(
            [
                sys.executable,
                "data/scannet/extract-frames.py",
                "--dataset",
                str(args.dataset),
                "--scannet-root",
                str(args.scannet_root),
                "--output-directory",
                str(args.frames_root),
            ],
            cwd=Path.cwd(),
        )

    if not args.skip_data_check:
        check = subprocess.run(
            [
                sys.executable,
                "spatial-memory-evaluation/scripts/check_openeqa_scannet_data.py",
                "--dataset",
                str(args.dataset),
                "--scannet-root",
                str(args.scannet_root),
                "--frames-root",
                str(args.frames_root),
            ],
            cwd=Path.cwd(),
            check=False,
        )
        if check.returncode != 0:
            print(
                "RGB-D data is not ready. Provide ScanNet .sens files and run with "
                "--extract-rgbd, or pre-extract data/frames/scannet-v0 first."
            )
            return check.returncode

    command = [
        sys.executable,
        "-m",
        "spatial_memory_evaluation.run_memory",
        "--method",
        args.method,
        "--method-kwargs",
        str(args.method_kwargs),
        "--dataset",
        str(args.dataset),
        "--frames-root",
        str(args.frames_root),
        "--episode-prefix",
        "scannet-v0",
        "--output",
        str(args.output),
        "--frame-stride",
        str(args.frame_stride),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    if args.dry_run:
        command.append("--dry-run")
    if args.force:
        command.append("--force")

    _run(command, cwd=Path.cwd())
    return 0


def _run(command: list[str], cwd: Path) -> None:
    print("running:")
    print(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
