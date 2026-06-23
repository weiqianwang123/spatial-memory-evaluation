#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track2.evaluator import evaluate_track2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Track 2 ScanRefer instance-level referring queries."
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track2/scanrefer/<scannet-split>",
    )
    parser.add_argument("--scannet-split", default="val")
    parser.add_argument("--mode", choices=("fixed_api", "tool_llm"), default="fixed_api")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--llm-command",
        default=None,
        help="Command template for --mode tool_llm. Placeholders: {prompt_path}, {sandbox_dir}, {output_path}.",
    )
    parser.add_argument("--max-tool-iterations", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir or Path("benchmarks") / "track2" / "scanrefer" / args.scannet_split
    summary = evaluate_track2(
        package_dir=args.package_dir,
        benchmark_dir=benchmark_dir,
        mode=args.mode,
        output=args.output,
        llm_command=args.llm_command,
        max_tool_iterations=args.max_tool_iterations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
