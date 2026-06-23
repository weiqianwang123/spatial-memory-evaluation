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
from spatial_memory_evaluation.track1.evaluator import evaluate_track1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Track 1 object-level location query + build cost."
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track1/scannetpp/<scene-id>",
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--mode", choices=("fixed_api", "tool_llm"), default="fixed_api")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--llm-command",
        default=None,
        help=(
            "Command template for --mode tool_llm. Placeholders: "
            "{prompt_path}, {sandbox_dir}, {output_path}."
        ),
    )
    parser.add_argument("--max-tool-iterations", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir or Path("benchmarks") / "track1" / "scannetpp" / args.scene_id
    summary = evaluate_track1(
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
