#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track3.evaluator import evaluate_track3
from spatial_memory_evaluation.track3.judge import make_cli_judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Track 3 OpenEQA general spatial QA (ScanNet + HM3D)."
    )
    parser.add_argument("package_dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track3/openeqa/<dataset>",
    )
    parser.add_argument("--dataset", choices=("scannet", "hm3d"), default="scannet")
    parser.add_argument("--mode", choices=("fixed_api", "tool_llm"), default="fixed_api")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--llm-command",
        default=None,
        help="Command template for --mode tool_llm. Placeholders: {prompt_path}, {sandbox_dir}, {output_path}.",
    )
    parser.add_argument(
        "--judge-command",
        default=None,
        help=(
            "LLM-Match judge command template with a {prompt_path} placeholder "
            "(must print a 1-5 rating). Kept separate from the answering LLM. "
            "Without it, a transparent exact/substring fallback judge is used."
        ),
    )
    parser.add_argument("--max-tool-iterations", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir or Path("benchmarks") / "track3" / "openeqa" / args.dataset
    judge = make_cli_judge(args.judge_command) if args.judge_command else None
    summary = evaluate_track3(
        package_dir=args.package_dir,
        benchmark_dir=benchmark_dir,
        mode=args.mode,
        output=args.output,
        llm_command=args.llm_command,
        max_tool_iterations=args.max_tool_iterations,
        judge=judge,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
