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
from spatial_memory_evaluation.track2.evaluator import evaluate_track2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track 2 object-location queries.")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track2/scannetpp/<scene-id>",
    )
    parser.add_argument(
        "--track1-benchmark-dir",
        type=Path,
        default=None,
        help="Default: benchmarks/track1/scannetpp/<scene-id>",
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument(
        "--mode",
        choices=("fixed_api", "agentic_memory_only", "agentic_full_access"),
        default="fixed_api",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--agent-command", default=None)
    parser.add_argument("--agent-output", type=Path, default=None)
    parser.add_argument("--sandbox-root", type=Path, default=None)
    parser.add_argument(
        "--agent-extra-path",
        action="append",
        type=Path,
        default=None,
        help="optional file/directory to copy into the agent sandbox; repeatable",
    )
    parser.add_argument(
        "--agent-include-build-code",
        action="store_true",
        help="deprecated alias for including source code context; source code is included by default",
    )
    parser.add_argument(
        "--no-agent-include-source-code",
        action="store_false",
        dest="agent_include_source_code",
        default=True,
        help="do not copy scripts/methods adapters, shared_modules, or the method root repo source tree",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_dir = args.benchmark_dir or Path("benchmarks") / "track2" / "scannetpp" / args.scene_id
    track1_dir = args.track1_benchmark_dir or Path("benchmarks") / "track1" / "scannetpp" / args.scene_id
    summary = evaluate_track2(
        package_dir=args.package_dir,
        benchmark_dir=benchmark_dir,
        track1_benchmark_dir=track1_dir,
        mode=args.mode,
        output=args.output,
        agent_command=args.agent_command,
        agent_output=args.agent_output,
        sandbox_root=args.sandbox_root,
        agent_extra_paths=args.agent_extra_path or [],
        agent_include_build_code=args.agent_include_build_code,
        agent_include_source_code=args.agent_include_source_code,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
