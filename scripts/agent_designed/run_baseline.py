#!/usr/bin/env python
"""CLI for the agent-designed memory auto-research harness.

See `.codex/agent_designed_baseline.md`. Until the designer launch seam is enabled
(`agent_designed/designer.py`), this runs the wired control flow and reports
`ready_for_designer` with a full provenance trace — exercising workspace assembly,
the dev/held-out split, the leakage scan, and the journal plumbing.

Examples:
    # Assemble the workspace + run the (seam-off) loop with default dev scenes:
    python scripts/agent_designed/run_baseline.py --variant auto_research

    # Override dev scenes:
    python scripts/agent_designed/run_baseline.py \
        --dev-scene-id scene0527_00 --dev-scene-id scene0406_00
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.agent_designed.contract import DEFAULT_VARIANT, VARIANTS
from spatial_memory_evaluation.agent_designed.harness import run_agent_designed
from spatial_memory_evaluation.agent_designed.splits import default_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent-designed memory auto-research harness.")
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    parser.add_argument("--dataset", default="scannet")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("sandboxes") / "agent_designed_workspace",
    )
    parser.add_argument(
        "--dev-scene-id",
        action="append",
        default=[],
        dest="dev_scene_ids",
        help="Override DEV scenes (repeatable). Default: the splits.py dev set.",
    )
    parser.add_argument(
        "--example-package",
        action="append",
        default=[],
        dest="example_packages",
        type=Path,
        help="Example package dir to copy into the workspace (repeatable).",
    )
    parser.add_argument("--llm-command", default=None, help="Coding-agent transport (seam off by default).")
    parser.add_argument("--dev-eval-mode", choices=("fixed_api", "tool_llm"), default="fixed_api")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--no-gain-window", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = default_split(tuple(args.dev_scene_ids) if args.dev_scene_ids else None)
    run = run_agent_designed(
        variant=args.variant,
        workspace_root=args.workspace_root,
        split=split,
        dataset=args.dataset,
        example_package_dirs=args.example_packages or None,
        llm_command=args.llm_command,
        dev_eval_mode=args.dev_eval_mode,
        max_rounds=args.max_rounds,
        no_gain_window=args.no_gain_window,
        output_dir=args.output_dir,
    )
    print(json.dumps(run.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
