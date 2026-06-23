#!/usr/bin/env python
"""CLI for the agent-designed memory baseline harness (skeleton).

See `.codex/agent_designed_baseline.md`. Until the designer invocation lands
(Phase 4), this runs the wired control flow and reports `skeleton_incomplete`
with a step-by-step trace, exercising workspace assembly and the leakage scan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.agent_designed.contract import VARIANTS
from spatial_memory_evaluation.agent_designed.harness import run_agent_designed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent-designed memory baseline harness.")
    parser.add_argument("--variant", choices=VARIANTS, default="coding_agent")
    parser.add_argument("--dataset", default="scannetpp")
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("sandboxes") / "agent_designed_workspace",
    )
    parser.add_argument("--train-scene-id", action="append", default=[], dest="train_scene_ids")
    parser.add_argument("--heldout-scene-id", action="append", default=[], dest="heldout_scene_ids")
    parser.add_argument(
        "--example-package",
        action="append",
        default=[],
        dest="example_packages",
        type=Path,
    )
    parser.add_argument("--llm-command", default=None)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = run_agent_designed(
        variant=args.variant,
        dataset=args.dataset,
        workspace_root=args.workspace_root,
        train_scene_ids=args.train_scene_ids,
        heldout_scene_ids=args.heldout_scene_ids,
        example_package_dirs=args.example_packages,
        llm_command=args.llm_command,
        max_rounds=args.max_rounds,
        output_dir=args.output_dir,
    )
    print(json.dumps(run.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
