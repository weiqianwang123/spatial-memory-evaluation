#!/usr/bin/env python
"""Build Track 3 OpenEQA QA benchmark data (ScanNet + HM3D).

Splits the OpenEQA v0 question file into per-dataset benchmark dirs with a
questions.jsonl (no answers) and an evaluator-only answers.jsonl. The default
question file is the OpenEQA repo copy on this machine; override with
`--openeqa-questions`. See `.codex/path_registry.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.track3.data import DEFAULT_OPENEQA_QUESTIONS, build_track3_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 3 OpenEQA benchmark data.")
    parser.add_argument("--openeqa-questions", type=Path, default=DEFAULT_OPENEQA_QUESTIONS)
    parser.add_argument("--dataset", choices=("scannet", "hm3d"), default=None, help="Default: build both.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: benchmarks/track3/openeqa",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root or Path("benchmarks") / "track3" / "openeqa"
    if not Path(args.openeqa_questions).exists():
        print(
            json.dumps(
                {
                    "status": "data_unavailable",
                    "reason": f"OpenEQA question file not found: {args.openeqa_questions}",
                    "output_root": str(output_root),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = build_track3_data(
        openeqa_questions=args.openeqa_questions,
        output_root=output_root,
        dataset=args.dataset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
