#!/usr/bin/env python
"""Score a designed memory package on the DEV tests — the loop's feedback tool.

This is what you (or, later, the designer agent) run inside the sandbox to see how
a design performs. It scores a built package on the DEV-scene benchmarks using the
UNCHANGED Track 1/2/3 evaluators (so the number is on the same scale as the final
held-out report), prints per-track metrics + the mean dev score, and points at the
per-eval detail files so you can see which queries failed and revise the code.

Run inside the sandbox (reads ./sandbox_config.json for repo root + dev split):
    python score_design.py --package-dir memories/my_design/scene0527_00 --mode fixed_api

Or in-repo (auto-detects repo root):
    python scripts/agent_designed/score_design.py --package-dir <pkg> --mode fixed_api
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_repo_root() -> Path:
    """Find the repo root so the evaluators import, both in-repo and in a sandbox."""

    here = Path(__file__).resolve()
    # In-repo: scripts/agent_designed/score_design.py -> repo root is parents[2].
    candidate = here.parents[2]
    if (candidate / "spatial_memory_evaluation").is_dir():
        return candidate
    # Sandbox: read sandbox_config.json next to this script.
    cfg = here.parent / "sandbox_config.json"
    if cfg.exists():
        root = Path(json.loads(cfg.read_text())["repo_root"])
        if (root / "spatial_memory_evaluation").is_dir():
            return root
    raise SystemExit(
        "cannot locate the spatial-memory-evaluation repo; expected it at "
        f"{candidate} or via sandbox_config.json next to {here}"
    )


REPO_ROOT = _bootstrap_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.agent_designed.dev_eval import PRIMARY_METRIC, evaluate_dev
from spatial_memory_evaluation.agent_designed.splits import default_split


def _load_sandbox_config() -> dict:
    cfg = Path(__file__).resolve().parent / "sandbox_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text())
    return {}


def parse_args() -> argparse.Namespace:
    cfg = _load_sandbox_config()
    ap = argparse.ArgumentParser(description="Score a designed package on the DEV tests.")
    ap.add_argument("--package-dir", type=Path, required=True, help="A built memory package to score.")
    ap.add_argument(
        "--dev-tests-root",
        type=Path,
        default=Path(cfg["dev_tests_root"]) if cfg.get("dev_tests_root") else None,
        help="Root with track1/<scene>, track2/<scene>, track3/<scene> dirs "
        "(default: from sandbox_config.json).",
    )
    ap.add_argument(
        "--dev-scene-id",
        action="append",
        default=None,
        dest="dev_scene_ids",
        help="DEV scenes to score on (repeatable; REPLACES the config default). "
        "Default: config dev_scene_ids, else splits.py.",
    )
    ap.set_defaults(_config_dev_scene_ids=cfg.get("dev_scene_ids") or [])
    ap.add_argument("--mode", choices=("fixed_api", "tool_llm"), default="fixed_api")
    ap.add_argument("--llm-command", default=None, help="Required for --mode tool_llm.")
    ap.add_argument("--output-root", type=Path, default=None, help="Where per-eval details land.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dev_scene_ids = (
        args.dev_scene_ids
        or args._config_dev_scene_ids
        or list(default_split().dev_scene_ids)
    )
    dev_tests_root = args.dev_tests_root
    if dev_tests_root is None:
        raise SystemExit("--dev-tests-root is required (or run inside a sandbox with sandbox_config.json)")

    output_root = args.output_root or (args.package_dir.parent / "_dev_eval")
    result = evaluate_dev(
        package_dir=args.package_dir,
        dev_tests_root=dev_tests_root,
        dev_scene_ids=dev_scene_ids,
        mode=args.mode,
        llm_command=args.llm_command,
        output_root=output_root,
    )

    print("=" * 64)
    print(f"DEV SCORE: {result.dev_score if result.dev_score is not None else 'n/a'}   "
          f"(mean of per-track means over supported tracks)")
    print(f"status: {result.status}   mode: {args.mode}   scenes: {dev_scene_ids}")
    print("-" * 64)
    if result.per_track:
        for track, info in result.per_track.items():
            print(f"  {track:26s} {info['metric_key']:14s} mean={info['mean']:.3f}  (n={info['n']})")
    else:
        print("  no track scored — does your capabilities.json declare a supported "
              f"fixed-API entrypoint? Primary metrics: {PRIMARY_METRIC}")
    print("-" * 64)
    print("per (track, scene):")
    for row in result.per_eval:
        m = f"{row['metric']:.3f}" if isinstance(row.get("metric"), (int, float)) else "n/a"
        print(f"  {row['track']:26s} {row['scene']:14s} status={row['status']:10s} metric={m}")
    print(f"\ndetails (per-query) under: {output_root}")
    print("=" * 64)

    # Machine-readable sidecar next to the details.
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dev_score.json").write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
