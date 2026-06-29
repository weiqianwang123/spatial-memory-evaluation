#!/usr/bin/env python
"""Create a self-contained EXTERNAL sandbox to drive the designer agent by hand.

Builds the agent-designed workspace OUTSIDE the repo so you can `cd` in, run
`claude` yourself, watch it design a memory, and score it — exactly the loop that
`invoke_designer` will later automate. The sandbox contains everything the agent
needs and nothing about the held-out scenes:

    <sandbox>/
      CONTRACT.md  metrics.md  shared_modules.md  README.md
      examples/                 # real example packages (copied)
      dev_scenes/<scene>/       # symlinks to DEV RGB-D layouts (+ README)
      dev_tests/                # (auto_research: you/agent author here)
      starter/                  # build_memory.py / query_*.py templates
      score_design.py           # standalone scorer (copied in)
      sandbox_config.json       # repo_root + dev_tests_root + dev split
      RUN_CLAUDE.md             # how to launch claude here + the loop
      memories/                 # where the agent writes its built package(s)

Usage:
    python scripts/agent_designed/make_sandbox.py \
        --sandbox-root /home/robin_wang/agent_designed_sandbox \
        --variant auto_research
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.agent_designed.contract import DEFAULT_VARIANT, VARIANTS
from spatial_memory_evaluation.agent_designed.splits import default_split, write_split_manifest
from spatial_memory_evaluation.agent_designed.workspace import build_workspace

DEV_BENCH = {
    "track1": REPO_ROOT / "benchmarks" / "track1" / "scannet",
    "track2": REPO_ROOT / "benchmarks" / "track2" / "scanents3d",
    "track3": REPO_ROOT / "benchmarks" / "track3" / "openeqa",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create an external designer sandbox.")
    ap.add_argument(
        "--sandbox-root",
        type=Path,
        default=None,
        help="Default: ~/agent_designed_sandbox_<variant>",
    )
    ap.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT)
    ap.add_argument("--dev-scene-id", action="append", default=[], dest="dev_scene_ids")
    ap.add_argument(
        "--run-id",
        default=None,
        help="Unique tag for this run's fresh sandbox dir "
        "(~/agent_designed_sandbox_<variant>_<run-id>). If omitted, the next free "
        "run number is used. Every launch is a NEW sandbox — runs are never resumed.",
    )
    return ap.parse_args()


def _fresh_sandbox_dir(variant: str, run_id: str | None) -> Path:
    """Always return a NEW, non-existent sandbox dir (never reuse/resume a prior run).

    Policy: each auto-design launch starts from a clean sandbox so a run can never
    inherit code/state from a previous one. If ``run_id`` is given it must not
    already exist; otherwise we pick the lowest free run number.
    """

    base = Path.home()
    if run_id:
        d = base / f"agent_designed_sandbox_{variant}_{run_id}"
        if d.exists():
            raise SystemExit(
                f"{d} already exists; runs are never resumed — choose a new --run-id "
                "or delete that dir explicitly."
            )
        return d
    n = 1
    while (base / f"agent_designed_sandbox_{variant}_run{n}").exists():
        n += 1
    return base / f"agent_designed_sandbox_{variant}_run{n}"


def _seed_into(seed_root: Path, dev_scene_ids: list[str]) -> dict:
    """Copy the prepared DEV benchmarks into ``seed_root/<track>/<scene>`` (real files).

    Used for the non-authoring variants (loop_fixed_tests / one_shot): the agent
    sees a FIXED test set rather than authoring its own. build_workspace then copies
    this seed into the sandbox's ``dev_tests/`` (children = track dirs). Copies (not
    symlinks) so the agent can read/inspect the queries directly.
    """

    status: dict[str, dict] = {}
    for track, root in DEV_BENCH.items():
        for scene in dev_scene_ids:
            src = root / scene
            if not src.exists():
                status.setdefault(scene, {})[track] = "MISSING"
                continue
            dst = seed_root / track / scene
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            status.setdefault(scene, {})[track] = "ok"
    return status


def _link_dev_tests(sandbox: Path, dev_scene_ids: list[str]) -> dict:
    """Expose the prepared DEV benchmarks so score_design.py works offline.

    For auto_research we expose them under a separate dev_tests_ref/ (so the agent's
    OWN dev_tests/ stays its authored space); the scorer reads dev_tests_ref/. For
    loop_fixed_tests build_workspace already seeded dev_tests/.
    """

    ref = sandbox / "dev_tests_ref"
    status: dict[str, dict] = {}
    for track, root in DEV_BENCH.items():
        for scene in dev_scene_ids:
            src = root / scene
            if not src.exists():
                status.setdefault(scene, {})[track] = "MISSING"
                continue
            dst = ref / track / scene
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink() if dst.is_symlink() else shutil.rmtree(dst)
            dst.symlink_to(src.resolve())
            status.setdefault(scene, {})[track] = "ok"
    return status


def main() -> int:
    args = parse_args()
    split = default_split(tuple(args.dev_scene_ids) if args.dev_scene_ids else None)
    # Policy: every auto-design launch starts from a NEW sandbox — runs are never
    # resumed from a previous one (no inherited code/state). An explicit
    # --sandbox-root must not already exist.
    if args.sandbox_root is not None:
        sandbox = args.sandbox_root
        if sandbox.exists():
            raise SystemExit(
                f"{sandbox} already exists; auto-design runs are never resumed. "
                "Pass a fresh --sandbox-root / --run-id, or delete that dir explicitly."
            )
    else:
        sandbox = _fresh_sandbox_dir(args.variant, args.run_id)
    sandbox.mkdir(parents=True)

    # Non-authoring variants get a FIXED test set seeded into dev_tests/; the
    # centerpiece auto_research variant authors its own (dev_tests/ left empty).
    authors_own = args.variant == "auto_research"

    # 1. Assemble the workspace in-place (docs, examples, dev-scene layout links).
    #    For seeded variants, build_workspace copies the seed into dev_tests/.
    seed_root = None
    if not authors_own:
        seed_root = sandbox / "_seed_dev_tests"
        _seed_into(seed_root, list(split.dev_scene_ids))
    build_workspace(
        variant=args.variant,
        workspace_root=sandbox,
        split=split,
        dataset="scannet",
        seeded_dev_tests_root=seed_root,
    )
    if seed_root and seed_root.exists():
        shutil.rmtree(seed_root)  # build_workspace already copied it into dev_tests/

    # 2. Expose the prepared DEV benchmarks for the scorer (always, via dev_tests_ref).
    dev_tests_status = _link_dev_tests(sandbox, list(split.dev_scene_ids))

    # 3. Copy the standalone scorer + write its config.
    shutil.copy2(REPO_ROOT / "scripts" / "agent_designed" / "score_design.py", sandbox / "score_design.py")
    (sandbox / "sandbox_config.json").write_text(
        json.dumps(
            {
                "repo_root": str(REPO_ROOT),
                "dev_tests_root": str(sandbox / "dev_tests_ref"),
                "dev_scene_ids": list(split.dev_scene_ids),
                "variant": args.variant,
                "python": sys.executable,
            },
            indent=2,
        )
    )
    (sandbox / "memories").mkdir(exist_ok=True)
    write_split_manifest(sandbox / "splits.json", split)
    _write_run_claude_md(sandbox / "RUN_CLAUDE.md", args.variant, split, sys.executable, authors_own)

    # 4. Report.
    seeded = sorted(
        str(p.relative_to(sandbox)) for p in (sandbox / "dev_tests").glob("*/*")
    ) if not authors_own else []
    print(json.dumps({
        "sandbox": str(sandbox),
        "variant": args.variant,
        "authors_own_dev_tests": authors_own,
        "dev_scene_ids": list(split.dev_scene_ids),
        "dev_tests_seeded": seeded,
        "dev_tests_ref_exposed": dev_tests_status,
        "next": f"cd {sandbox} && cat RUN_CLAUDE.md",
    }, indent=2))
    return 0


_VARIANT_POLICY = {
    "one_shot": (
        "ONE-SHOT (no iteration). The agent designs and builds ONCE, then you score.\n"
        "No revise-and-rebuild loop. Dev tests are PRE-SEEDED under dev_tests/ (the\n"
        "agent may read them to understand the task but must not edit them). This\n"
        "measures: can a coding agent design usable memory in a single pass?"
    ),
    "loop_fixed_tests": (
        "LOOP with a FIXED test set. The agent iterates (build -> score -> revise ->\n"
        "rebuild) but the dev tests are PRE-SEEDED under dev_tests/ and frozen (the\n"
        "agent must not add/edit tests). This isolates: does iteration help when the\n"
        "evaluation is held constant?"
    ),
    "auto_research": (
        "AUTO-RESEARCH (centerpiece). The agent iterates AND authors its OWN dev\n"
        "tests under dev_tests/ from the DEV scenes (dev_tests/ starts empty;\n"
        "dev_tests_ref/ holds the harness's metric-faithful reference for scoring).\n"
        "This is the full self-improvement loop."
    ),
}


def _write_run_claude_md(path: Path, variant: str, split, python: str, authors_own: bool) -> None:
    tests_note = (
        "dev_tests/ starts EMPTY — you author your own self-tests here (see\n"
        "dev_scenes/README.md). The scorer reads the harness reference in\n"
        "dev_tests_ref/, so you get a metric-faithful signal regardless."
        if authors_own
        else
        "dev_tests/ is PRE-SEEDED with a fixed test set (do not edit it). The scorer\n"
        "uses dev_tests_ref/ (identical content) for the dev score."
    )
    path.write_text(
        "# Driving the designer agent by hand\n\n"
        f"## Variant: {variant}\n\n"
        f"{_VARIANT_POLICY[variant]}\n\n"
        "This sandbox is OUTSIDE the eval repo. Launch a coding agent here, let it\n"
        "design a semantic-map memory, score it, and (if the variant loops) iterate.\n\n"
        "## 0. Read the brief\n\n"
        "    cat CONTRACT.md metrics.md shared_modules.md\n"
        "    cat dev_scenes/README.md\n\n"
        "## 1. Launch claude as a coding agent (cwd = this sandbox)\n\n"
        "    claude --permission-mode bypassPermissions --add-dir .\n\n"
        "Then prompt it, e.g.:\n\n"
        "    Read CONTRACT.md, metrics.md, shared_modules.md, and the examples/.\n"
        "    Implement starter/build_memory.py and the per-track entrypoints to\n"
        "    build a semantic-map memory from the DEV scene layouts in dev_scenes/.\n"
        "    Write a validated package under memories/<name>/<scene>/. Then run\n"
        "    `python score_design.py --package-dir memories/<name>/<scene>`"
        + ("" if variant == "one_shot" else " and\n    improve your code based on the per-track dev scores")
        + ".\n\n"
        "(Headless equivalent: `claude -p \"<prompt>\" --permission-mode "
        "bypassPermissions --add-dir . --max-turns 60 --output-format text`.)\n\n"
        f"Dev-tests policy: {tests_note}\n\n"
        "## 2. Score a built package (the feedback signal)\n\n"
        f"    {python} score_design.py --package-dir memories/<name>/<scene> --mode fixed_api\n\n"
        "Prints per-track metrics (T1 success@5, T2 acc@0.5m, T3 llm_match) + the\n"
        "mean dev score, and writes _dev_eval/dev_score.json.\n\n"
        "## What to watch for (notes for automating invoke_designer later)\n\n"
        "- Does the agent find everything it needs in CONTRACT/shared_modules?\n"
        "- Does it use the shared perception stack + local qwen (not Claude) at\n"
        "  build/query time?\n"
        "- How many rounds / how much wall-clock to a non-trivial dev score?\n"
        "- Does the package pass the validator on the first try? What trips it up?\n"
        "- Does it ever try to read held-out scenes? (It must not; none are here.)\n"
        + ("- (auto_research) Are the dev tests it authors metric-faithful, or does\n"
           "  it game them? Compare its dev_tests/ against dev_tests_ref/.\n" if authors_own else "")
        + f"\nDEV scenes: {', '.join(split.dev_scene_ids)}\n"
        "Held-out scenes are NOT present in this sandbox by design.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
