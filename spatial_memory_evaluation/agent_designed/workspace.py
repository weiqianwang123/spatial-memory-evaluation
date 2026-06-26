"""Assemble the sandbox workspace given to a designer agent.

The workspace gives the designer everything it needs to design + self-improve a
semantic-map memory on the DEV scenes, and NOTHING that would leak the held-out
evaluation. It is fully materialized from live data (no placeholders):

    <workspace>/
      CONTRACT.md           # what to build, the rules, the loop
      metrics.md            # Track 1/2/3 metric definitions (the dev score)
      shared_modules.md     # resolved perception stack + ollama call docs
      examples/             # real example memory packages (copied)
      dev_scenes/           # symlinks to DEV RGB-D layouts + GT-derivation tooling
      dev_tests/            # agent authors self-tests here (seeded in loop_fixed_tests)
      starter/              # build_memory.py / query_*.py templates to fill in
      journal.jsonl         # append-only experiment journal (created empty)
      provided_manifest.json
      README.md

Anti-leakage is structural: only DEV scene ids are ever resolved/linked; the
held-out scene ids are written into CONTRACT.md as an explicit blocklist and never
materialized. ``build_workspace`` asserts the dev/held-out split is disjoint.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import write_json

from .contract import (
    FORBIDDEN_WORKSPACE_INPUTS,
    PROVIDED_WORKSPACE_INPUTS,
    WorkspaceContract,
)
from .shared_modules_doc import render_shared_modules_md
from .splits import Split, default_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LAYOUT_ROOT = REPO_ROOT / "data" / "scannet_layouts"
DEFAULT_SCANS_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannet/scans")


@dataclass(frozen=True)
class Workspace:
    root: Path
    contract: WorkspaceContract
    provided_manifest: dict[str, Any]
    split: Split


def build_workspace(
    *,
    variant: str,
    workspace_root: Path,
    split: Split | None = None,
    dataset: str = "scannet",
    example_package_dirs: list[Path] | None = None,
    layout_root: Path = DEFAULT_LAYOUT_ROOT,
    scans_root: Path = DEFAULT_SCANS_ROOT,
    seeded_dev_tests_root: Path | None = None,
) -> Workspace:
    """Create a designer workspace directory and return its handle.

    Materializes CONTRACT/metrics/shared_modules docs, copies example packages,
    symlinks DEV-scene layouts + the GT-derivation tooling, writes starter
    templates, and creates an empty journal. For ``loop_fixed_tests`` the harness
    may pass ``seeded_dev_tests_root`` to pre-populate ``dev_tests/`` so the agent
    does NOT author its own (the ablation).
    """

    split = split or default_split()
    split.assert_disjoint()
    contract = WorkspaceContract(
        variant=variant,
        dataset=dataset,
        authors_own_dev_tests=(variant == "auto_research"),
    )
    contract.validate()

    workspace_root.mkdir(parents=True, exist_ok=True)
    for sub in ("examples", "dev_scenes", "dev_tests", "starter"):
        (workspace_root / sub).mkdir(exist_ok=True)

    _write_contract_md(workspace_root / "CONTRACT.md", contract, split)
    _write_metrics_md(workspace_root / "metrics.md")
    (workspace_root / "shared_modules.md").write_text(render_shared_modules_md(), encoding="utf-8")
    _write_starter_templates(workspace_root / "starter", contract)
    _write_readme(workspace_root / "README.md", contract, split)

    # Example packages (copied, not linked, so the agent can read freely).
    example_package_dirs = example_package_dirs or _default_example_packages()
    examples_copied = _copy_examples(workspace_root / "examples", example_package_dirs)

    # DEV-scene RGB-D layouts (symlinked) + GT-derivation tooling.
    dev_scene_status = _link_dev_scenes(
        workspace_root / "dev_scenes", split.dev_scene_ids, layout_root, scans_root
    )

    # Seeded dev tests (loop_fixed_tests only).
    seeded = False
    if seeded_dev_tests_root and seeded_dev_tests_root.exists():
        _copy_tree_into(seeded_dev_tests_root, workspace_root / "dev_tests")
        seeded = True

    # Empty journal so the loop can append from round 0.
    journal_path = workspace_root / "journal.jsonl"
    if not journal_path.exists():
        journal_path.write_text("", encoding="utf-8")

    provided = {
        "variant": variant,
        "dataset": dataset,
        "authors_own_dev_tests": contract.authors_own_dev_tests,
        "provided_inputs": list(PROVIDED_WORKSPACE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_WORKSPACE_INPUTS),
        "dev_scene_ids": list(split.dev_scene_ids),
        "heldout_scene_ids_blocked": list(split.heldout_scene_ids),
        "example_package_dirs": [str(p) for p in example_package_dirs],
        "examples_copied": examples_copied,
        "dev_scenes_linked": dev_scene_status,
        "dev_tests_seeded": seeded,
        "notes": (
            "Workspace for the auto-research self-improvement loop. Only DEV scenes "
            "are materialized; the held-out 10 scenes are blocked and never present. "
            "No answers / GT for held-out scenes are ever placed here."
        ),
    }
    write_json(workspace_root / "provided_manifest.json", provided)

    return Workspace(root=workspace_root, contract=contract, provided_manifest=provided, split=split)


def _default_example_packages() -> list[Path]:
    examples = REPO_ROOT / "examples"
    candidates = [
        examples / "minimal_memory_package",
        examples / "caption_control_package",
    ]
    return [p for p in candidates if p.exists()]


def _copy_examples(dest_root: Path, example_dirs: list[Path]) -> bool:
    copied_any = False
    for src in example_dirs:
        if not src.exists():
            continue
        dest = dest_root / src.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        copied_any = True
    return copied_any


def _copy_tree_into(src: Path, dest: Path) -> None:
    for child in src.iterdir():
        target = dest / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(child, target)


def _link_dev_scenes(
    dest_root: Path, dev_scene_ids: tuple[str, ...], layout_root: Path, scans_root: Path
) -> dict[str, Any]:
    """Symlink each DEV scene's RGB-D layout + record GT-file presence.

    Records, per scene, whether the layout and the 4 GT files exist so a run
    cannot silently claim data it does not have.
    """

    status: dict[str, Any] = {}
    for scene in dev_scene_ids:
        layout = layout_root / scene / "layout"
        link = dest_root / scene
        layout_linked = False
        if layout.exists():
            if link.is_symlink() or link.exists():
                link.unlink() if link.is_symlink() else shutil.rmtree(link)
            link.symlink_to(layout.resolve())
            layout_linked = True
        gt = scans_root / scene
        gt_files = {
            "txt": (gt / f"{scene}.txt").exists(),
            "aggregation": (gt / f"{scene}.aggregation.json").exists(),
            "segs": (gt / f"{scene}_vh_clean_2.0.010000.segs.json").exists(),
            "ply": (gt / f"{scene}_vh_clean_2.ply").exists(),
        }
        status[scene] = {
            "layout_linked": layout_linked,
            "layout_path": str(layout),
            "gt_files_present": gt_files,
            "gt_complete": all(gt_files.values()),
        }
    return status


def _write_contract_md(path: Path, contract: WorkspaceContract, split: Split) -> None:
    lines = [
        "# Agent-Designed Memory: Build & Self-Improvement Contract",
        "",
        f"Variant: `{contract.variant}`  Dataset: `{contract.dataset}`",
        "",
        "You are designing a **semantic-map spatial memory** for embodied agents, and",
        "improving it in an auto-research loop. Build the memory from posed RGB-D,",
        "expose query/tool interfaces, then **evaluate yourself on your own dev test",
        "cases** and revise your code based on the failures. Your frozen package is",
        "scored by the benchmark's Track 1/2/3 evaluators, unchanged.",
        "",
        "## The loop (each round)",
        "",
        "1. Edit `build_memory.py` / the query entrypoints / your memory schema.",
        "2. Build the memory on each DEV scene -> a memory package.",
        "3. Score it on your own dev tests (Track 1/2/3 evaluators).",
        "4. Read the failures + evidence + build logs; note what to change.",
        "5. Keep the change if the dev score improved, else it is reverted.",
        "Repeat until the budget or no-gain convergence. Then your best design is",
        "FROZEN and scored once on held-out scenes you never see.",
        "",
        "## You must output",
        "",
    ]
    lines += [f"- {item}" for item in contract.must_output]
    lines += ["", "## Fixed-API entrypoint names (per track)", ""]
    lines += [f"- `{track}` -> `{fn}(package_dir, query) -> dict`" for track, fn in contract.entrypoint_names.items()]
    lines += ["", "## Constraints", ""]
    lines += [f"- {item}" for item in contract.constraints]
    lines += [
        "",
        "## HELD-OUT scene blocklist (NEVER read, build, evaluate, or branch on these)",
        "",
        "These scene ids are the reported benchmark. They are not present in this",
        "workspace and must never appear in your code, tests, or builds:",
        "",
    ]
    lines += [f"- `{sid}`" for sid in split.heldout_scene_ids]
    lines += [
        "",
        "## DEV scenes (your playground)",
        "",
        "Built/evaluated freely. RGB-D layouts are under `dev_scenes/<scene>/`,",
        "GT-derivation tooling is documented in `dev_scenes/README.md`:",
        "",
    ]
    lines += [f"- `{sid}`" for sid in split.dev_scene_ids]
    lines += [
        "",
        "## Package contract",
        "",
        "Your output must pass the memory-package validator. See `examples/` for two",
        "real packages and `metrics.md` for what is scored. Use `method.family =",
        '"agent_designed"` in your manifest.',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics_md(path: Path) -> None:
    path.write_text(
        "# Metrics (what the evaluators score = your dev score)\n\n"
        "Your dev score is the mean over supported tracks of the per-track PRIMARY\n"
        "metric (the loop maximizes this). Primary metrics:\n\n"
        "- Track 1 (`track1_object_location`): **success@5**. Also reported:\n"
        "  success@1, recall@1/@5, mrr, mean_first_hit_distance_m,\n"
        "  proximity@{1,3,5}m, proximity_top1@{1,3,5}m, latency.\n"
        "- Track 2 (`track2_scanrefer`): **acc@0.5m** (top-1 center within 0.5m of\n"
        "  GT). Also: acc@0.25m, acc_top5@{0.25,0.5}m, proximity@{1,3,5}m,\n"
        "  proximity_top5@{1,3,5}m, mean_center_distance_m. Distance-only (no name\n"
        "  match required).\n"
        "- Track 3 (`track3_openeqa`): **llm_match** (LLM-Match judge in [0,1]).\n"
        "  Also: answered_rate, per-category, evidence presence, latency.\n\n"
        "Design for these metrics, but you may NOT hardcode answers to specific\n"
        "queries or condition on scene ids. You never see held-out data; your dev\n"
        "tests are generated from the DEV scenes with the provided GT tooling.\n",
        encoding="utf-8",
    )


def _write_starter_templates(starter_dir: Path, contract: WorkspaceContract) -> None:
    (starter_dir / "build_memory.py").write_text(
        '"""Build a memory package from posed RGB-D for one scene.\n\n'
        "Fill this in. Read frames from a DEV layout (rgb/ depth/ pose/ +\n"
        "camera_info.json), build your memory representation, and write a VALIDATED\n"
        "package under memories/agent_designed/<dataset>/<scene>/<run-id>/.\n"
        "Use only the shared perception modules (shared_modules.md) and the LOCAL\n"
        "qwen stack for any describer/captioner/embedder.\n"
        '"""\n\n\n'
        "def build_memory(layout_dir, output_package_dir, scene_id):\n"
        '    raise NotImplementedError("designer must implement build_memory")\n',
        encoding="utf-8",
    )
    for track, fn in contract.entrypoint_names.items():
        (starter_dir / f"{fn}.py").write_text(
            f'"""Fixed-API entrypoint for {track}: {fn}(package_dir, query) -> dict."""\n\n\n'
            f"def {fn}(package_dir, query):\n"
            f'    raise NotImplementedError("designer must implement {fn}")\n',
            encoding="utf-8",
        )


def _write_readme(path: Path, contract: WorkspaceContract, split: Split) -> None:
    authoring = (
        "Author your own dev test cases under `dev_tests/` (this is the centerpiece:\n"
        "you optimize against your own tests). Use the GT tooling in\n"
        "`dev_scenes/README.md` to generate metric-faithful queries+GT from the DEV\n"
        "scenes only.\n"
        if contract.authors_own_dev_tests
        else
        "Dev test cases are pre-seeded under `dev_tests/` (this ablation holds the\n"
        "evaluation fixed). Do NOT add or edit dev tests; only edit your code.\n"
    )
    path.write_text(
        "# Designer Workspace (auto-research)\n\n"
        "Read `CONTRACT.md` first, then `metrics.md` and `shared_modules.md`.\n\n"
        "Implement `starter/build_memory.py` and the per-track entrypoints, build a\n"
        "validated package, then score it on your dev tests and iterate.\n\n"
        f"{authoring}\n"
        "Record each round's change + rationale; the harness keeps the journal in\n"
        "`journal.jsonl`. The harness builds, validates, and scores your design on\n"
        f"the Track 1/2/3 evaluators for dataset `{contract.dataset}`.\n\n"
        f"DEV scenes: {', '.join(split.dev_scene_ids)}\n",
        encoding="utf-8",
    )
