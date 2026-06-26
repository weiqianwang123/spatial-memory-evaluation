"""Orchestrate one agent-designed memory auto-research run.

Control flow (see ``.codex/agent_designed_baseline.md`` §8):

    1. build_workspace(variant, dev split)
    2. for round r in range(max_rounds):
         a. invoke_designer(workspace, feedback)        [SEAM: off until enabled]
         b. scan_for_leakage(design, held-out ids)
         c. run design.build_memory per DEV scene -> package; validate
         d. evaluate_dev(package, the agent's own dev tests)  -> dev_score
         e. journal.append(round); keep (snapshot) or discard (revert) vs best
         f. stop on no-gain convergence or budget
    3. FREEZE best; (separately) evaluate_on_heldout(best, the 10 scenes)
    4. write run report

Steps (c)/(d)/(3) reuse the unchanged validator + Track 1/2/3 evaluators. Step (a)
is the single un-fired seam. While it is off, the harness completes after step 1
with ``status="ready_for_designer"`` and a full provenance trace — proving the
workspace, split, journal, leakage scan, and dev-eval plumbing are all in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from spatial_memory_evaluation.common.jsonl import write_json
from spatial_memory_evaluation.memory_package_validator import validate_package
from spatial_memory_evaluation.output_paths import timestamped_result_dir

from .contract import VARIANTS
from .designer import DesignResult, invoke_designer
from .dev_eval import evaluate_dev
from .journal import Journal, RoundRecord
from .leakage import scan_for_leakage
from .splits import Split, default_split, write_split_manifest
from .workspace import build_workspace


@dataclass
class AgentDesignedRun:
    variant: str
    dataset: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None
    best_dev_score: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "dataset": self.dataset,
            "status": self.status,
            "best_dev_score": self.best_dev_score,
            "steps": self.steps,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


def run_agent_designed(
    *,
    variant: str,
    workspace_root: Path,
    split: Split | None = None,
    dataset: str = "scannet",
    example_package_dirs: list[Path] | None = None,
    llm_command: str | None = None,
    dev_eval_mode: str = "fixed_api",
    dev_judge: Callable[[str, str, str], float] | None = None,
    max_rounds: int = 5,
    no_gain_window: int = 2,
    output_dir: Path | None = None,
) -> AgentDesignedRun:
    """Run the agent-designed auto-research harness for one variant.

    Wires every step and records a structured run report + journal. Because the
    designer launch (``invoke_designer``) is the single un-fired seam, the run
    completes after workspace assembly with ``status="ready_for_designer"`` rather
    than fabricating a design/scores. Once the seam is enabled, the loop in this
    function runs unchanged.
    """

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    split = split or default_split()
    split.assert_disjoint()
    if output_dir is None:
        output_dir = timestamped_result_dir("agent_designed", f"agent_designed-{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run = AgentDesignedRun(variant=variant, dataset=dataset, status="running", output_dir=output_dir)

    # Step 1: workspace + split manifest.
    workspace = build_workspace(
        variant=variant,
        workspace_root=workspace_root,
        split=split,
        dataset=dataset,
        example_package_dirs=example_package_dirs,
    )
    write_split_manifest(output_dir / "splits.json", split)
    run.steps.append(
        {
            "step": "build_workspace",
            "status": "ok",
            "root": str(workspace.root),
            "dev_scenes": list(split.dev_scene_ids),
            "dev_scenes_linked": workspace.provided_manifest.get("dev_scenes_linked"),
            "examples_copied": workspace.provided_manifest.get("examples_copied"),
        }
    )

    journal = Journal(journal_path=output_dir / "journal.jsonl", best_dir=output_dir / "best")
    design_dir = output_dir / "design"

    for r in range(max_rounds):
        # Step 2a: designer (SEAM — off until enabled).
        design: DesignResult = invoke_designer(
            workspace=workspace,
            design_dir=design_dir,
            feedback=_round_feedback(journal, output_dir, r),
            llm_command=llm_command,
        )

        if design.status != "ok":
            # Seam off (or designer failed): record readiness and stop cleanly.
            leakage = scan_for_leakage(
                design_dir=design_dir,
                heldout_scene_ids=list(split.heldout_scene_ids),
                extra_dirs=[workspace.root / "dev_tests"],
            )
            run.steps.append(
                {
                    "step": "round",
                    "round_index": r,
                    "designer_status": design.status,
                    "designer_message": design.message,
                    "leakage": leakage.to_json(),
                    "build_and_eval": "skipped (no design to build)",
                }
            )
            run.status = "ready_for_designer" if design.status == "skipped" else "designer_error"
            break

        # Step 2b: leakage scan over the design + any authored dev tests.
        leakage = scan_for_leakage(
            design_dir=design_dir,
            heldout_scene_ids=list(split.heldout_scene_ids),
            extra_dirs=[workspace.root / "dev_tests"],
        )

        # Step 2c: build per DEV scene -> package; validate.
        build_records, package_dir = _build_dev_packages(design, workspace, split, output_dir, r)

        # Step 2d: self-eval on the agent's own dev tests.
        dev = evaluate_dev(
            package_dir=package_dir,
            dev_tests_root=workspace.root / "dev_tests",
            dev_scene_ids=list(split.dev_scene_ids),
            mode=dev_eval_mode,
            llm_command=llm_command,
            judge=dev_judge,
            output_root=output_dir / f"round{r}" / "dev_eval",
        )

        # Step 2e: keep/discard vs best.
        improved = leakage.clean and journal.improves(dev.dev_score)
        decision = "keep" if improved else "discard"
        record = RoundRecord(
            round_index=r,
            design_summary=design.message,
            dev_score=dev.dev_score,
            dev_metrics=dev.to_json(),
            decision=decision,
            rationale=("dev score improved over best" if improved else "no improvement / leakage"),
            leakage_clean=leakage.clean,
            cost=design.cost or {},
        )
        if decision == "keep":
            journal.snapshot_best(design_dir)
        else:
            journal.revert_to_best(design_dir)
        journal.append(record)
        run.best_dev_score = journal.best_score
        run.steps.append(
            {
                "step": "round",
                "round_index": r,
                "designer_status": "ok",
                "leakage": leakage.to_json(),
                "build": build_records,
                "dev_eval": dev.to_json(),
                "decision": decision,
            }
        )

        # Step 2f: stop on no-gain convergence.
        if journal.converged(no_gain_window):
            run.status = "converged"
            break
    else:
        run.status = "max_rounds_reached"

    write_json(output_dir / "run_summary.json", run.to_json())
    write_json(output_dir / "design_manifest.json", workspace.provided_manifest)
    return run


def evaluate_on_heldout(
    *,
    best_design_dir: Path,
    heldout_scene_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Score the FROZEN best design on the held-out 10 scenes (the reported number).

    Separate from the loop on purpose: held-out scoring happens ONCE, after freeze,
    and never feeds back into the design. This builds the design's memory on each
    held-out scene and runs the Track 1/2/3 evaluators against the real benchmarks
    under ``benchmarks/track{1,2,3}/.../<scene>``. Returns the per-scene/track
    summary. (Requires the designer seam to have produced a real best design.)
    """

    raise NotImplementedError(
        "evaluate_on_heldout runs only after a real frozen design exists (designer "
        "seam enabled). It builds the best design on each held-out scene and calls "
        "evaluate_track1/2/3 against the real benchmarks. Wire alongside the seam."
    )


def validate_designed_package(package_dir: Path) -> dict[str, Any]:
    """Thin wrapper so the harness/CLI can validate a designed package."""

    return validate_package(package_dir).to_json()


# --- internal helpers (reached only once the designer seam is enabled) ---


def _round_feedback(journal: Journal, output_dir: Path, round_index: int) -> dict[str, Any] | None:
    """Assemble the previous round's dev report + journal tail as designer feedback.

    Returns ``None`` on round 0. NEVER includes held-out data — only DEV-scene
    dev-eval outputs and the journal.
    """

    if round_index == 0:
        return None
    return {
        "round_index": round_index,
        "journal_path": str(journal.journal_path),
        "best_dev_score": journal.best_score,
        "prev_dev_eval_dir": str(output_dir / f"round{round_index - 1}" / "dev_eval"),
    }


def _build_dev_packages(
    design: DesignResult,
    workspace: "Any",
    split: Split,
    output_dir: Path,
    round_index: int,
) -> tuple[list[dict[str, Any]], Path]:
    """Build the design's memory on each DEV scene and validate the package(s).

    Reached only when the designer seam is enabled. Calls the agent-written
    ``build_memory`` per DEV scene, then ``validate_package`` on the result.
    Returns (per-scene build records, the package dir for dev-eval).
    """

    raise NotImplementedError(
        "build-per-DEV-scene runs only with a real design.build_memory (designer "
        "seam enabled). It imports design_dir/build_memory.py, builds a package per "
        "DEV scene under memories/agent_designed/<dataset>/<scene>/, validates each, "
        "and returns the package dir(s) for evaluate_dev."
    )
