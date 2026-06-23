"""Orchestrate one agent-designed memory baseline run.

Control flow (see ``.codex/agent_designed_baseline.md`` §7):

    1. build_workspace(variant, dataset, train split)
    2. invoke_designer(workspace) -> design   [STUB until Phase 4]
    3. run design.build_memory per scene -> package; validate
    4. scan_for_leakage(design, heldout)      [STUB until Phase 4]
    5. evaluate package on Track 1/2/3
    6. iterative variant: summarize training failures, loop
    7. final scoring on held-out split
    8. write run report

Steps 3/5/7 reuse the unchanged evaluators and validator. Steps 2/4/6 are the
agent-designed-specific stubs. The harness is wired end-to-end so that, once the
stubs are implemented, no plumbing changes are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import write_json
from spatial_memory_evaluation.memory_package_validator import validate_package
from spatial_memory_evaluation.output_paths import timestamped_result_dir

from .contract import VARIANTS
from .designer import DesignResult, invoke_designer
from .leakage import scan_for_leakage
from .workspace import build_workspace


@dataclass
class AgentDesignedRun:
    variant: str
    dataset: str
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "dataset": self.dataset,
            "status": self.status,
            "steps": self.steps,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


def run_agent_designed(
    *,
    variant: str,
    dataset: str,
    workspace_root: Path,
    train_scene_ids: list[str],
    heldout_scene_ids: list[str],
    example_package_dirs: list[Path] | None = None,
    llm_command: str | None = None,
    max_rounds: int = 1,
    output_dir: Path | None = None,
) -> AgentDesignedRun:
    """Run the agent-designed baseline harness for one variant.

    This skeleton wires every step and records a structured run report. Because the
    designer and the per-scene build are Phase-4 stubs, the run completes with
    ``status="skeleton_incomplete"`` and a step-by-step trace, rather than
    fabricating scores.
    """

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if output_dir is None:
        output_dir = timestamped_result_dir("agent_designed", f"agent_designed-{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run = AgentDesignedRun(variant=variant, dataset=dataset, status="running", output_dir=output_dir)

    # Step 1: workspace.
    workspace = build_workspace(
        variant=variant,
        dataset=dataset,
        workspace_root=workspace_root,
        train_scene_ids=train_scene_ids,
        example_package_dirs=example_package_dirs,
    )
    run.steps.append({"step": "build_workspace", "status": "ok", "root": str(workspace.root)})

    # Step 2: designer (STUB).
    design_dir = output_dir / "design"
    design: DesignResult = invoke_designer(
        workspace=workspace,
        design_dir=design_dir,
        feedback=None,
        llm_command=llm_command,
    )
    run.steps.append({"step": "invoke_designer", "status": design.status, "message": design.message})

    # Step 4: leakage scan over whatever the designer wrote (STUB; safe on empty dir).
    leakage = scan_for_leakage(design_dir=design_dir, heldout_scene_ids=heldout_scene_ids)
    run.steps.append({"step": "scan_for_leakage", **leakage.to_json()})

    if design.status != "ok":
        # Steps 3/5/6/7 need a real design. Stop cleanly and record why.
        run.status = "skeleton_incomplete"
        run.steps.append(
            {
                "step": "build_and_evaluate",
                "status": "skipped",
                "reason": (
                    "designer is a Phase-4 stub; no build_memory.py to run. "
                    "Once invoke_designer returns status='ok', this step builds a "
                    "package per scene, validates it, and calls evaluate_track1/2/3."
                ),
            }
        )
        write_json(output_dir / "run_summary.json", run.to_json())
        write_json(output_dir / "design_manifest.json", workspace.provided_manifest)
        return run

    # Step 3/5/7 (reached only once the designer stub is implemented).
    run.status = "ok"
    run.steps.append(
        {
            "step": "build_and_evaluate",
            "status": "ok",
            "note": "build per scene -> validate_package -> evaluate_track1/2/3 on held-out split",
        }
    )
    write_json(output_dir / "run_summary.json", run.to_json())
    write_json(output_dir / "design_manifest.json", workspace.provided_manifest)
    return run


def validate_designed_package(package_dir: Path) -> dict[str, Any]:
    """Thin wrapper so the harness/CLI can validate a designed package."""

    return validate_package(package_dir).to_json()
