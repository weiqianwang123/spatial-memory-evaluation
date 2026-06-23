"""Assemble the sandbox workspace given to a designer agent.

The workspace gives the designer everything it needs to design a memory and
nothing that would leak the held-out evaluation. The skeleton assembles the
directory layout, writes the contract/metrics docs, links training inputs, and
records a manifest of exactly what was provided so anti-leakage is auditable.

What is NOT yet implemented (Phase 4): copying real training RGB-D scene links,
selecting worked examples per variant, and resolving the shared-module call docs
from the live registry. Those spots are marked with TODO and the manifest records
them as ``provided: false`` so a run cannot silently claim more than it gave.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import write_json

from .contract import (
    FORBIDDEN_WORKSPACE_INPUTS,
    PROVIDED_WORKSPACE_INPUTS,
    WorkspaceContract,
)


@dataclass(frozen=True)
class Workspace:
    root: Path
    contract: WorkspaceContract
    provided_manifest: dict[str, Any]


def build_workspace(
    *,
    variant: str,
    dataset: str,
    workspace_root: Path,
    train_scene_ids: list[str],
    example_package_dirs: list[Path] | None = None,
) -> Workspace:
    """Create a designer workspace directory and return its handle.

    ``train_scene_ids`` and ``example_package_dirs`` describe what the harness
    intends to expose; the skeleton records them in the provided-inputs manifest
    and lays out the directory, but linking real RGB-D scene data is a Phase-4
    TODO (flagged in the manifest).
    """

    contract = WorkspaceContract(variant=variant, dataset=dataset)
    contract.validate()

    workspace_root.mkdir(parents=True, exist_ok=True)
    for sub in ("examples", "train", "starter"):
        (workspace_root / sub).mkdir(exist_ok=True)

    _write_contract_md(workspace_root / "CONTRACT.md", contract)
    _write_metrics_md(workspace_root / "metrics.md")
    _write_shared_modules_md(workspace_root / "shared_modules.md")
    _write_starter_templates(workspace_root / "starter", contract)
    _write_readme(workspace_root / "README.md", contract)

    example_package_dirs = example_package_dirs or []
    provided = {
        "variant": variant,
        "dataset": dataset,
        "provided_inputs": list(PROVIDED_WORKSPACE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_WORKSPACE_INPUTS),
        "train_scene_ids": train_scene_ids,
        "example_package_dirs": [str(p) for p in example_package_dirs],
        # Phase-4 TODOs: these inputs are declared but not yet materialized.
        "train_scenes_linked": False,
        "examples_copied": False,
        "shared_module_calls_resolved": False,
        "notes": (
            "Skeleton workspace. Training RGB-D links, worked examples, and live "
            "shared-module call docs are Phase-4 TODOs. No answers / GT / held-out "
            "queries are ever placed here."
        ),
    }
    write_json(workspace_root / "provided_manifest.json", provided)

    return Workspace(root=workspace_root, contract=contract, provided_manifest=provided)


def _write_contract_md(path: Path, contract: WorkspaceContract) -> None:
    lines = [
        "# Agent-Designed Memory: Build Contract",
        "",
        f"Variant: `{contract.variant}`  Dataset: `{contract.dataset}`",
        "",
        "You are designing a spatial memory for embodied agents. Build the memory",
        "from posed RGB-D, then expose query/tool interfaces. Your package is scored",
        "by the benchmark's Track 1/2/3 evaluators, unchanged.",
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
        "## Package contract",
        "",
        "Your output must pass the memory-package validator. See the example",
        "packages under `examples/` and the spec excerpt the harness provides.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics_md(path: Path) -> None:
    path.write_text(
        "# Metrics (what the evaluators score)\n\n"
        "- Track 1 (`track1_object_location`): object-location query success@1/@5,\n"
        "  recall@1/@5, MRR, first-hit distance, query latency; plus build cost\n"
        "  (native memory size, time per frame, peak RAM/VRAM).\n"
        "- Track 2 (`track2_scanrefer`): referring acc@0.25 / acc@0.5 IoU, center\n"
        "  distance, query latency.\n"
        "- Track 3 (`track3_openeqa`): LLM-Match, per-category accuracy, evidence\n"
        "  presence, answered rate, latency.\n\n"
        "Design for these metrics, but you may NOT hardcode answers to specific\n"
        "queries or condition on held-out scene ids. You never see held-out data.\n",
        encoding="utf-8",
    )


def _write_shared_modules_md(path: Path) -> None:
    # TODO(Phase 4): resolve from spatial_memory_evaluation.shared_modules.registry
    # and render the concrete callable API + checkpoints for the active profile.
    path.write_text(
        "# Shared Modules (TODO: resolve from the live registry)\n\n"
        "The harness will document the shared detector / segmenter / CLIP / VLM /\n"
        "embeddings here, with how to call each and which checkpoints are used, so\n"
        "your memory is built on the same perception stack as hand-built methods.\n"
        "Until Phase 4 this is a placeholder; see\n"
        "`spatial_memory_evaluation/shared_modules/registry.py`.\n",
        encoding="utf-8",
    )


def _write_starter_templates(starter_dir: Path, contract: WorkspaceContract) -> None:
    (starter_dir / "build_memory.py").write_text(
        '"""Build a memory package from posed RGB-D for one scene/episode.\n\n'
        "Fill this in. Write a validated package under\n"
        "memories/agent_designed/<dataset>/<scene>/<run-id>/.\n"
        '"""\n\n\n'
        "def build_memory(scene_dir, output_package_dir):\n"
        "    raise NotImplementedError(\"designer must implement build_memory\")\n",
        encoding="utf-8",
    )
    for track, fn in contract.entrypoint_names.items():
        (starter_dir / f"{fn}.py").write_text(
            f'"""Fixed-API entrypoint for {track}: {fn}(package_dir, query) -> dict."""\n\n\n'
            f"def {fn}(package_dir, query):\n"
            f"    raise NotImplementedError(\"designer must implement {fn}\")\n",
            encoding="utf-8",
        )


def _write_readme(path: Path, contract: WorkspaceContract) -> None:
    path.write_text(
        "# Designer Workspace\n\n"
        "Read `CONTRACT.md` first, then `metrics.md` and `shared_modules.md`.\n"
        "Implement `starter/build_memory.py` and the per-track entrypoints, then\n"
        "produce a validated memory package. The harness will build, validate, and\n"
        f"score it on the Track 1/2/3 evaluators for dataset `{contract.dataset}`.\n",
        encoding="utf-8",
    )
