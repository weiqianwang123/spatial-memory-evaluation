"""Invoke the designer coding agent (STUB).

The designer agent reads the workspace and writes memory-construction + query code
into a design directory. The default backend is Claude Code through Bedrock, run as
a coding agent inside the workspace sandbox (distinct from the per-query tool-LLM
evaluator and from the evaluator judge).

This is a documented stub. It defines the call/return contract so the harness is
fully wired; the actual agent launch is implemented in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workspace import Workspace


@dataclass(frozen=True)
class DesignResult:
    """What a designer run produces.

    ``design_dir`` holds the agent-written ``build_memory.py``, entrypoints,
    ``schema.md``, and ``capabilities.json`` draft. ``status`` is ``ok`` when the
    agent produced a build script and entrypoints, else ``error``/``skipped``.
    """

    status: str
    design_dir: Path
    message: str = ""
    cost: dict[str, Any] | None = None


def invoke_designer(
    *,
    workspace: Workspace,
    design_dir: Path,
    feedback: dict[str, Any] | None = None,
    llm_command: str | None = None,
    budget_usd: float | None = None,
) -> DesignResult:
    """Run the designer agent over ``workspace`` and write code into ``design_dir``.

    Args:
        workspace: the sandbox produced by ``build_workspace``.
        design_dir: where the agent writes its memory design.
        feedback: training-split failure summary for the iterative variant
            (``None`` on the first round). NEVER contains held-out data.
        llm_command: transport command for the coding agent (Claude Code/Bedrock).
        budget_usd: per-round budget; recorded in the result cost.

    Returns a ``DesignResult``. STUB: until Phase 4 this returns ``status="skipped"``
    with a clear message rather than fabricating a design.
    """

    design_dir.mkdir(parents=True, exist_ok=True)
    # TODO(Phase 4): launch the coding agent with the workspace as cwd, e.g.
    #   command = llm_command.format(workspace=workspace.root, design_dir=design_dir)
    #   subprocess.run(command, shell=True, cwd=workspace.root, check=True)
    # then collect build_memory.py + entrypoints + capabilities.json from design_dir.
    return DesignResult(
        status="skipped",
        design_dir=design_dir,
        message=(
            "invoke_designer is a Phase-4 stub. Wire the coding-agent launch here. "
            f"workspace={workspace.root} variant={workspace.contract.variant} "
            f"feedback={'present' if feedback else 'none'} "
            f"llm_command={'set' if llm_command else 'unset'}."
        ),
        cost={"budget_usd": budget_usd, "tokens": None, "wall_clock_seconds": None},
    )
