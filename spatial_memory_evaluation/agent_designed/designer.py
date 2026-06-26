"""Invoke the designer coding agent — THE SINGLE UN-FIRED SEAM.

Everything else in this package is wired and runnable. This is the one place that
actually launches a coding agent to read the workspace and write/edit memory code.
It is intentionally left as a documented stub: per the implementation plan, all
preparation up to (but not including) firing the agent is done first.

The intended backend is Claude Code through Bedrock, run as a coding agent with the
workspace as its working directory (distinct from the per-query tool-LLM evaluator
and from the Track 3 judge). The concrete command we will use:

    timeout {turn_timeout} claude -p "{round_prompt}" \
        --model {coding_model} \
        --permission-mode bypassPermissions \
        --add-dir {workspace_root} \
        --max-turns {max_turns} \
        --output-format text

where ``round_prompt`` points the agent at CONTRACT.md / metrics.md /
shared_modules.md, tells it whether to author its own dev_tests (auto_research) or
not (loop_fixed_tests), and — from round 2 on — includes the previous round's dev
report + failing cases + the journal tail as feedback. The agent edits files in
``workspace.root`` and writes its design (build_memory.py, the entrypoints,
schema.md, capabilities.json) into ``design_dir``.

To turn this on: replace the stub body below with the subprocess launch, collect
the written design from ``design_dir``, and return ``status="ok"``. No other file
in this package needs to change — the harness already consumes ``DesignResult``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workspace import Workspace

# Default coding-agent transport (filled in when the seam is enabled). The harness
# passes a concrete command template; this documents the shape.
DEFAULT_CODING_AGENT_COMMAND = (
    'timeout {turn_timeout} claude -p "{round_prompt}" '
    "--model {coding_model} --permission-mode bypassPermissions "
    "--add-dir {workspace_root} --max-turns {max_turns} --output-format text"
)
DEFAULT_CODING_MODEL = "us.anthropic.claude-opus-4-8[1m]"


@dataclass(frozen=True)
class DesignResult:
    """What a designer round produces.

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
    coding_model: str = DEFAULT_CODING_MODEL,
    max_turns: int = 60,
    turn_timeout: int = 3600,
    budget_usd: float | None = None,
) -> DesignResult:
    """Run the designer agent over ``workspace`` and write code into ``design_dir``.

    Args:
        workspace: the sandbox produced by ``build_workspace``.
        design_dir: where the agent writes its memory design.
        feedback: previous round's dev report + failing cases + journal tail
            (``None`` on round 0). NEVER contains held-out data.
        llm_command: coding-agent transport template; defaults to
            ``DEFAULT_CODING_AGENT_COMMAND`` when the seam is enabled.
        coding_model / max_turns / turn_timeout / budget_usd: launch knobs,
            recorded in the result cost.

    Returns a ``DesignResult``. While the seam is OFF this returns
    ``status="skipped"`` with a clear message rather than fabricating a design, so
    the harness records a clean ``ready_for_designer`` run.
    """

    design_dir.mkdir(parents=True, exist_ok=True)
    # === SEAM: enable the coding-agent launch here ===
    # command = (llm_command or DEFAULT_CODING_AGENT_COMMAND).format(
    #     turn_timeout=turn_timeout, round_prompt=_render_round_prompt(workspace, feedback, design_dir),
    #     coding_model=coding_model, workspace_root=workspace.root, max_turns=max_turns,
    # )
    # subprocess.run(command, shell=True, cwd=workspace.root, check=True)
    # then collect build_memory.py + entrypoints + capabilities.json from design_dir.
    return DesignResult(
        status="skipped",
        design_dir=design_dir,
        message=(
            "invoke_designer seam is OFF (preparation-only). All inputs are ready: "
            f"workspace={workspace.root} variant={workspace.contract.variant} "
            f"dev_scenes={list(workspace.split.dev_scene_ids)} "
            f"feedback={'present' if feedback else 'none'} "
            f"transport={'custom' if llm_command else 'default-template'}. "
            "Enable the SEAM block to launch the coding agent."
        ),
        cost={
            "budget_usd": budget_usd,
            "coding_model": coding_model,
            "max_turns": max_turns,
            "turn_timeout": turn_timeout,
            "tokens": None,
            "wall_clock_seconds": None,
        },
    )
