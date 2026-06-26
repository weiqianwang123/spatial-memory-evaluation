"""Contract constants for the agent-designed memory baseline (auto-research).

These are the stable agreements the harness, the workspace builder, and the
(future) designer agent all depend on. Keeping them in one place lets the whole
package stay coherent. See ``.codex/agent_designed_baseline.md``.

The framework is an *auto-research self-improvement loop*: a coding agent designs
a semantic-map memory, builds it on DEV scenes, self-evaluates on its own DEV test
cases (via the unchanged Track 1/2/3 evaluators), reads its own failures, edits its
code, and repeats until a budget / no-gain convergence; only the FROZEN best design
is scored once on the HELD-OUT 10 scenes. The designer never sees held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


AGENT_DESIGNED_FAMILY = "agent_designed"

# The fixed-API entrypoint function names a designed package may implement, keyed
# by track. These match the recommended signatures in memory_package_spec.md.
ENTRYPOINT_NAMES: Mapping[str, str] = {
    "track1_object_location": "query_object",
    "track2_scanrefer": "resolve_referring_expression",
    "track3_openeqa": "answer_question",
}

TRACK_KEYS = tuple(ENTRYPOINT_NAMES.keys())

# Variants of the loop (see agent_designed_baseline.md §9). ``auto_research`` is the
# centerpiece (loops AND the agent authors its own dev tests); the others are
# ablations isolating where the gains come from.
VARIANTS = ("one_shot", "loop_fixed_tests", "auto_research")

# Default variant for the CLI / harness.
DEFAULT_VARIANT = "auto_research"

# Files/inputs that must NEVER be placed into a designer workspace (anti-leakage).
# The held-out 10 scenes, in any form, are first-class forbidden inputs.
FORBIDDEN_WORKSPACE_INPUTS = (
    "held-out (the 10 benchmark) scenes, frames, or layouts",
    "held-out GT labels / bboxes / answers",
    "held-out queries or questions",
    "any benchmark answers.jsonl for the held-out split",
    "evaluator adapter code",
    "the LLM-judge prompt or judge model config",
)

# Files the workspace SHOULD provide to the designer (DEV-only).
PROVIDED_WORKSPACE_INPUTS = (
    "CONTRACT.md",
    "shared_modules.md",
    "metrics.md",
    "examples/ (1-2 example memory packages)",
    "dev_scenes/ (DEV RGB-D layout links + GT-derivation tooling)",
    "dev_tests/ (agent-authored self-tests live here; harness-seeded in loop_fixed_tests)",
    "starter/ (build_memory.py / query_*.py templates)",
    "journal.jsonl (append-only experiment journal across rounds)",
    "README.md",
)


@dataclass(frozen=True)
class WorkspaceContract:
    """What a designer agent must produce and how it is scored.

    This is documentation-as-data: the harness writes it into ``CONTRACT.md`` so
    the designer sees exactly the same contract the harness enforces.
    """

    variant: str
    dataset: str
    tracks: tuple[str, ...] = TRACK_KEYS
    authors_own_dev_tests: bool = True
    must_output: tuple[str, ...] = (
        "build_memory.py (posed RGB-D for one scene -> validated memory package)",
        "schema.md and any schemas/*.json",
        "query/tool interface implementing the fixed-API entrypoints it claims",
        "capabilities.json declaring supported vs invalid tracks",
        "README.md (how to build and query)",
        "evidence format for every prediction",
        "dev_tests/ self-tests (auto_research variant only): T1/T2/T3-shaped "
        "queries + GT derived from the DEV scenes via the provided tooling",
    )
    constraints: tuple[str, ...] = (
        "NEVER access held-out (the 10 benchmark) scenes, GT, or queries in any form",
        "no test-query-specific hardcoded rules; no scene-id-conditioned branches",
        "same raw inputs and shared modules (perception stack) as hand-built methods",
        "runtime describers/captioners/embedders must be the LOCAL qwen stack (ollama)",
        "respect the per-round and per-loop compute & token budget",
        "produced memory must be reproducible (re-running build yields equivalent package)",
        "every query must return evidence",
    )
    entrypoint_names: Mapping[str, str] = field(default_factory=lambda: dict(ENTRYPOINT_NAMES))

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(
                f"unknown agent-designed variant: {self.variant!r}; expected one of {VARIANTS}"
            )
