"""Contract constants for the agent-designed memory baseline.

These are the stable agreements the harness, the workspace builder, and the
(future) designer agent all depend on. Keeping them in one place lets the
skeleton stay coherent before the designer invocation is implemented.
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

# Variants of the baseline (increasing capability). See agent_designed_baseline.md.
VARIANTS = ("prompt_only", "few_example", "coding_agent", "iterative")

# Files/inputs that must NEVER be placed into a designer workspace (anti-leakage).
FORBIDDEN_WORKSPACE_INPUTS = (
    "test answers / held-out answers",
    "ground-truth label files",
    "held-out queries",
    "evaluator adapter code",
    "the LLM-judge prompt or judge model config",
    "any benchmark answers.jsonl",
)

# Files the workspace SHOULD provide to the designer.
PROVIDED_WORKSPACE_INPUTS = (
    "CONTRACT.md",
    "shared_modules.md",
    "metrics.md",
    "examples/ (1-2 example memory packages)",
    "train/ (training scenes RGB-D links + training queries, no answers)",
    "starter/ (build_memory.py / query_*.py templates)",
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
    tracks: tuple[str, ...] = ("track1_object_location", "track2_scanrefer", "track3_openeqa")
    must_output: tuple[str, ...] = (
        "build_memory.py (posed RGB-D -> validated memory package)",
        "schema.md and any schemas/*.json",
        "query/tool interface implementing the fixed-API entrypoints it claims",
        "capabilities.json declaring supported vs invalid tracks",
        "README.md (how to build and query)",
        "evidence format for every prediction",
    )
    constraints: tuple[str, ...] = (
        "no access to test answers / GT labels / held-out queries",
        "no test-query-specific hardcoded rules",
        "same raw inputs and shared modules as hand-built methods",
        "respect the per-build / per-round compute & token budget",
        "produced memory must be reproducible (re-running build yields equivalent package)",
        "every query must return evidence",
    )
    entrypoint_names: Mapping[str, str] = field(default_factory=lambda: dict(ENTRYPOINT_NAMES))

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown agent-designed variant: {self.variant!r}; expected one of {VARIANTS}")
