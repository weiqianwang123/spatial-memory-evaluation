"""Agent-designed memory baseline harness.

Lets a coding agent design its own spatial memory (schema + construction code +
query/tool interface) under a fixed contract, then scores the resulting package
with the unchanged Track 1/2/3 evaluators. See ``.codex/agent_designed_baseline.md``.

This package is a skeleton: the contract, workspace assembly, and harness control
flow are defined with stable signatures; the actual designer-agent invocation and
leakage scanner are documented stubs to be implemented in Phase 4.
"""

from .contract import (
    AGENT_DESIGNED_FAMILY,
    ENTRYPOINT_NAMES,
    FORBIDDEN_WORKSPACE_INPUTS,
    VARIANTS,
    WorkspaceContract,
)

__all__ = [
    "AGENT_DESIGNED_FAMILY",
    "ENTRYPOINT_NAMES",
    "FORBIDDEN_WORKSPACE_INPUTS",
    "VARIANTS",
    "WorkspaceContract",
]
