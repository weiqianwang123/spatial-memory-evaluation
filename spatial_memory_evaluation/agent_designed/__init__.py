"""Agent-designed memory baseline (auto-research self-improvement).

A coding agent designs its own semantic-map spatial memory (schema + construction
code + query/tool interface), builds it on DEV scenes, self-evaluates on its OWN
dev test cases via the unchanged Track 1/2/3 evaluators, reads its failures, edits
its code, and repeats until budget / no-gain convergence. Only the FROZEN best
design is scored once on the held-out 10 scenes. See
``.codex/agent_designed_baseline.md``.

Everything except the actual designer-agent launch (``invoke_designer``) is wired
and runnable: split definition, dev-scene workspace assembly, the experiment
journal + best-so-far snapshot/revert, dev self-evaluation, the leakage scan, and
the harness loop. ``invoke_designer`` is the single documented seam to flip on.
"""

from .contract import (
    AGENT_DESIGNED_FAMILY,
    DEFAULT_VARIANT,
    ENTRYPOINT_NAMES,
    FORBIDDEN_WORKSPACE_INPUTS,
    TRACK_KEYS,
    VARIANTS,
    WorkspaceContract,
)
from .dev_eval import PRIMARY_METRIC, DevEvalResult, evaluate_dev
from .journal import Journal, RoundRecord
from .splits import (
    DEFAULT_DEV_SCENE_IDS,
    HELDOUT_SCENE_IDS,
    Split,
    default_split,
    write_split_manifest,
)
from .workspace import Workspace, build_workspace

__all__ = [
    "AGENT_DESIGNED_FAMILY",
    "DEFAULT_VARIANT",
    "ENTRYPOINT_NAMES",
    "FORBIDDEN_WORKSPACE_INPUTS",
    "TRACK_KEYS",
    "VARIANTS",
    "WorkspaceContract",
    "PRIMARY_METRIC",
    "DevEvalResult",
    "evaluate_dev",
    "Journal",
    "RoundRecord",
    "DEFAULT_DEV_SCENE_IDS",
    "HELDOUT_SCENE_IDS",
    "Split",
    "default_split",
    "write_split_manifest",
    "Workspace",
    "build_workspace",
]
