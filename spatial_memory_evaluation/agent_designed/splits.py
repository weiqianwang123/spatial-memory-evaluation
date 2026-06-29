"""Dataset splits for the agent-designed memory baseline.

Single source of truth for which ScanNet scenes are HELD-OUT (the 10 shared
benchmark scenes, scored once after freeze) versus DEV (scenes OUTSIDE the 10 that
the self-improvement loop builds/evaluates on). The held-out list is duplicated
here intentionally so the anti-leakage guarantee does not depend on importing the
eval drivers; a unit check (``assert_disjoint``) guards against overlap.

See ``.codex/agent_designed_baseline.md`` §5 and ``.codex/eval_set_inventory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import write_json


# The 10 shared ScanNet scenes used by every method's reported result. The
# self-improvement loop must NEVER build, evaluate, read, or branch on these.
HELDOUT_SCENE_IDS: tuple[str, ...] = (
    "scene0015_00",
    "scene0050_00",
    "scene0077_00",
    "scene0084_00",
    "scene0131_00",
    "scene0193_00",
    "scene0207_00",
    "scene0222_00",
    "scene0256_00",
    "scene0314_00",
)

# Default DEV scenes (outside the held-out 10) chosen to span small/medium/large
# rooms with strong Track 2 (referring) + Track 3 (QA) coverage. All have a local
# ``.sens``; their GT geometry is fetched from kaldir by download_scannet_gt.sh.
#   scene0527_00: small  (~155MB .sens, 92 refer / 13 QA)
#   scene0406_00: medium (~432MB .sens, 72 refer / 14 QA)
#   scene0426_00: large  (~901MB .sens, 97 refer / 13 QA)
# Changeable: edit this tuple (or pass --dev-scene-id) and re-run prepare_dev_scene.sh.
DEFAULT_DEV_SCENE_IDS: tuple[str, ...] = (
    "scene0527_00",
    "scene0406_00",
    "scene0426_00",
)

DATASET = "scannet"


@dataclass(frozen=True)
class Split:
    dataset: str
    dev_scene_ids: tuple[str, ...]
    heldout_scene_ids: tuple[str, ...]

    def assert_disjoint(self) -> None:
        overlap = set(self.dev_scene_ids) & set(self.heldout_scene_ids)
        if overlap:
            raise ValueError(
                "anti-leakage violation: dev scenes overlap held-out scenes: "
                f"{sorted(overlap)}"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dev_scene_ids": list(self.dev_scene_ids),
            "heldout_scene_ids": list(self.heldout_scene_ids),
            "note": (
                "DEV scenes are built/evaluated by the self-improvement loop; "
                "HELD-OUT scenes are scored once after freeze and are NEVER shown "
                "to the designer. Edit dev_scene_ids to change the dev split."
            ),
        }


def default_split(dev_scene_ids: tuple[str, ...] | None = None) -> Split:
    split = Split(
        dataset=DATASET,
        dev_scene_ids=tuple(dev_scene_ids) if dev_scene_ids else DEFAULT_DEV_SCENE_IDS,
        heldout_scene_ids=HELDOUT_SCENE_IDS,
    )
    split.assert_disjoint()
    return split


def write_split_manifest(path: Path, split: Split) -> Path:
    """Write ``splits.json`` next to the benchmarks (or anywhere the caller wants)."""

    split.assert_disjoint()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, split.to_json())
    return path
