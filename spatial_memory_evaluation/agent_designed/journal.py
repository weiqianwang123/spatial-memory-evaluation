"""Experiment journal + best-so-far snapshot/revert for the auto-research loop.

This is the loop's memory (AutoResearch's "log of experiments"; Learning Beyond
Gradients' explicit, readable, refactorable history). Each round appends one row:
what changed, dev score before/after, the keep/discard decision, a rationale, and
budget spent. ``snapshot_best`` / ``revert_to_best`` implement the keep-or-discard
rule by copying the design dir to/from a ``best/`` snapshot, so a regression in a
later round is detectable and revertible.

No model calls here — pure bookkeeping the harness drives.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RoundRecord:
    round_index: int
    design_summary: str
    dev_score: float | None
    dev_metrics: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"  # "keep" | "discard" | "pending"
    rationale: str = ""
    leakage_clean: bool | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None  # stamped by the caller (scripts may pass it in)

    def to_json(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "design_summary": self.design_summary,
            "dev_score": self.dev_score,
            "dev_metrics": self.dev_metrics,
            "decision": self.decision,
            "rationale": self.rationale,
            "leakage_clean": self.leakage_clean,
            "cost": self.cost,
            "timestamp": self.timestamp,
        }


class Journal:
    """Append-only round journal backed by a JSONL file, plus a best snapshot."""

    def __init__(self, journal_path: Path, best_dir: Path) -> None:
        self.journal_path = journal_path
        self.best_dir = best_dir
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._best_score: float | None = None
        self._records: list[RoundRecord] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.journal_path.exists():
            return
        for line in self.journal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Reconstruct a lightweight record so converged()/best_score stay
            # correct after a resume-from-journal (only decision + dev_score +
            # ordering matter to the loop's control flow).
            self._records.append(
                RoundRecord(
                    round_index=row.get("round_index", -1),
                    design_summary=row.get("design_summary", ""),
                    dev_score=row.get("dev_score"),
                    decision=row.get("decision", "pending"),
                )
            )
            score = row.get("dev_score")
            if row.get("decision") == "keep" and isinstance(score, (int, float)):
                if self._best_score is None or score > self._best_score:
                    self._best_score = float(score)

    @property
    def best_score(self) -> float | None:
        return self._best_score

    @property
    def rounds_so_far(self) -> int:
        return sum(1 for _ in self.journal_path.read_text(encoding="utf-8").splitlines()) \
            if self.journal_path.exists() else 0

    def improves(self, dev_score: float | None) -> bool:
        """Keep-or-discard rule: strictly improve the best kept dev score."""

        if dev_score is None:
            return False
        if self._best_score is None:
            return True
        return dev_score > self._best_score

    def append(self, record: RoundRecord) -> None:
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
        self._records.append(record)
        if record.decision == "keep" and isinstance(record.dev_score, (int, float)):
            if self._best_score is None or record.dev_score > self._best_score:
                self._best_score = float(record.dev_score)

    def snapshot_best(self, design_dir: Path) -> None:
        """Promote ``design_dir`` to the best snapshot (called on a 'keep')."""

        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        shutil.copytree(design_dir, self.best_dir)

    def revert_to_best(self, design_dir: Path) -> bool:
        """Restore ``design_dir`` from the best snapshot (called on a 'discard').

        Returns False if there is no best snapshot yet (first round can't revert).
        """

        if not self.best_dir.exists():
            return False
        if design_dir.exists():
            shutil.rmtree(design_dir)
        shutil.copytree(self.best_dir, design_dir)
        return True

    def converged(self, no_gain_window: int) -> bool:
        """True if the last ``no_gain_window`` rounds were all discards (no gain)."""

        if no_gain_window <= 0:
            return False
        if len(self._records) < no_gain_window:
            return False
        tail = self._records[-no_gain_window:]
        return all(r.decision == "discard" for r in tail)
