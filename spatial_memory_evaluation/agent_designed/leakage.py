"""Anti-leakage scan for an agent-designed memory design (STUB).

A designed memory must not embed test answers or condition its behavior on
held-out scene ids. The primary guarantee is structural: the held-out split is
never shown to the designer. This scanner is a secondary defense that inspects the
designed code for embedded answer strings and held-out-scene-id branches.

This is a documented stub: it defines the report contract and a couple of cheap,
high-precision checks, and flags ``scanner_complete=false`` so a clean result is
not over-trusted until the full scanner lands in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LeakageReport:
    clean: bool
    scanner_complete: bool
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "scanner_complete": self.scanner_complete,
            "findings": self.findings,
        }


def scan_for_leakage(
    *,
    design_dir: Path,
    heldout_scene_ids: list[str],
    heldout_answer_strings: list[str] | None = None,
) -> LeakageReport:
    """Scan designed code for embedded held-out answers / scene-id conditioning.

    Args:
        design_dir: directory holding the agent-written build/query code.
        heldout_scene_ids: scene ids the designer must not branch on.
        heldout_answer_strings: optional GT answer strings that must not appear
            verbatim in the code.

    STUB scope: flags literal occurrences of held-out scene ids and answer strings
    in ``*.py``. Semantic checks (data-flow, obfuscated literals) are Phase-4 work,
    so ``scanner_complete`` is ``False``.
    """

    findings: list[dict[str, Any]] = []
    needles = [(sid, "heldout_scene_id") for sid in heldout_scene_ids if sid]
    needles += [(s, "heldout_answer_string") for s in (heldout_answer_strings or []) if s]

    if design_dir.exists():
        for py_file in sorted(design_dir.rglob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for needle, kind in needles:
                if needle in text:
                    findings.append(
                        {
                            "kind": kind,
                            "needle": needle,
                            "file": str(py_file.relative_to(design_dir)),
                        }
                    )

    return LeakageReport(clean=not findings, scanner_complete=False, findings=findings)
