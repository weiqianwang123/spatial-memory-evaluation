"""Anti-leakage scan for an agent-designed memory design.

The primary guarantee is STRUCTURAL: the held-out 10 scenes are never placed in
the workspace, so the designer cannot read them. This scanner is the secondary
defense — it inspects everything the designer wrote (code, dev tests, configs) for
(a) held-out scene ids used as literals / branch conditions and (b) embedded
held-out answer strings.

It is high-precision (literal substring match over text files) but not a semantic
data-flow analysis, so ``scanner_complete`` reflects that honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Text-ish files worth scanning; binary/large artifacts are skipped.
_SCAN_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".cfg", ".toml", ".sh"}
_MAX_BYTES = 5_000_000  # skip files larger than 5MB (e.g. embeddings/ply)


@dataclass
class LeakageReport:
    clean: bool
    scanner_complete: bool
    findings: list[dict[str, Any]] = field(default_factory=list)
    files_scanned: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "scanner_complete": self.scanner_complete,
            "files_scanned": self.files_scanned,
            "findings": self.findings,
        }


def scan_for_leakage(
    *,
    design_dir: Path,
    heldout_scene_ids: list[str],
    heldout_answer_strings: list[str] | None = None,
    extra_dirs: list[Path] | None = None,
) -> LeakageReport:
    """Scan designed artifacts for held-out scene ids / answer strings.

    Args:
        design_dir: directory holding the agent-written build/query code.
        heldout_scene_ids: scene ids the designer must not reference or branch on.
        heldout_answer_strings: optional GT answer strings that must not appear.
        extra_dirs: additional dirs to scan (e.g. the workspace's dev_tests/).

    A finding on a held-out scene id is a hard failure (``clean=False``). The scan
    covers text files only; ``scanner_complete=False`` flags that obfuscated or
    data-flow leakage is out of scope.
    """

    findings: list[dict[str, Any]] = []
    needles = [(sid, "heldout_scene_id") for sid in heldout_scene_ids if sid]
    needles += [(s, "heldout_answer_string") for s in (heldout_answer_strings or []) if s]

    roots = [design_dir] + list(extra_dirs or [])
    files_scanned = 0
    for root in roots:
        if not root or not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in _SCAN_SUFFIXES:
                continue
            try:
                if f.stat().st_size > _MAX_BYTES:
                    continue
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            files_scanned += 1
            for needle, kind in needles:
                if needle in text:
                    findings.append(
                        {
                            "kind": kind,
                            "needle": needle,
                            "file": str(f),
                        }
                    )

    return LeakageReport(
        clean=not findings,
        scanner_complete=False,
        findings=findings,
        files_scanned=files_scanned,
    )
