#!/usr/bin/env python
"""Initialize the sandbox as a git repo for the AutoResearch keep/revert loop.

The git log becomes the experiment journal: autoresearch_round.py commits every
improving round and reverts regressions. We track the DESIGN code + the (small)
memory package artifacts + history, and ignore the heavy/derived inputs (frame
symlinks, large model caches, per-query eval scratch).

Run once per sandbox:
    python init_sandbox_git.py --sandbox <dir>
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

GITIGNORE = """\
# heavy / derived — not part of the design history
dev_scenes/
dev_tests_ref/
examples/
__pycache__/
*.pyc
memories/**/_dev_eval/
_run_logs/
# The experiment journal must NOT be git-tracked: autoresearch_round.py appends a
# round's result row AFTER its commit, so a later REVERT's `git reset --hard` would
# silently delete the prior KEEP's uncommitted row, corrupting best-so-far. Keeping
# it untracked makes reverts touch only design files, never the journal.
history.jsonl
progress.png
# keep: starter/ (design code), dev_tests/ (fixed tests), memories/ packages,
#       *.md, sandbox_config.json, splits.json
"""


def _git(sandbox: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=sandbox, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", type=Path, required=True)
    args = ap.parse_args()
    sb = args.sandbox

    (sb / ".gitignore").write_text(GITIGNORE)
    if not (sb / ".git").exists():
        _git(sb, "init", "-q")
        _git(sb, "config", "user.email", "autoresearch@local")
        _git(sb, "config", "user.name", "autoresearch")
    _git(sb, "add", "-A")
    rc, out = _git(sb, "commit", "-q", "-m", "round 0: workspace baseline (pre-design)")
    head = _git(sb, "rev-parse", "--short", "HEAD")[1].strip()
    print(f"sandbox git initialized at {sb} (HEAD {head})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
