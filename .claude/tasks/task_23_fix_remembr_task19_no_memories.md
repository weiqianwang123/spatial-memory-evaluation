# Task 23: Fix ReMEmbR Task 19 Memory Artifact Leak

## Goal

Repair the Task 19 ReMEmbR branch so it no longer commits anything under
`memories/`, while preserving the correct ReMEmbR Track 1/2 fixed-API conclusion.

## Current Problem

Task 19 correctly concluded that ReMEmbR Track 1/2 fixed API is `invalid`, but it
committed a minimal declaration package under:

```text
memories/remembr/oc-navqa/sequence_0/20260617-000000/
```

This violates the project rule that `memories/` is generated output and must not
be uploaded to git. Deleting those files in a later commit is not enough if the
branch is going to be pushed, because the files would still exist in branch
history. The final branch history must not contain tracked `memories/` files.

## Scope

Work in the existing Task 19 worktree:

- Worktree: `/home/robin_wang/spatial-memory-evaluation-task19`
- Branch: `claude/task-19-remembr-outcome`
- Current commit to repair: `681a172 docs: finalize remembr track12 fixed api outcome`

It is acceptable in this worktree to rewrite the Task 19 branch history so the
final `main..HEAD` commit set contains no tracked `memories/` paths. Do not touch
main or external method repos.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/baseline_registry.md`
- `.codex/memory_package_spec.md`
- `.gitignore`
- Existing ReMEmbR task branch changes
- `/home/robin_wang/remembr`

## Implementation Rules

- Do not modify `/home/robin_wang/remembr`.
- Preserve the correct fixed API outcome:
  - Track 1 invalid: caption/time/robot-pose memory has no object inventory.
  - Track 2 invalid: caption retrieval returns text and robot pose, not object
    locations.
  - Track 3 invalid; Track 4 candidate/agentic, not promoted by this task.
- Remove all tracked `memories/remembr/...` files from the branch history.
- Replace the committed memory package with one of:
  - a generator script under `scripts/methods/remembr/` that writes an ignored
    declaration package when explicitly run; or
  - a small fixture under `examples/` if a static example is genuinely useful.
- Prefer a generator script if possible, because declaration packages are still
  generated artifacts.
- Ensure `.codex/baseline_registry.md` does not point to a committed
  `memories/` path as if it were source code.
- If you rewrite history, leave the branch with one clean focused commit.

## Deliverables

- Clean repaired Task 19 branch.
- No tracked `memories/`, `results/`, or `data/` files in `git diff
  --name-only main...HEAD`.
- Registry docs still contain the ReMEmbR fixed API decision and root evidence.
- Optional source script to generate the invalid declaration package locally.

## Acceptance Checks

- `git log --oneline main..HEAD` is focused and contains no commit that adds
  `memories/`.
- `git diff --name-only main...HEAD | grep '^memories/'` returns nothing.
- `git status --short` is clean after commit.
- `/home/robin_wang/remembr` remains clean.

## PR Title

`docs: finalize remembr fixed api invalid outcome`
