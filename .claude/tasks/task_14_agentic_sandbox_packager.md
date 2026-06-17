# Task 14: Agentic Sandbox Packager

## Goal

Build reusable sandbox packaging utilities for agentic evaluation: copy memory
package, method adapter code, shared module code, and original method root
source code into an isolated workspace.

## Scope

Implement source-context packaging primitives only. Do not change Track 1/2
scoring or fixed API behavior.

## Context Files

- `.codex/agentic_eval_plan.md`
- `.codex/memory_package_spec.md`
- Existing agentic code paths under `spatial_memory_evaluation/`
- `scripts/methods/`
- `spatial_memory_evaluation/shared_modules/`

## Implementation Rules

- Copy memory package into sandbox; do not mutate the original package.
- Copy `scripts/methods/<method>/` when present.
- Copy `spatial_memory_evaluation/shared_modules/` and
  `scripts/methods/shared_modules.py`.
- Copy original method root repo source from `manifest.method.repo_path`.
- Exclude `.git`, caches, generated data, checkpoints, memories, results, and
  other heavy/leaky artifacts.
- Keep source, configs, scripts, schema, README, and docs.
- GT answers must not enter sandbox.

## Deliverables

- Reusable sandbox/source-copy helper.
- Include/exclude policy documented in code or docs.
- Smoke command that creates a sandbox for an existing package.
- Clear errors when source paths are missing.

## Acceptance Checks

- Sandbox contains package, adapter code, shared module code, and method root
  source code.
- Sandbox excludes GT answers and heavy generated artifacts.
- Existing fixed API runs are unaffected.

## PR Title

`feat: add agentic sandbox source packager`
