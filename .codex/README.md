# Codex Notes

Start with `project-context.md` when working in this repository.

For the agentic benchmark direction, read these in order:

1. `agentic_eval.md`: research vision.
2. `agentic_eval_plan.md`: execution roadmap and milestones.

Keep generated artifacts out of Git:

- `memories/`: method-generated spatial memory, exported DBs, maps, and other reusable memory artifacts.
- `results/`: predictions, metrics, reports, logs, and other evaluation outputs.
- `data/`: optional local data cache. Prefer the NAS paths in configs when available.

Result outputs must use:

```text
results/<method>/<evaluation>/<timestamp>/
```

Use `_data` as the method folder only for data-prep or data-check outputs that
do not belong to a specific method.
