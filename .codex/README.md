# Codex Notes

Start with `project-context.md` when working in this repository.

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
