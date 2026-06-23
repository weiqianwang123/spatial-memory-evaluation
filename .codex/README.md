# Codex Notes

Start with `project-context.md` when working in this repository.

For the agentic benchmark direction, read these in order:

1. `agentic_eval.md`: research vision (3 tracks + agent-designed memory).
2. `agentic_eval_plan.md`: execution roadmap and milestones.
3. `memory_package_spec.md`: minimal package contract, capabilities, and schema.
4. `agent_designed_baseline.md`: the centerpiece agent-designed memory baseline.
5. `baseline_registry.md`: baseline methods and track-level API support.
6. `path_registry.md`: data, checkpoint, repo, output, and runtime paths.
7. `modules.md`: shared module/checkpoint registry for fair comparisons.

The benchmark has three tracks. Capability keys used everywhere:
`track1_object_location`, `track2_scanrefer`, `track3_openeqa`. The old
5-track / 4-key layout (`track1_memory_construction`, `track2_object_location`,
`track3_scanrefer`, `track4_openeqa`) is retired.

Keep generated artifacts out of Git:

- `memories/`: method-generated spatial memory, exported DBs, maps, and other reusable memory artifacts.
- `results/`: predictions, metrics, reports, logs, and other evaluation outputs.
- `data/`: optional local data cache. Prefer the NAS paths in configs when available.

Result outputs must use:

```text
results/<method>/<evaluation>/<timestamp>/
```

Where `<evaluation>` is e.g. `track1-fixed_api`, `track2-tool_llm`,
`track3-fixed_api`, or `agent_designed-iterative`.

Use `_data` as the method folder only for data-prep or data-check outputs that
do not belong to a specific method.
