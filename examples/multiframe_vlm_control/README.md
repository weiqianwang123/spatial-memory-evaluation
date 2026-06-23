# Multi-frame VLM Control Package (fixture)

Minimal **no-explicit-memory control** package fixture. It demonstrates how a
raw-frame VLM control (the ReMEmbR `VLMNonAgent` path) is represented so it can
never be confused with a Track 1/2 object-memory baseline.

Key markers:

- `manifest.explicit_memory = false`
- `manifest.method.family = raw_frame_control`
- All four fixed-API tracks declared `invalid` with control-only reasons.
- Raw sampled frames live under `raw_links/` and are clearly marked as
  ablation/control input, never exported object memory.

Validate it:

```bash
python scripts/package/validate_memory_package.py examples/multiframe_vlm_control
```

Run a fixed-API track against it to see the readable control invalid result:

```bash
python scripts/evaluate_track1.py examples/multiframe_vlm_control --mode fixed_api
python scripts/evaluate_track2.py examples/multiframe_vlm_control --mode fixed_api
```

Both Track 1 and Track 2 return `status=invalid` with
`reason_code=control_no_explicit_memory`, `control=true`, and a message that
says there is no explicit object memory / no fixed object-location API. See
`.codex/baseline_registry.md` (Multi-frame VLM control section) for the policy.
