# Multi-frame VLM Control Schema

This is a **no-explicit-memory control**, not a spatial-memory baseline. It
exists to bound how well a VLM answers from raw sampled frames *without* any
explicit object memory, so it can be compared against object-memory methods. It
must never be read as a Track 1/2 object-memory baseline.

Coordinate frame and units: there is no reconstructed world frame. Inputs are
raw sampled camera frames plus per-frame pose/time text. Where pose text is
present its distances are in meters, but no object is ever localized in a shared
3D world frame by this control.

Object id format: none. This control produces no object inventory, so there are
no object ids (no `obj_0001`-style identifiers) and no per-object records.

Timestamp format: per-frame timestamps are passed to the VLM as text alongside
each sampled frame (ISO-8601 or seconds-since-start, as recorded in
`raw_links/sampled_frames.jsonl`). They index raw frames, not memory entries.

Relation representation: none. No object-object spatial relations are stored or
exported.

Confidence or score meaning: none. There are no object scores or retrieval
scores because there is no object memory and no fixed object-location query API.

Native artifact formats: the only artifact is
`raw_links/sampled_frames.jsonl`, a UTF-8 JSONL index of the raw sampled frames
the VLM reads at query time, with `frame_id`, `timestamp`, and optional
`pose_text`. It is an ablation/control input, not exported memory, and is
disabled for object-memory fixed APIs.

Known limitations and unsupported tracks: Track 1 (object inventory), Track 2
(object-location query), Track 3 (ScanRefer referring query), and Track 4
(OpenEQA QA) fixed APIs are all **invalid** by design:

- Track 1 invalid: no explicit object memory (no labels + 3D positions built at
  memory-construction time).
- Track 2 invalid: no fixed object-location query API over memory artifacts.
- Track 3 invalid: no referring-expression resolver over object memory.
- Track 4 invalid: any answer comes from raw frames, not exported memory.

`manifest.explicit_memory` is `false` and `method.family` is
`raw_frame_control`. The raw sampled frames are an ablation/control input only;
they must never enter the main Track 1/2 fixed object-memory API table. This
control is evaluated agentic/control-only.
