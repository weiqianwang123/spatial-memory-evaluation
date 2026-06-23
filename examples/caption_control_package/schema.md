# LLM-with-captions Caption Control Schema

Coordinate frame and units: caption `position` is the mean robot/camera position
over the caption window, in the native ReMEmbR/dataset world frame. Distances are
in meters. `theta` is a yaw placeholder (radians); ReMEmbR sets it to a constant
(`remembr/scripts/preprocess_captions.py:114`), so it is not a reliable heading.

Object id format: not applicable. This control has no object-level memory and
therefore no object ids. Each row is a caption window keyed by `caption_id`
(the native ReMEmbR caption `id`, typically a frame timestamp string).

Timestamp format: `time` is a float (seconds) — the mean timestamp of the frames
in the caption window. `file_start`/`file_end` reference native frame names when
available.

Relation representation: none. Caption memory stores free text per window with no
object nodes and no relation edges.

Confidence or score meaning: none. Captions carry no per-object confidence or
retrieval score; this control exposes no scored predictions.

Native artifact formats: `memory/captions.jsonl` is UTF-8 JSONL with one caption
window per line (`caption_id`, `caption`, `time`, `position`, `theta`). When built
from a native ReMEmbR caption JSON, the original file is copied verbatim under
`memory/native/`. The native schema is ReMEmbR `MemoryItem`
(`remembr/memory/memory.py:5-9`): `caption`, `time`, `position`, `theta` only.
This package contains 3 caption window(s).

Known limitations and unsupported tracks: this is a no-explicit-memory caption
**control** (`explicit_memory=false`), not an object-memory baseline. All fixed
APIs are `invalid`:

- Track 1 (`track1_object_location`): invalid — no native object inventory and no
  deterministic native object-location query API; the only answerer
  (`NonAgent.query`, `remembr/agents/non_agent.py:40-91`) is an LLM over caption
  context and may not be used as fixed-API support.
- Track 2 (`track2_scanrefer`): invalid — no referring-expression resolver.
- Track 3 (`track3_openeqa`): invalid — caption memory has no method-native
  QA/retrieval fixed API; ReMEmbR's native QA path is the agentic `ReMEmbRAgent`
  tool-LLM route, not this fixed API.

The caption artifact is preserved so that the agentic tool-LLM path (Track 3)
can read and reason over captions later. That agentic use is explicitly separate
from the fixed API and must never promote this control to an object-memory API.
