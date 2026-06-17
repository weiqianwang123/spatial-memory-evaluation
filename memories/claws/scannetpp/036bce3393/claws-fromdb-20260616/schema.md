# ClawS SpatialRAG Minimal Memory Schema

Coordinate frame and units: object positions use the ScanNet++ world frame
produced from aligned iPhone camera poses (same frame as the Track 1 GT object
inventory). Units are meters. Positions come from the ClawS spatial memory
record `pos_x/pos_y/pos_z`.

Object id format: `claws_<memory_id>`, where `<memory_id>` is the native
`spatial_memories` rowid in the ClawS sqlite-vec database.

Label meaning: `label` is extracted from the memory `snapshot_text` (the leading
`**<object>**` token or `object:` line), which is the YOLO11/COCO class assigned
by the ClawS visual trigger and optionally refined by the VLM describer. The
full `snapshot_text` description is preserved per object for evidence/agentic use.

Timestamp format: `timestamp` is the ClawS memory timestamp (float seconds).

Relation representation: this package exports object-level memory only. ClawS
does not export an explicit relation graph; relations live implicitly inside the
free-text `snapshot_text`.

Confidence: `confidence` is null because the saved ClawS memory record does not
expose a single scalar object confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `evidence/object_snapshots.jsonl` keeps each object's full
snapshot text and crop link. `evidence/crops/<memory_id>.jpg` holds
183 exported object crops. The native ClawS sqlite-vec database is
`/home/robin_wang/ClawS-SpatialRAG/outputs/scannetpp_memory_036bce3393_ollama_vlm.db` (vec0 table `spatial_memories` + `crop_images`).

Known limitations and unsupported tracks: Track 1 object inventory and Track 2
basic object query are supported. Track 3 ScanRefer and Track 4 OpenEQA are
invalid for this package (no native referring-expression resolver or general QA
API is exported here).
