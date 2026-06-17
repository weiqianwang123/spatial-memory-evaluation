# DualMap Minimal Memory Schema

Coordinate frame and units: object positions and bounding boxes use the DualMap
local map frame produced from ScanNet++ aligned camera poses. Units are meters.

Object id format: object ids follow the native DualMap pickle stem, usually the
UUID assigned by DualMap.

Timestamp format: this smoke package exports static object memory, so object
timestamps are not present. Source frame indices are recorded in
`raw_links/native_sources.json`.

Relation representation: this smoke package does not export DualMap global
relations or navigation graph state. It only exports concrete object memory from
DualMap local map pickles.

Confidence or score meaning: `label` is derived from DualMap's configured YOLO
class list and the object `class_id`. `confidence` and `label_score` are null
because the saved local map pickle does not expose one scalar confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `memory/object_features.npy` is present only when object CLIP
features are readable and shape-consistent; present=True. The native
DualMap map directory is `data/dualmap_native/scannetpp_036bce3393/dualmap-smoke-20260615-191702/scannet_scannetpp_036bce3393/map`.

Known limitations and unsupported tracks: Track 1 object inventory and Track 2
basic object query are exported for smoke testing. Track 3 ScanRefer and Track 4
OpenEQA are invalid for this package.
