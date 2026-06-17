# HOV-SG Minimal Memory Schema

Coordinate frame and units: object positions and bounding boxes use the HOV-SG
world coordinate frame produced from ScanNet++ aligned camera poses. Units are
meters.

Object id format: object ids follow the native HOV-SG object PLY stem, such as
`pcd_0` or `pcd_143`.

Timestamp format: this smoke package exports static object memory, so object
timestamps are not present. Source frame indices are recorded in
`raw_links/native_sources.json`.

Relation representation: this smoke package does not export room or relation
edges. It only exports object-level geometry from HOV-SG semantic segmentation.

Confidence or score meaning: `label_score` is a CLIP text similarity when label
classification succeeds. It can be null when labels are not classified.
`confidence` is null because HOV-SG object PLYs do not expose one scalar object
confidence.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. `memory/object_features.npy` is present only when
`mask_feats.pt` was readable during export; present=True. The native
HOV-SG result directory is `data/hovsg_native/scannetpp_036bce3393/hovsg-stride10-20260616-124403/scannet`.

Known limitations and unsupported tracks: Track 1 object inventory and Track 2
basic object query are exported for smoke testing. Track 3 ScanRefer and Track 4
OpenEQA are invalid for this package.
