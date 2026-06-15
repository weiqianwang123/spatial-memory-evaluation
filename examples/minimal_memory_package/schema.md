# Example Minimal Memory Schema

Coordinate frame and units: positions use an example right-handed world
coordinate frame, and all 3D distances are in meters.

Object id format: object ids use the `obj_0001` style and are stable within this
package.

Timestamp format: timestamps are ISO-8601 strings when present. This fixture has
only one static frame, so per-object time can be null.

Relation representation: object relations are omitted in this fixture. A method
that exports relations should document relation names, direction, and evidence.

Confidence or score meaning: confidence is a method-specific score in `[0, 1]`;
this fixture uses `1.0` for its synthetic object.

Native artifact formats: `memory/object_table.jsonl` is UTF-8 JSONL with one
object per line. Each object includes `object_id`, `label`, `position_3d`,
`bbox_3d`, `confidence`, and `evidence`.

Known limitations and unsupported tracks: this fixture only supports Track 1.
Track 2, Track 3, and Track 4 are intentionally invalid.
