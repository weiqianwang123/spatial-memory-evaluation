# Minimal Memory Package Spec

Last updated: 2026-06-23 (refactor: 3-track keys + agent_designed family)

This spec fixes the package format used by fixed-API evaluators and by the
per-query LLM/tool agentic evaluators. Every method, control baseline, or
agent-designed memory must export one minimal memory package before evaluation.

Implementation files:

- `spatial_memory_evaluation/schemas/memory_package/manifest.schema.json`
- `spatial_memory_evaluation/schemas/memory_package/capabilities.schema.json`
- `spatial_memory_evaluation/schemas/memory_package/schema_md.schema.json`
- `spatial_memory_evaluation/memory_package_validator.py`

Validator command:

```bash
python -m spatial_memory_evaluation.memory_package_validator <package_dir>
python scripts/package/validate_memory_package.py <package_dir>
```

## Track Keys (after the 3-track refactor)

The benchmark has exactly three tracks. The capability keys are stable strings
used by `capabilities.json`, the validator, and the evaluators:

| Track | Capability key | What it tests | Dataset |
|---|---|---|---|
| Track 1 | `track1_object_location` | object-level location query + build cost | ScanNet++ |
| Track 2 | `track2_scanrefer` | instance-level referring query | ScanRefer / ScanNet |
| Track 3 | `track3_openeqa` | general spatial QA | OpenEQA (ScanNet + HM3D) |

Migration note (old → new):

- old `track1_memory_construction` + old `track2_object_location`
  → merged into `track1_object_location`.
- old `track3_scanrefer` → `track2_scanrefer`.
- old `track4_openeqa` → `track3_openeqa`.

Packages written for the old 5-track / 4-key layout must be regenerated.

## Goals

- Give every method a common package boundary while preserving its native memory.
- Make supported and unsupported fixed APIs explicit through `capabilities.json`.
- Let evaluators produce `invalid` for unsupported tracks instead of guessing.
- Give per-query agentic/tool eval a self-contained folder with raw/native memory
  and declared method-native tools. Agentic/tool eval must not depend on
  evaluator-created fixed-API conversion views unless those views are also
  method-native artifacts.
- Support an `agent_designed` family: memory whose schema/build/query code was
  produced by a coding agent under the same contract (see
  `agent_designed_baseline.md`).

## Package Path

```text
memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
```

Examples:

```text
memories/claws/scannetpp/036bce3393/20260623-153000/
memories/daaam/scannetpp/036bce3393/20260623-153000/
memories/agent_designed/scannetpp/036bce3393/iter03-20260623-153000/
memories/remembr/openeqa-scannet/scene0709_00/20260623-153000/
```

`run-id` is normally a `YYYYMMDD-HHMMSS` timestamp; a stable hash is allowed only
if the package is deterministic and documented. For iterative agent-designed runs
prefix with the iteration, e.g. `iter03-<timestamp>`.

## Directory Layout

Required:

```text
manifest.json
capabilities.json
schema.md
memory/
evidence/
raw_links/
schemas/
tools/
build_log.json
```

Recommended: `README.md`, `checksums.json`, `environment.txt`.

For methods with native query/retrieval tools, distinguish:

- `memory/native/`: method-native artifacts used by agentic/tool eval.
- fixed-API conversion views (e.g. `memory/object_table.jsonl`): deterministic
  evaluator-facing views used only by fixed API, unless the method natively stores
  that artifact.
- `tools/`: fixed-API thin entrypoints, not automatically agentic tools.

## Manifest

`manifest.json` describes method, dataset, inputs, artifacts, and leakage
constraints. Required top-level fields:

```json
{
  "schema_version": "0.2",
  "package_id": "claws/scannetpp/036bce3393/20260623-153000",
  "method": {
    "name": "claws",
    "display_name": "ClawS SpatialRAG",
    "family": "object_map",
    "repo_path": "/home/robin_wang/ClawS-SpatialRAG",
    "commit": null,
    "version": null
  },
  "dataset": {
    "name": "scannetpp",
    "split": "current-scene",
    "scene_id": "036bce3393",
    "episode_id": null
  },
  "input": {
    "modality": ["rgb", "depth", "pose", "intrinsics", "timestamp"],
    "frame_count": 0,
    "rgbd_root": null,
    "poses_path": null,
    "intrinsics_path": null,
    "timestamp_path": null,
    "coordinate_frame": "method-defined; see schema.md"
  },
  "explicit_memory": true,
  "memory_artifacts": [],
  "evidence_artifacts": [],
  "raw_links": [],
  "tools": [],
  "build": {
    "command": null,
    "config_paths": [],
    "environment": null,
    "started_at": null,
    "finished_at": null,
    "build_runtime_seconds": null,
    "runtime_seconds": null,
    "frame_count": 0,
    "time_per_frame_seconds": null,
    "native_memory_size_bytes": null,
    "native_memory_artifacts": [],
    "memory_artifact_size_bytes": null,
    "package_size_bytes": null,
    "peak_ram_bytes": null,
    "peak_ram_unavailable_reason": null,
    "peak_vram_bytes": null,
    "peak_vram_unavailable_reason": null
  },
  "allowed_access": {
    "contains_gt_annotations": false,
    "contains_benchmark_answers": false,
    "contains_test_labels": false,
    "contains_question_specific_rules": false
  },
  "notes": ""
}
```

`schema_version` is bumped to `0.2` for the 3-track refactor.

### Build Accounting

Same accounting fields appear in `manifest.json.build` and `build_log.json`
(`build_log.json` is authoritative; manifest mirrors stable fields). Required:

- `frame_count`, `build_runtime_seconds`, `time_per_frame_seconds`
- `native_memory_size_bytes`, `native_memory_artifacts`, `memory_artifact_size_bytes`,
  `package_size_bytes`
- `peak_ram_bytes`, `peak_ram_unavailable_reason`, `peak_vram_bytes`,
  `peak_vram_unavailable_reason`

Rules: `native_memory_size_bytes` is the primary memory-size metric (original
method artifact). `package_size_bytes` reported separately. Size accounting must
not include GT/query files. Peaks filled only when measured; if `null`, the
unavailable reason must be non-empty. `time_per_frame_seconds =
build_runtime_seconds / frame_count` when `frame_count > 0`, else `null`.

Track 1 reads these accounting fields to report the build-cost half of the track.

### Method Family

Use one of:

- `object_map`
- `scene_graph`
- `caption_memory`
- `vector_db`
- `raw_frame_control`
- `caption_control`
- `agent_designed`
- `other`

`agent_designed`: memory whose schema, construction code, and query interface were
authored by a coding agent under the agent-designed baseline. It is a real
explicit-memory family (`explicit_memory=true` unless the agent intentionally
designed a no-memory control) and may declare any track `supported` when it
provides the required entrypoint, exactly like a hand-built method.

Controls (`raw_frame_control` for Multi-frame VLM, `caption_control` for
LLM-with-captions) must set `explicit_memory=false` and declare every fixed-API
track `invalid` (validator rejects any `supported` fixed API for control
families). They are evaluated control-only / agentic-only.

### Artifact Records

Each record in `memory_artifacts` / `evidence_artifacts` / `raw_links` / `tools`
uses package-relative paths and may declare which tracks it supports:

```json
{
  "name": "object_table",
  "type": "jsonl",
  "path": "memory/object_table.jsonl",
  "description": "Canonical object inventory exported from native memory.",
  "required_for": ["track1_object_location"]
}
```

`required_for` values must be valid track keys. Do not store absolute artifact
paths unless they are external raw-data links in `raw_links` (read-only, no GT).

### Raw Native Memory vs Fixed-API Views

- `memory/native/` is preferred for raw/native artifacts copied into the package.
- `raw_links/` records external provenance; agentic/tool eval must not require
  following raw links.
- Agentic/tool eval defaults to raw/native artifacts + method-native tools. It must
  not use evaluator-created fixed-API views as primary memory unless
  `manifest.memory_artifacts[].native=true` or schema states the view is native.
- Fixed-API evaluators may use conversion views (deterministic scoring only).

## Capabilities

`capabilities.json` is the evaluator contract.

```json
{
  "schema_version": "0.2",
  "fixed_api": {
    "track1_object_location": {
      "status": "supported",
      "entrypoint": "tools/query_object.py:query_object",
      "reason": "",
      "input_schema": "schemas/track1_input.schema.json",
      "output_schema": "schemas/object_query_result.schema.json"
    },
    "track2_scanrefer": {
      "status": "invalid",
      "entrypoint": null,
      "reason": "No referring-expression resolver is exported."
    },
    "track3_openeqa": {
      "status": "invalid",
      "entrypoint": null,
      "reason": "No native QA or retrieval API is exported."
    }
  },
  "agent_access": {
    "mode": "tool_llm",
    "read_manifest": true,
    "read_schema": true,
    "read_native_memory": true,
    "read_fixed_api_views": false,
    "read_evidence": true,
    "read_adapter_code": false,
    "read_shared_module_code": false,
    "read_method_root_source_code": true,
    "read_build_code": false,
    "read_raw_links": false,
    "read_raw_frames": false,
    "read_source_keyframes_or_crops": false,
    "run_method_native_tools": true,
    "write_package": false
  }
}
```

### Fixed API Status

- `supported`: evaluator may call the Python entrypoint.
- `invalid`: method/package does not support this track by design.

Do not use `partial`. If a track is not ready, mark it `invalid` and explain why.

### Python Entrypoint

```text
relative/path.py:function_name
```

Rules: path relative to package root; importable without changing cwd outside the
package; must not write into the package; may read package artifacts; receives and
returns JSON-serializable data.

Recommended signatures (renamed to the new track keys):

```python
def query_object(package_dir: str, query: dict) -> dict: ...                 # track1
def resolve_referring_expression(package_dir: str, query: dict) -> dict: ...  # track2
def answer_question(package_dir: str, query: dict) -> dict: ...               # track3
```

`package_dir` is the copied or original package path depending on mode.

### Agentic/Tool Access

`agent_access` describes what an LLM/tool-loop evaluator may use. The formal
agentic setting is **per-query LLM + declared tools**, not a coding agent.

Modes:

- `tool_llm`: method has native retrieval/query tools over raw/native memory. The
  LLM gets one query at a time and may only request declared tool calls or return
  final JSON. Requires `read_native_memory`, `run_method_native_tools`, and
  `read_method_root_source_code` true; requires raw-frame / fixed-API-view /
  adapter-code / build-code access false.
- `not_applicable`: method has no natural LLM/tool-memory interface (e.g. HOV-SG on
  the current ScanNet++ package). Do not force agentic/tool eval; use fixed API.
  Must not declare method-native tool calls.

Optional `agent_tools` array documents each native tool with input/output schema
and which artifacts it reads.

## Track Contracts

### Track 1: Object-Level Location Query (`track1_object_location`)

Input:

```json
{ "query_id": "q_0001", "query": "where is the chair?",
  "target_label": "chair", "canonical_label": "chair",
  "scene_id": "036bce3393", "top_k": 5 }
```

Output:

```json
{ "status": "ok", "predictions": [
  { "object_id": "obj_0001", "label": "chair",
    "position_3d": [1.2, 0.4, 2.0], "bbox_3d": null,
    "score": 0.91, "evidence": [] } ] }
```

Supported packages provide a `query_object`-style entrypoint that prefers
exact/normalized-label match on `target_label`. The build-cost half of Track 1 is
read from build accounting, not from the entrypoint.

Packages with no object-level memory or no comparable object-location API declare
Track 1 `invalid`.

### Track 2: ScanRefer Referring Query (`track2_scanrefer`)

Input:

```json
{ "query_id": "scanrefer_0001", "dataset": "scanrefer",
  "scene_id": "scene0000_00",
  "utterance": "the red chair next to the table", "top_k": 10 }
```

Output:

```json
{ "status": "ok", "predictions": [
  { "object_id": "obj_0012", "bbox_3d": null, "score": 0.73, "evidence": [] } ] }
```

Methods without a native or package-exported referring-expression resolver declare
Track 2 `invalid`.

### Track 3: OpenEQA General Spatial QA (`track3_openeqa`)

Input:

```json
{ "question_id": "openeqa_0001", "question": "What is next to the sofa?",
  "episode_id": "scannet-v0/002-scannet-scene0709_00" }
```

Output:

```json
{ "status": "ok", "answer": "a coffee table", "evidence": [
  { "source_type": "memory_object", "source_path": "memory/object_table.jsonl",
    "object_id": "obj_0007", "notes": "proximity supports the answer" } ] }
```

Only method-native QA/retrieval APIs count as fixed-API support. A generic
object-table-to-LLM answerer does not make Track 3 supported. OpenEQA covers both
ScanNet and HM3D episodes; `episode_id` carries the dataset prefix.

## Agent Access Policy

The formal agentic setting is per-query LLM tool calling. The evaluator
trace/sandbox contains only: the current prompt, `tool_specs.json`, links/copies to
raw/native memory artifacts, and links/copies to original method source required by
the native tool runtime. It must not expose evaluator adapters, shared-module
adapter code, fixed-API conversion views, benchmark GT/answers, raw frames, or
memory build code. Evidence must be method-exported memory evidence, not a dump of
raw/source frames.

## Required Schema Documentation

`schema.md` must explain: coordinate frame and units; object id format; timestamp
format; relation representation (if any); confidence/score meaning; non-obvious
native artifact formats; known limitations and unsupported tracks. Keep it short.

## Required Build Log

`build_log.json` records how the package was produced (status, timestamps, runtime,
command, config paths, accounting fields, warnings). If export fails, write a build
log with `status: "error"` outside the package result directory and do not mark the
package valid.

## Invalid Results

```json
{ "status": "invalid", "reason_code": "unsupported_fixed_api",
  "required_api": "track3_openeqa", "method": "hovsg",
  "package_path": "memories/hovsg/...",
  "message": "Package does not declare a native OpenEQA fixed API." }
```

For a no-explicit-memory control:

```json
{ "status": "invalid", "reason_code": "control_no_explicit_memory",
  "required_api": "track1_object_location", "method": "multiframe_vlm",
  "control": true, "explicit_memory": false,
  "method_family": "raw_frame_control", "package_path": "memories/multiframe_vlm/...",
  "message": "Control-only: no fixed object-location query API. ..." }
```

`invalid` is a valid benchmark outcome. Runtime failures are `error`, not `invalid`.

## Validation Rules

The validator checks: required files/dirs exist; `manifest.json` and
`capabilities.json` parse; artifact paths are relative and exist when required;
`method.name`, `dataset.name`, `package_id` present; every Track 1-3 capability has
`status`; `supported` fixed APIs have a Python entrypoint; `invalid` fixed APIs have
a reason; `agent_access.write_package` is false; `tool_llm` packages disable
raw-frame / source-crop / fixed-API-view / adapter-code / shared-module-code /
build-code access and enable native memory + method source + native tools;
`not_applicable` packages do not declare native tool access; leakage flags false
unless justified. The validator does not require every method to support every
track.

## Method Family Defaults

| Family | Track 1 | Track 2 | Track 3 |
|---|---|---|---|
| `object_map` | supported if query entrypoint exists | invalid by default | invalid by default |
| `scene_graph` | invalid unless object query exists | invalid by default | invalid unless native QA exists |
| `caption_memory` | invalid by default | invalid by default | supported only if native QA/retrieval API exists |
| `agent_designed` | supported if agent built a query entrypoint | per agent design | per agent design |
| `raw_frame_control` | invalid | invalid | invalid fixed API; agentic only |
| `caption_control` | invalid | invalid | invalid fixed API; agentic/control only |

These are defaults, not hard rules. A package declares support when it provides the
required Python entrypoint and output schema.

## Versioning

Use `schema_version: "0.2"` for the 3-track refactor. Any breaking change to
required fields, track keys, or output schemas should bump the version and update
this file.
