# Minimal Memory Package Spec

Last updated: 2026-06-15

This spec fixes the first-version package format used by all fixed-API and
agentic evaluators. Every method or control baseline must export one minimal
memory package before it can be evaluated.

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

## Goals

- Give every method a common package boundary while preserving its native memory
  format.
- Make supported and unsupported fixed APIs explicit through `capabilities.json`.
- Let evaluators produce `invalid` for unsupported tracks instead of guessing,
  silently falling back, or returning empty predictions.
- Give agentic evaluation a self-contained folder it can copy into a sandbox.

## Package Path

Packages live under:

```text
memories/<method>/<dataset>/<scene-or-episode>/<run-id>/
```

Examples:

```text
memories/claws/scannetpp/036bce3393/20260615-153000/
memories/dualmap/openeqa/scene0709_00/20260615-153000/
memories/remembr/oc-navqa/sequence_0/20260615-153000/
```

`run-id` should normally be a timestamp in `YYYYMMDD-HHMMSS` format. It may be a
stable hash only if the package is deterministic and already documented.

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

Recommended:

```text
README.md
checksums.json
environment.txt
```

The package must be self-describing enough that a fixed evaluator or sandboxed
agent can understand what artifacts exist, what they mean, and which tracks are
supported without inspecting the original method repo.

## Manifest

`manifest.json` describes the method, dataset, inputs, artifacts, and leakage
constraints. Required top-level fields:

```json
{
  "schema_version": "0.1",
  "package_id": "claws/scannetpp/036bce3393/20260615-153000",
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
    "runtime_seconds": null,
    "memory_size_bytes": null
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

### Method Family

Use one of:

- `object_map`
- `scene_graph`
- `caption_memory`
- `vector_db`
- `raw_frame_control`
- `caption_control`
- `other`

Controls such as Multi-frame VLM and LLM-with-captions must set
`explicit_memory` to `false`.

### Artifact Records

Each artifact record in `memory_artifacts`, `evidence_artifacts`, `raw_links`,
or `tools` uses package-relative paths:

```json
{
  "name": "object_table",
  "type": "jsonl",
  "path": "memory/object_table.jsonl",
  "description": "Canonical object inventory exported from native memory.",
  "required_for": ["track1_memory_construction", "track2_object_location"]
}
```

Do not store absolute artifact paths unless they are external raw-data links in
`raw_links`. If an absolute path is included, it must be read-only input data and
must not contain GT answers.

## Capabilities

`capabilities.json` is the evaluator contract. It says what the fixed API can
and cannot do, and what the agent may access.

Required top-level fields:

```json
{
  "schema_version": "0.1",
  "fixed_api": {
    "track1_memory_construction": {
      "status": "supported",
      "entrypoint": "tools/list_objects.py:list_objects",
      "reason": "",
      "input_schema": "schemas/track1_input.schema.json",
      "output_schema": "schemas/object_table.schema.json"
    },
    "track2_object_location": {
      "status": "invalid",
      "entrypoint": null,
      "reason": "No native object-location query API is exported."
    },
    "track3_scanrefer": {
      "status": "invalid",
      "entrypoint": null,
      "reason": "No referring-expression resolver is exported."
    },
    "track4_openeqa": {
      "status": "invalid",
      "entrypoint": null,
      "reason": "No native QA or retrieval API is exported."
    }
  },
  "agent_access": {
    "mode": "agentic_full_access",
    "read_manifest": true,
    "read_schema": true,
    "read_memory_artifacts": true,
    "read_evidence": true,
    "read_adapter_code": true,
    "read_shared_module_code": true,
    "read_method_root_source_code": true,
    "read_raw_links": false,
    "read_raw_frames": false,
    "read_source_keyframes_or_crops": false,
    "run_package_tools": false,
    "write_package": false
  }
}
```

### Fixed API Status

Allowed values:

- `supported`: evaluator may call the Python entrypoint.
- `invalid`: method/package does not support this track by design.

Do not use `partial` in `capabilities.json`. If a track is not ready for the
fixed evaluator, mark it `invalid` and explain why.

### Python Entrypoint

First-version fixed APIs use Python entrypoints only:

```text
relative/path.py:function_name
```

Rules:

- Path is relative to the package root.
- Function must be importable without changing cwd outside the package.
- Function must not write into the package.
- Function may read package artifacts.
- Function receives JSON-serializable input and returns JSON-serializable output.

Recommended function signatures:

```python
def list_objects(package_dir: str, query: dict) -> dict: ...
def query_object(package_dir: str, query: dict) -> dict: ...
def resolve_referring_expression(package_dir: str, query: dict) -> dict: ...
def answer_question(package_dir: str, query: dict) -> dict: ...
```

The evaluator will pass `package_dir` as the copied package path or original
package path depending on mode.

## Track Contracts

### Track 1: Memory Construction / Object Inventory

Capability key:

```text
fixed_api.track1_memory_construction
```

Supported packages must provide either:

- a Python entrypoint that returns an object table; or
- a Python entrypoint that reads `memory/object_table.jsonl` and returns it.

Canonical object output:

```json
{
  "status": "ok",
  "objects": [
    {
      "object_id": "obj_0001",
      "label": "chair",
      "aliases": ["seat"],
      "position_3d": [1.2, 0.4, 2.0],
      "bbox_3d": null,
      "confidence": 0.82,
      "source_artifacts": ["memory/object_table.jsonl"],
      "evidence": [
        {
          "source_type": "crop",
          "source_path": "evidence/crops/obj_0001.jpg",
          "frame_id": "000123",
          "notes": "object crop used by the method"
        }
      ]
    }
  ]
}
```

DSG/caption methods may declare Track 1 `invalid` if they cannot honestly export
object-level inventory.

### Track 2: Basic Object Location Query

Capability key:

```text
fixed_api.track2_object_location
```

Input:

```json
{
  "query_id": "q_0001",
  "query": "where is the chair?",
  "scene_id": "036bce3393",
  "top_k": 5
}
```

Output:

```json
{
  "status": "ok",
  "predictions": [
    {
      "object_id": "obj_0001",
      "label": "chair",
      "position_3d": [1.2, 0.4, 2.0],
      "score": 0.91,
      "evidence": []
    }
  ]
}
```

### Track 3: ScanRefer Referring Query

Capability key:

```text
fixed_api.track3_scanrefer
```

Input:

```json
{
  "query_id": "scanrefer_0001",
  "dataset": "scanrefer",
  "scene_id": "scene0000_00",
  "utterance": "the red chair next to the table",
  "top_k": 10
}
```

Output:

```json
{
  "status": "ok",
  "predictions": [
    {
      "object_id": "obj_0012",
      "bbox_3d": null,
      "score": 0.73,
      "evidence": []
    }
  ]
}
```

Methods without a native or package-exported referring-expression resolver should
declare Track 3 `invalid`.

### Track 4: OpenEQA General Spatial QA

Capability key:

```text
fixed_api.track4_openeqa
```

Input:

```json
{
  "question_id": "openeqa_0001",
  "question": "What is next to the sofa?",
  "episode_id": "scannet-v0/002-scannet-scene0709_00"
}
```

Output:

```json
{
  "status": "ok",
  "answer": "a coffee table",
  "evidence": [
    {
      "source_type": "memory_object",
      "source_path": "memory/object_table.jsonl",
      "object_id": "obj_0007",
      "notes": "object relation or proximity supports the answer"
    }
  ]
}
```

Only method-native QA/retrieval APIs count as fixed API support. A generic
object-table-to-LLM answerer does not make Track 4 supported.

## Agent Access Policy

The default agentic setting is package-plus-source-code access. The sandbox
receives the memory package, evaluation adapter code, shared module code, and
the original method root repo source code recorded in `manifest.method.repo_path`.
The agent may design temporary parsers, query scripts, or small interfaces to
interact with the memory, rather than being limited to fixed APIs.

Raw/source frames remain disabled by default:

```json
{
  "mode": "agentic_full_access",
  "read_manifest": true,
  "read_schema": true,
  "read_memory_artifacts": true,
  "read_evidence": true,
  "read_adapter_code": true,
  "read_shared_module_code": true,
  "read_method_root_source_code": true,
  "read_raw_links": false,
  "read_raw_frames": false,
  "read_source_keyframes_or_crops": false,
  "write_package": false
}
```

Agentic modes:

- `agentic_full_access`: agent may read manifest, schema, memory artifacts,
  evidence, package docs, package tools, evaluation adapters, shared module
  code, and original method root source code. It may create temporary scripts in
  the sandbox to parse/query memory artifacts. This is the default agentic mode.
- `agentic_memory_only`: legacy/ablation mode where the agent reads only the
  memory package and package docs.
- `memory_plus_crops`: agent may additionally read package-local sampled
  keyframes or source crops.
- `memory_plus_raw`: agent may additionally read raw RGB-D links if allowed.

Source frames, sampled keyframes, and source crops remain ablations. Evidence
must be method-exported memory evidence, such as object crops, thumbnails,
relation visualizations, or retrieval traces. It must not be a general dump of
raw/source frames.

The implementation copies the memory package and source-code context into the
sandbox. The agent works on copies and must not modify the source package.
Source-code context copies should exclude `.git`, generated data, checkpoints,
caches, memories, and result artifacts, but keep original method source,
configs, scripts, schema, README, and docs.

## Required Schema Documentation

`schema.md` must explain:

- coordinate frame and units;
- object id format;
- timestamp format;
- relation representation, if any;
- confidence/score meaning;
- native artifact formats that are not obvious;
- known limitations and unsupported tracks.

Keep it short and practical. The goal is to let a fixed evaluator or sandboxed
agent read it quickly.

## Required Build Log

`build_log.json` records how the package was produced:

```json
{
  "status": "ok",
  "started_at": "2026-06-15T15:30:00+08:00",
  "finished_at": "2026-06-15T15:45:00+08:00",
  "runtime_seconds": 900.0,
  "command": "python scripts/export_memory_package.py ...",
  "config_paths": ["configs/claws_current_scene_method_kwargs.json"],
  "source_outputs": [],
  "warnings": []
}
```

If export fails, write a build log with `status: "error"` outside the package
result directory and do not mark the package valid.

## Invalid Results

Fixed evaluators must write an invalid result when a package declares a track
unsupported:

```json
{
  "status": "invalid",
  "reason_code": "unsupported_fixed_api",
  "required_api": "track4_openeqa.answer_question",
  "method": "dualmap",
  "package_path": "memories/dualmap/openeqa/scene0709_00/20260615-153000",
  "message": "Package does not declare a native OpenEQA fixed API."
}
```

`invalid` is a valid benchmark outcome. It means the method does not expose that
fixed API. Runtime failures are `error`, not `invalid`.

## Validation Rules

The first validator should check:

- required files exist;
- `manifest.json` and `capabilities.json` parse as JSON;
- package paths in artifacts are relative and exist when required;
- `method.name`, `dataset.name`, and `package_id` are present;
- every Track 1-4 capability has `status`;
- `supported` fixed APIs have a non-empty Python entrypoint;
- `invalid` fixed APIs have a non-empty reason;
- `agent_access.write_package` is false;
- `agentic_full_access` and `memory_only` packages have raw-frame and
  source-keyframe/crop access disabled;
- `agentic_full_access` packages declare adapter, shared module, and method root
  source-code access;
- `allowed_access` leakage flags are false unless explicitly justified.

The validator should not require every method to support every track.

## Method Family Defaults

Suggested default capabilities before method-specific overrides:

| Family | Track 1 | Track 2 | Track 3 | Track 4 |
|---|---|---|---|---|
| `object_map` | supported if object table exists | supported if query entrypoint exists | invalid by default | invalid by default |
| `scene_graph` | invalid unless object export exists | invalid unless object query exists | invalid by default | invalid unless native QA exists |
| `caption_memory` | invalid by default | invalid by default | invalid by default | supported only if native QA/retrieval API exists |
| `raw_frame_control` | invalid | invalid | invalid | invalid fixed API; agentic only |
| `caption_control` | invalid | invalid | invalid | invalid fixed API; agentic/control only |

These are defaults, not hard rules. A package can declare support when it
provides the required Python entrypoint and output schema.

## Versioning

Use `schema_version: "0.1"` for the first implementation. Any breaking change to
required fields or output schemas should bump the version and update this file.
