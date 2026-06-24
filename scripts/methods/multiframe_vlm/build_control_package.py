"""Build a Multi-frame VLM *control* package for a scene (tool_llm-runnable).

This is the ReMEmbR `VLMNonAgent` raw-frame control: a no-explicit-memory
baseline that bounds how well a VLM answers from a handful of sampled camera
frames *without* any built object/caption memory. It is NEVER an object-memory
baseline: `manifest.explicit_memory = false`, `method.family =
raw_frame_control`, and all fixed-API tracks are declared `invalid` with
control-only reasons. The validator blocks any `supported` fixed API for this
family.

To make the control runnable through the shared `tool_llm` evaluator (the way
DAAAM/ReMEmbR are scored here), the package exposes its sampled frames via the
native `retrieve_frames` tool (see tool_llm/native_tools.py): the agent VLM
(local Claude CLI, multimodal) is handed the sampled frame image paths + per-
frame pose/time text and answers by reasoning over the raw frames. No object
memory is read because none exists.

Frames are sampled from a prepared RGB layout (e.g. the DAAAM layouts under
data/daaam_layouts/<scene>/<run>/rgb, or any dir of frame images) and referenced
by absolute path under raw_links/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.build_accounting import write_build_log_with_accounting
from spatial_memory_evaluation.memory_package_validator import validate_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Multi-frame VLM control package.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory of frame images (rgb).")
    parser.add_argument("--pose-dir", type=Path, default=None, help="Optional 4x4 pose .txt dir (matches frame stems).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-frames", type=int, default=12, help="How many frames to sample for the control.")
    parser.add_argument("--package-root", type=Path, default=Path("memories"))
    return parser.parse_args()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()


def _read_pose_text(pose_path: Path) -> str:
    try:
        rows = [[float(v) for v in line.split()] for line in pose_path.read_text().splitlines() if line.strip()]
        import math
        x, y, z = rows[0][3], rows[1][3], rows[2][3]
        yaw = math.degrees(math.atan2(rows[1][0], rows[0][0]))
        return f"x={x:.2f} y={y:.2f} z={z:.2f} yaw={yaw:.1f}"
    except Exception:
        return ""


def main(args: argparse.Namespace) -> int:
    started = time.time()
    run_id = args.run_id or f"multiframe-vlm-{args.scene_id}"
    package_dir = args.package_root / "multiframe_vlm" / args.dataset / args.scene_id / run_id
    for sub in ("memory", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / sub).mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(args.frames_dir.glob("*.jpg")) or sorted(args.frames_dir.glob("*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"no frame images in {args.frames_dir}")
    # Evenly sample num_frames across the sequence.
    n = min(args.num_frames, len(frame_paths))
    step = max(1, len(frame_paths) // n)
    sampled = frame_paths[::step][:n]

    # Copy sampled frames into the package (raw_links/frames) so the tool can hand
    # absolute image paths to the multimodal agent.
    frames_out = package_dir / "raw_links" / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i, fp in enumerate(sampled):
        dst = frames_out / f"{i:04d}{fp.suffix}"
        shutil.copy2(fp, dst)
        pose_text = ""
        if args.pose_dir is not None:
            pose_path = args.pose_dir / f"{fp.stem}.txt"
            if pose_path.exists():
                pose_text = _read_pose_text(pose_path)
        rows.append({
            "frame_id": f"{i:04d}",
            "image_path": str(dst.resolve()),
            "source_frame": fp.name,
            "timestamp": float(i),
            "pose_text": pose_text,
        })
    (package_dir / "raw_links" / "sampled_frames.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (package_dir / "memory" / "README.md").write_text(
        "No explicit memory by design (raw_frame_control). The agent answers from "
        "the sampled raw frames in raw_links/sampled_frames.jsonl via the "
        "retrieve_frames native tool.\n", encoding="utf-8"
    )

    _write_schema_md(package_dir, frame_count=len(rows))
    _write_manifest(package_dir, args=args, run_id=run_id, frame_count=len(rows))
    _write_capabilities(package_dir)
    write_build_log_with_accounting(
        package_dir=package_dir,
        native_memory_artifact_paths=[package_dir / "raw_links" / "sampled_frames.jsonl"],
        frame_count=len(rows),
        build_log={
            "status": "ok",
            "started_at": _iso(started),
            "finished_at": _iso(time.time()),
            "build_runtime_seconds": time.time() - started,
            "runtime_seconds": time.time() - started,
            "command": " ".join(sys.argv),
            "frame_count": len(rows),
            "note": "Multi-frame VLM raw-frame control; no object memory built.",
            "warnings": [],
        },
    )
    report = validate_package(package_dir)
    if not report.valid:
        raise RuntimeError(json.dumps(report.to_json(), indent=2))
    print(json.dumps({"status": "ok", "package_dir": str(package_dir),
                      "frame_count": len(rows), "validation": report.to_json()}, indent=2))
    return 0


def _write_schema_md(package_dir: Path, *, frame_count: int) -> None:
    (package_dir / "schema.md").write_text(
        f"""# Multi-frame VLM Control Schema

No-explicit-memory control (`raw_frame_control`), not a spatial-memory baseline.
It bounds how well a multimodal VLM answers from {frame_count} sampled raw camera
frames + per-frame pose/time text, with no object/caption/graph memory built.

Coordinate frame/units: no reconstructed world frame. `pose_text` distances are
meters in the source pose frame; no object is localized in a shared 3D frame.

Object ids: none (no object inventory). Relations/confidence: none.

Native tool: `retrieve_frames` returns sampled frame image paths + pose/time
text; the multimodal agent reasons over the raw frames. Frames live under
`raw_links/frames/` and are indexed by `raw_links/sampled_frames.jsonl`.

Native artifact formats: `raw_links/sampled_frames.jsonl` is UTF-8 JSONL, one
row per sampled frame with `frame_id`, absolute `image_path`, `source_frame`,
`timestamp`, and `pose_text`. `raw_links/frames/<id>.jpg|png` are the sampled
raw RGB frames (the only artifacts; no object table, caption store, vector DB,
or scene graph is produced). The agent reads the image files directly via the
paths returned by `retrieve_frames`.

This control must never be read as a Track 1/2 object-memory baseline; all
fixed-API tracks are `invalid` and `explicit_memory=false`.
""",
        encoding="utf-8",
    )


def _write_manifest(package_dir: Path, *, args: argparse.Namespace, run_id: str, frame_count: int) -> None:
    (package_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "0.2",
        "package_id": f"multiframe_vlm/{args.dataset}/{args.scene_id}/{run_id}",
        "method": {
            "name": "multiframe_vlm",
            "display_name": "Multi-frame VLM (raw-frame control)",
            "family": "raw_frame_control",
            "repo_path": "/home/robin_wang/remembr",
            "commit": None, "version": None,
        },
        "dataset": {"name": args.dataset, "split": "control", "scene_id": args.scene_id, "episode_id": None},
        "input": {
            "modality": ["rgb", "pose", "timestamp"],
            "frame_count": frame_count,
            "rgbd_root": str(args.frames_dir),
            "poses_path": str(args.pose_dir) if args.pose_dir else None,
            "intrinsics_path": None, "timestamp_path": None,
            "coordinate_frame": "raw camera frames; no reconstructed world frame",
        },
        "explicit_memory": False,
        "memory_artifacts": [],
        "evidence_artifacts": [],
        "raw_links": [{
            "name": "sampled_frames", "type": "jsonl", "path": "raw_links/sampled_frames.jsonl",
            "description": f"{frame_count} sampled raw frames + pose/time text for the VLM control.",
            "required_for": [],
        }],
        "tools": [{
            "name": "retrieve_frames", "type": "native", "path": "tool_llm/native_tools.py",
            "description": "Return sampled raw frame image paths + pose/time for the multimodal agent.",
            "required_for": [],
        }],
        "build": {
            "command": " ".join(sys.argv), "config_paths": [], "environment": None,
            "started_at": None, "finished_at": None, "build_runtime_seconds": None,
            "runtime_seconds": None, "frame_count": frame_count, "time_per_frame_seconds": None,
            "native_memory_size_bytes": None, "native_memory_artifacts": [],
            "memory_artifact_size_bytes": None, "package_size_bytes": None,
            "peak_ram_bytes": None, "peak_ram_unavailable_reason": None,
            "peak_vram_bytes": None, "peak_vram_unavailable_reason": None,
        },
        "allowed_access": {
            "contains_gt_annotations": False, "contains_benchmark_answers": False,
            "contains_test_labels": False, "contains_question_specific_rules": False,
        },
        "notes": "Multi-frame VLM raw-frame control; no-explicit-memory; tool_llm via retrieve_frames.",
    }, indent=2), encoding="utf-8")


def _write_capabilities(package_dir: Path) -> None:
    control_reason = (
        "No-explicit-memory raw-frame control (family=raw_frame_control). It builds no "
        "object inventory / location API; it answers only by a VLM reasoning over raw "
        "sampled frames. Scored via the tool_llm path, never the fixed object API."
    )
    (package_dir / "capabilities.json").write_text(json.dumps({
        "schema_version": "0.2",
        "fixed_api": {
            "track1_object_location": {"status": "invalid", "entrypoint": None, "reason": control_reason},
            "track2_scanrefer": {"status": "invalid", "entrypoint": None, "reason": control_reason},
            "track3_openeqa": {"status": "invalid", "entrypoint": None, "reason": control_reason},
        },
        "agent_access": {
            "mode": "tool_llm",
            "read_manifest": True, "read_schema": True, "read_native_memory": True,
            "read_fixed_api_views": False, "read_evidence": True, "read_adapter_code": False,
            "read_shared_module_code": False, "read_method_root_source_code": True,
            "read_build_code": False, "read_raw_links": False, "read_raw_frames": False,
            "read_source_keyframes_or_crops": False, "run_method_native_tools": True,
            "write_package": False,
        },
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
