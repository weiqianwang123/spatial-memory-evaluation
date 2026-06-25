"""Build a ReMEmbR caption-memory package from a posed RGB-D scene.

This is the real ReMEmbR adaptation used by all three tracks: build a caption
memory, then evaluate it with per-query LLM tool calling (the `tool_llm` mode of
the Track 1/2/3 evaluators).

ReMEmbR's native stack (VILA captioner + Milvus + langchain) is not installed in
this environment. Faithful to ReMEmbR's design, we keep its memory shape exactly
- a list of `MemoryItem(caption, time, position, theta)` - and reproduce its
`retrieve_from_text` / `retrieve_from_position` tools (see
`spatial_memory_evaluation/tool_llm/native_tools.py`). The captioner backend is
pluggable:

- `--captioner claude` (default): caption each sampled frame with the local
  Claude CLI (a multimodal VLM), standing in for ReMEmbR's VILA captioner.
- `--captioner none`: write empty captions (positions/time only) - useful for a
  fast geometry-only smoke build.

`position` is the translation of the 4x4 camera->world pose; `theta` is the yaw
extracted from the pose rotation. Frames come from a prepared layout with
`color/<frame>.jpg` and `pose/<frame>.txt` (e.g. the HOV-SG/ScanNet++ layouts
under `data/<...>_layouts/...`).

Examples
--------
Build from a ScanNet++ prepared layout (Track 1)::

    python scripts/methods/remembr/build_memory_package.py \
        --layout-dir data/hovsg_layouts/scannetpp_036bce3393/<run> \
        --dataset scannetpp --scene-id 036bce3393 \
        --captioner claude --max-frames 24
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.build_accounting import (  # noqa: E402
    write_build_log_with_accounting,
)
from spatial_memory_evaluation.common.jsonl import write_json, write_jsonl  # noqa: E402
from spatial_memory_evaluation.memory_package_validator import validate_package  # noqa: E402

REMEMBR_REPO = Path("/home/robin_wang/remembr")

CAPTION_PROMPT = (
    "Read the image file {image_rel} and reply with a single concise sentence "
    "describing the indoor scene and the salient objects visible (furniture, "
    "appliances, equipment). Output only the caption sentence, no preamble."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a ReMEmbR caption-memory package from a posed RGB-D scene.")
    parser.add_argument("--layout-dir", type=Path, required=True, help="Prepared layout with color/ and pose/.")
    parser.add_argument("--dataset", default="scannetpp")
    parser.add_argument("--split", default="current-scene")
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--captioner", choices=("ollama", "claude", "none"), default="ollama")
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:4b",
        help="Local Ollama multimodal model for captioning (native VILA substitute; "
        "vision-capable, ~VILA-3B scale). Used when --captioner ollama.",
    )
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument(
        "--caption-command",
        default=(
            "cd {layout_dir} && claude -p {prompt_q} --output-format text "
            "--permission-mode bypassPermissions"
        ),
        help="Caption command template. Placeholders: {layout_dir}, {image_rel}, {prompt_q}.",
    )
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1, help="Stride over the layout's sorted frames.")
    parser.add_argument("--caption-timeout", type=int, default=180)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-path", type=Path, default=REMEMBR_REPO)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    layout_dir = args.layout_dir.resolve()
    color_dir = layout_dir / "color"
    pose_dir = layout_dir / "pose"
    if not color_dir.is_dir() or not pose_dir.is_dir():
        raise FileNotFoundError(f"layout must contain color/ and pose/: {layout_dir}")

    frames = _select_frames(color_dir, pose_dir, stride=args.frame_stride, max_frames=args.max_frames)
    if not frames:
        raise ValueError(f"no color/pose frame pairs found under {layout_dir}")

    scene_or_episode = args.scene_id or args.episode_id or layout_dir.name
    run_id = args.run_id or f"remembr-{_safe(scene_or_episode)}-{time.strftime('%Y%m%d-%H%M%S')}"
    package_dir = args.output_dir or (
        REPO_ROOT / "memories" / "remembr" / args.dataset / scene_or_episode / run_id
    )
    package_dir = Path(package_dir)
    _reset_package_dirs(package_dir)

    started_epoch = time.time()
    caption_rows: list[dict[str, Any]] = []
    for index, (frame_id, image_path, pose_path) in enumerate(frames):
        position, theta = _pose_position_yaw(pose_path)
        caption = ""
        if args.captioner == "ollama":
            caption = _caption_with_ollama(
                image_path=image_path,
                model=args.ollama_model,
                endpoint=args.ollama_endpoint,
                timeout=args.caption_timeout,
            )
        elif args.captioner == "claude":
            caption = _caption_with_command(
                caption_command=args.caption_command,
                layout_dir=layout_dir,
                image_rel=f"color/{image_path.name}",
                timeout=args.caption_timeout,
            )
        caption_rows.append(
            {
                "caption_id": frame_id,
                "caption": caption,
                "time": float(index),  # frame index as a monotonic timestamp (seconds)
                "position": position,
                "theta": theta,
                "file_start": image_path.name,
                "file_end": image_path.name,
            }
        )
        print(f"[{index + 1}/{len(frames)}] {frame_id}: {caption[:80]}", flush=True)

    write_jsonl(package_dir / "memory" / "captions.jsonl", caption_rows)
    # Native ReMEmbR-shaped copy (MemoryItem fields only).
    native_rows = [
        {"caption": r["caption"], "time": r["time"], "position": r["position"], "theta": r["theta"]}
        for r in caption_rows
    ]
    write_jsonl(package_dir / "memory" / "native" / "remembr_memory.jsonl", native_rows)

    _write_schemas(package_dir)
    _write_docs(package_dir)
    _write_schema_md(package_dir, caption_count=len(caption_rows), captioner=args.captioner)
    _write_manifest(
        package_dir=package_dir,
        args=args,
        run_id=run_id,
        scene_or_episode=scene_or_episode,
        caption_count=len(caption_rows),
        layout_dir=layout_dir,
    )
    _write_capabilities(package_dir)

    finished_epoch = time.time()
    write_build_log_with_accounting(
        package_dir=package_dir,
        native_memory_artifact_paths=[
            package_dir / "memory" / "captions.jsonl",
            package_dir / "memory" / "native" / "remembr_memory.jsonl",
        ],
        frame_count=len(caption_rows),
        build_log={
            "status": "ok",
            "started_at": _iso(started_epoch),
            "finished_at": _iso(finished_epoch),
            "build_runtime_seconds": max(0.0, finished_epoch - started_epoch),
            "command": " ".join(sys.argv),
            "config_paths": [],
            "source_outputs": [str(layout_dir)],
            "caption_count": len(caption_rows),
            "captioner": args.captioner,
            "explicit_memory": True,
            "fixed_api": {
                "track1_object_location": "invalid",
                "track2_scanrefer": "invalid",
                "track3_openeqa": "invalid",
            },
            "agent_access_mode": "tool_llm",
            "warnings": [],
        },
    )

    report = validate_package(package_dir)
    print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    if not report.valid:
        return 1
    print(f"ReMEmbR caption-memory package written to {package_dir}")
    return 0


def _select_frames(
    color_dir: Path,
    pose_dir: Path,
    *,
    stride: int,
    max_frames: int,
) -> list[tuple[str, Path, Path]]:
    color_frames = sorted(color_dir.glob("*.jpg")) + sorted(color_dir.glob("*.png"))
    color_frames = sorted(set(color_frames))
    pairs: list[tuple[str, Path, Path]] = []
    for image_path in color_frames:
        frame_id = image_path.stem
        pose_path = pose_dir / f"{frame_id}.txt"
        if pose_path.exists():
            pairs.append((frame_id, image_path, pose_path))
    if stride > 1:
        pairs = pairs[::stride]
    if max_frames and max_frames > 0:
        pairs = pairs[:max_frames]
    return pairs


def _pose_position_yaw(pose_path: Path) -> tuple[list[float], float]:
    """Return (position [x,y,z], yaw theta) from a 4x4 camera->world pose."""

    rows = []
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 4:
            rows.append([float(v) for v in values[:4]])
    if len(rows) < 3:
        return [0.0, 0.0, 0.0], 0.0
    position = [rows[0][3], rows[1][3], rows[2][3]]
    # Yaw from the rotation's forward axis projected on the XY plane.
    yaw = math.atan2(rows[1][0], rows[0][0])
    if not all(math.isfinite(v) for v in position) or not math.isfinite(yaw):
        return [0.0, 0.0, 0.0], 0.0
    return [round(v, 6) for v in position], round(yaw, 6)


def _caption_with_ollama(*, image_path: Path, model: str, endpoint: str, timeout: int) -> str:
    """Caption one frame with a local Ollama multimodal model (qwen3.5:4b).

    This is the native-style captioner substitute for ReMEmbR's VILA (which is not
    installed): a small local vision-language model, no Claude. Posts the JPEG as
    base64 to Ollama's /api/chat. Strips any <think> reasoning block.
    """
    import base64
    import json as _json
    import re as _re
    import urllib.request

    try:
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": CAPTION_PROMPT.format(image_rel=image_path.name) + " /no_think",
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/api/chat",
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
        text = (data.get("message", {}) or {}).get("content", "") or ""
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S)
        return " ".join(text.strip().splitlines()).strip()
    except Exception:
        return ""


def _caption_with_command(*, caption_command: str, layout_dir: Path, image_rel: str, timeout: int) -> str:
    import shlex

    prompt = CAPTION_PROMPT.format(image_rel=image_rel)
    command = caption_command
    for key, value in {
        "layout_dir": str(layout_dir),
        "image_rel": image_rel,
        "prompt_q": shlex.quote(prompt),
    }.items():
        command = command.replace("{" + key + "}", value)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""
    if proc.returncode != 0:
        return ""
    return " ".join(proc.stdout.strip().splitlines()).strip()


def _reset_package_dirs(package_dir: Path) -> None:
    for name in ("memory", "memory/native", "evidence", "raw_links", "schemas", "tools"):
        (package_dir / name).mkdir(parents=True, exist_ok=True)


def _write_schemas(package_dir: Path) -> None:
    write_json(
        package_dir / "schemas" / "captions.schema.json",
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "ReMEmbR caption memory row",
            "type": "object",
            "required": ["caption_id", "caption", "time", "position", "theta"],
            "properties": {
                "caption_id": {"type": "string"},
                "caption": {"type": "string"},
                "time": {"type": ["number", "null"]},
                "position": {"type": ["array", "null"], "items": {"type": "number"}},
                "theta": {"type": ["number", "null"]},
                "file_start": {"type": ["string", "null"]},
                "file_end": {"type": ["string", "null"]},
            },
        },
    )


def _write_docs(package_dir: Path) -> None:
    package_dir.joinpath("evidence", "README.md").write_text(
        "# Evidence\n\nReMEmbR caption memory exposes caption text, pose, and time as\n"
        "provenance. There are no object crops; the method has no object inventory.\n",
        encoding="utf-8",
    )
    package_dir.joinpath("raw_links", "README.md").write_text(
        "# Raw Links\n\nRaw frames are not linked. Captions are derived from frames\n"
        "by the captioner; raw-frame access stays a separate ablation.\n",
        encoding="utf-8",
    )
    package_dir.joinpath("tools", "README.md").write_text(
        "# Tools\n\nNo fixed-API Python entrypoints. ReMEmbR is evaluated via the\n"
        "`tool_llm` path: the LLM calls the native `retrieve_from_text` /\n"
        "`retrieve_from_position` tools (reproduced in\n"
        "`spatial_memory_evaluation/tool_llm/native_tools.py`) over\n"
        "`memory/captions.jsonl`.\n",
        encoding="utf-8",
    )


def _write_schema_md(package_dir: Path, *, caption_count: int, captioner: str) -> None:
    package_dir.joinpath("schema.md").write_text(
        f"""# ReMEmbR Caption Memory Schema

Coordinate frame and units: `position` is the camera->world translation (meters)
from the scene's 4x4 poses. `theta` is the yaw (radians) extracted from the pose
rotation. Distances are in meters.

Object id format: not applicable. ReMEmbR has no object-level memory and no object
ids. Each row is a caption window keyed by `caption_id` (the source frame id).

Timestamp format: `time` is a float (seconds). For an offline posed-RGBD scene we
use the sampled frame index as a monotonic timestamp.

Relation representation: none. Caption memory stores free text per window with no
object nodes or relation edges.

Confidence or score meaning: none natively. The `tool_llm` retrieval tools attach
a lexical/positional similarity score at query time for ranking only.

Native artifact formats: `memory/captions.jsonl` is UTF-8 JSONL with one caption
window per line (`caption_id`, `caption`, `time`, `position`, `theta`).
`memory/native/remembr_memory.jsonl` is the ReMEmbR `MemoryItem` shape
(`caption`, `time`, `position`, `theta`) - see remembr/memory/memory.py:5-9. This
package has {caption_count} caption window(s); captioner = `{captioner}`.

Known limitations and unsupported tracks: ReMEmbR has no deterministic native
fixed API for object location, referring, or QA, so all fixed-API tracks are
`invalid`. It is evaluated through the agentic `tool_llm` path, where an LLM calls
the native `retrieve_from_text` / `retrieve_from_position` tools over caption
memory and produces the answer - this is ReMEmbR's own query mechanism.
""",
        encoding="utf-8",
    )


def _write_manifest(
    *,
    package_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    scene_or_episode: str,
    caption_count: int,
    layout_dir: Path,
) -> None:
    write_json(
        package_dir / "manifest.json",
        {
            "schema_version": "0.2",
            "package_id": f"remembr/{args.dataset}/{scene_or_episode}/{run_id}",
            "method": {
                "name": "remembr",
                "display_name": "ReMEmbR (caption memory + LLM tool calling)",
                "family": "caption_memory",
                "repo_path": str(args.repo_path),
                "commit": None,
                "version": None,
            },
            "dataset": {
                "name": args.dataset,
                "split": args.split,
                "scene_id": args.scene_id,
                "episode_id": args.episode_id,
            },
            "input": {
                "modality": ["rgb", "pose", "timestamp"],
                "frame_count": caption_count,
                "rgbd_root": str(layout_dir),
                "poses_path": str(layout_dir / "pose"),
                "intrinsics_path": str(layout_dir / "intrinsic"),
                "timestamp_path": None,
                "coordinate_frame": "scene camera->world frame; meters; theta is yaw in radians",
            },
            "vocabulary": {
                "vocabulary_mode": "module_ablation",
                "class_list_path": None,
                "source": "free-text VLM captions; no detector vocabulary",
                "captioner": args.captioner,
            },
            "modules": {
                "captioner": (
                    f"{args.ollama_model} via ollama (local VLM, stands in for ReMEmbR VILA captioner)"
                    if args.captioner == "ollama"
                    else f"{args.captioner} (stands in for ReMEmbR VILA captioner)"
                ),
                "retrieval_tools": "retrieve_from_text / retrieve_from_position over caption memory",
                "native_memory": "remembr.memory.MemoryItem (caption/time/position/theta)",
            },
            "explicit_memory": True,
            "memory_artifacts": [
                {
                    "name": "captions",
                    "type": "jsonl",
                    "path": "memory/captions.jsonl",
                    "description": f"ReMEmbR caption memory: {caption_count} caption window(s) with text, pose, time.",
                    "required_for": [],
                },
                {
                    "name": "native_memory",
                    "type": "jsonl",
                    "path": "memory/native/remembr_memory.jsonl",
                    "description": "Native ReMEmbR MemoryItem rows (caption/time/position/theta).",
                    "required_for": [],
                },
            ],
            "evidence_artifacts": [],
            "raw_links": [],
            "tools": [],
            "build": {
                "command": " ".join(sys.argv),
                "config_paths": [],
                "environment": None,
                "started_at": None,
                "finished_at": None,
                "build_runtime_seconds": None,
                "runtime_seconds": None,
                "frame_count": caption_count,
                "time_per_frame_seconds": None,
                "native_memory_size_bytes": None,
                "native_memory_artifacts": [],
                "memory_artifact_size_bytes": None,
                "package_size_bytes": None,
                "peak_ram_bytes": None,
                "peak_ram_unavailable_reason": None,
                "peak_vram_bytes": None,
                "peak_vram_unavailable_reason": None,
            },
            "allowed_access": {
                "contains_gt_annotations": False,
                "contains_benchmark_answers": False,
                "contains_test_labels": False,
                "contains_question_specific_rules": False,
            },
            "notes": (
                "ReMEmbR caption-memory package built from a posed RGB-D scene. Captioner "
                "stands in for ReMEmbR's VILA; native MemoryItem shape preserved. Evaluated "
                "via the tool_llm path (retrieve_from_text / retrieve_from_position). The "
                "external ReMEmbR repo was read-only."
            ),
        },
    )


def _write_capabilities(package_dir: Path) -> None:
    invalid_reason = (
        "ReMEmbR exposes no deterministic native fixed API; its query mechanism is the "
        "agentic ReMEmbRAgent tool loop. Evaluate via the tool_llm path."
    )
    write_json(
        package_dir / "capabilities.json",
        {
            "schema_version": "0.2",
            "fixed_api": {
                "track1_object_location": {"status": "invalid", "entrypoint": None, "reason": invalid_reason},
                "track2_scanrefer": {"status": "invalid", "entrypoint": None, "reason": invalid_reason},
                "track3_openeqa": {"status": "invalid", "entrypoint": None, "reason": invalid_reason},
            },
            "agent_access": {
                "mode": "tool_llm",
                "read_manifest": True,
                "read_schema": True,
                "read_native_memory": True,
                "read_fixed_api_views": False,
                "read_evidence": True,
                "read_adapter_code": False,
                "read_shared_module_code": False,
                "read_method_root_source_code": True,
                "read_build_code": False,
                "read_raw_links": False,
                "read_raw_frames": False,
                "read_source_keyframes_or_crops": False,
                "run_method_native_tools": True,
                "write_package": False,
            },
        },
    )


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(epoch))


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
