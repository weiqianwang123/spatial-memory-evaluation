#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = Path(".claude/session_logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Claude task with local session metadata/logging.")
    parser.add_argument(
        "task",
        help="task id like 01 or task_01, or path to .claude/tasks/task_*.md",
    )
    parser.add_argument("--worktree", type=Path, default=Path.cwd(), help="worktree to run Claude in")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="main repo root containing .claude/tasks")
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--model-env", default="CLAUDE_CODE_USE_BEDROCK", help="env var used to enable Bedrock")
    parser.add_argument("--prompt-extra", default="", help="extra instruction appended to the generated prompt")
    parser.add_argument("--background", action="store_true", help="return immediately after starting Claude")
    parser.add_argument("--no-commit-request", action="store_true", help="do not ask Claude to commit changes")
    parser.add_argument("--dry-run", action="store_true", help="resolve task/session paths but do not start Claude")
    parser.add_argument("--debug", action="store_true", help="print launch diagnostics and include them in metadata")
    parser.add_argument(
        "--stream-json",
        action="store_true",
        help="use Claude stream-json output and write a readable activity log",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    worktree = args.worktree.resolve()
    repo_root = args.repo_root.resolve()
    task_path = resolve_task(args.task, repo_root, worktree)
    branch = git_value(worktree, ["branch", "--show-current"]) or "(detached)"
    session_id = f"{timestamp()}_{safe_name(branch)}_{safe_name(task_path.stem)}"
    log_dir = worktree / SESSION_DIR
    stdout_log = log_dir / f"{session_id}.stdout.log"
    stderr_log = log_dir / f"{session_id}.stderr.log"
    stream_json_log = log_dir / f"{session_id}.stream.jsonl" if args.stream_json else None
    metadata_path = log_dir / f"{session_id}.json"
    prompt = build_prompt(task_path, commit_request=not args.no_commit_request, extra=args.prompt_extra)
    command = [
        "claude",
        "-p",
        prompt,
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "stream-json" if args.stream_json else "text",
        "--max-budget-usd",
        str(args.budget),
    ]
    if args.stream_json:
        command.extend(["--include-partial-messages", "--verbose"])
    env = os.environ.copy()
    env[args.model_env] = "1"
    env["CODE_USE_BEDROCK"] = "1"
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    env["AWS_REGION"] = args.region
    metadata = {
        "session_id": session_id,
        "status": "starting",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "worktree": str(worktree),
        "branch": branch,
        "task_path": str(task_path),
        "task_title": task_title(task_path),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "stream_json": args.stream_json,
        "stream_json_log": str(stream_json_log) if stream_json_log else None,
        "summary": f"Running {task_title(task_path) or task_path.name}",
        "command": redact_command(command),
    }
    if args.debug:
        metadata["debug"] = {
            "claude_path": shutil.which("claude"),
            "cwd_exists": worktree.exists(),
            "bedrock_env": {
                "CODE_USE_BEDROCK": env.get("CODE_USE_BEDROCK"),
                "CLAUDE_CODE_USE_BEDROCK": env.get("CLAUDE_CODE_USE_BEDROCK"),
                "AWS_REGION": env.get("AWS_REGION"),
            },
        }
    if args.dry_run:
        metadata["status"] = "dry_run"
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    if args.background:
        try:
            stdout_target = stream_json_log if stream_json_log else stdout_log
            assert stdout_target is not None
            with stdout_target.open("ab") as out, stderr_log.open("ab") as err:
                proc = subprocess.Popen(command, cwd=worktree, env=env, stdout=out, stderr=err, start_new_session=True)
        except OSError as exc:
            metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            metadata["status"] = "failed"
            metadata["summary"] = f"failed to launch Claude: {exc}"
            write_json(metadata_path, metadata)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            return 1
        metadata["pid"] = proc.pid
        metadata["status"] = "running"
        write_json(metadata_path, metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0

    raw_stream = None
    try:
        if stream_json_log is not None:
            raw_stream = stream_json_log.open("a", encoding="utf-8")
        with stdout_log.open("a", encoding="utf-8") as out, stderr_log.open("a", encoding="utf-8") as err:
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=worktree,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                metadata["status"] = "failed"
                metadata["summary"] = f"failed to launch Claude: {exc}"
                write_json(metadata_path, metadata)
                print(json.dumps(metadata, indent=2, sort_keys=True))
                return 1
            metadata["pid"] = proc.pid
            metadata["status"] = "running"
            write_json(metadata_path, metadata)
            print(json.dumps(metadata, indent=2, sort_keys=True))
            if args.stream_json:
                assert raw_stream is not None
                stdout_target = tee_stream_json
                stdout_args = (proc.stdout, raw_stream, out, sys.stdout)
            else:
                stdout_target = tee_stream
                stdout_args = (proc.stdout, out, sys.stdout)
            threads = [
                threading.Thread(target=stdout_target, args=stdout_args, daemon=True),
                threading.Thread(target=tee_stream, args=(proc.stderr, err, sys.stderr), daemon=True),
            ]
            for thread in threads:
                thread.start()
            try:
                return_code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                metadata["status"] = "interrupted"
                metadata["summary"] = "Interrupted by user"
                write_json(metadata_path, metadata)
                raise
            for thread in threads:
                thread.join(timeout=2)
    finally:
        if raw_stream is not None:
            raw_stream.close()
    metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata["return_code"] = return_code
    metadata["status"] = "complete" if return_code == 0 else "failed"
    metadata["summary"] = f"Claude exited with code {return_code}"
    write_json(metadata_path, metadata)
    print()
    print(metadata["summary"])
    print(f"stdout log: {stdout_log}")
    print(f"stderr log: {stderr_log}")
    print(f"session json: {metadata_path}")
    return return_code


def tee_stream(source: TextIO | None, log_file: TextIO, target: TextIO) -> None:
    if source is None:
        return
    for line in source:
        log_file.write(line)
        log_file.flush()
        target.write(line)
        target.flush()


def tee_stream_json(source: TextIO | None, raw_log: TextIO, activity_log: TextIO, target: TextIO) -> None:
    if source is None:
        return
    for line in source:
        raw_log.write(line)
        raw_log.flush()
        summary = summarize_stream_json_line(line)
        if not summary:
            continue
        rendered = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {summary}\n"
        activity_log.write(rendered)
        activity_log.flush()
        target.write(rendered)
        target.flush()


def summarize_stream_json_line(line: str) -> str | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        text = line.strip()
        return text[:500] if text else None
    if not isinstance(value, dict):
        return None
    event_type = str(value.get("type") or "")
    subtype = str(value.get("subtype") or "")
    if event_type == "system":
        if subtype == "init":
            cwd = value.get("cwd")
            model = value.get("model")
            parts = ["started Claude stream"]
            if model:
                parts.append(f"model={model}")
            if cwd:
                parts.append(f"cwd={cwd}")
            return " | ".join(parts)
        if subtype == "status":
            status = value.get("status")
            return f"status: {status}" if status else "status update"
        return f"system event: {subtype or event_type}"
    if event_type == "stream_event":
        return summarize_stream_event(value.get("event"))
    if event_type == "assistant":
        return summarize_assistant_event(value)
    if event_type == "user":
        return summarize_user_event(value)
    if event_type == "result":
        status = subtype or ("error" if value.get("is_error") else "success")
        duration = value.get("duration_ms")
        cost = value.get("total_cost_usd")
        parts = [f"result: {status}"]
        if duration is not None:
            parts.append(f"{float(duration) / 1000:.1f}s")
        if cost is not None:
            parts.append(f"${float(cost):.4f}")
        result_text = value.get("result")
        if isinstance(result_text, str) and result_text.strip():
            parts.append(excerpt(result_text, 220))
        return " | ".join(parts)
    if event_type == "error":
        return f"error: {excerpt(str(value.get('message') or value), 400)}"
    return summarize_generic_event(value, event_type, subtype)


def summarize_stream_event(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "")
    if event_type == "content_block_start":
        block = event.get("content_block")
        if isinstance(block, dict):
            return summarize_content_block(block)
    if event_type == "content_block_delta":
        delta = event.get("delta")
        if isinstance(delta, dict):
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text.strip():
                    return f"assistant: {excerpt(text, 180)}"
            if delta_type == "input_json_delta":
                return None
    if event_type == "message_delta":
        stop_reason = event.get("delta", {}).get("stop_reason") if isinstance(event.get("delta"), dict) else None
        return f"message: {stop_reason}" if stop_reason else None
    if event_type == "message_stop":
        metrics = event.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            latency = metrics.get("invocationLatency")
            output_tokens = metrics.get("outputTokenCount")
            parts = ["model response complete"]
            if latency is not None:
                parts.append(f"{float(latency) / 1000:.1f}s")
            if output_tokens is not None:
                parts.append(f"output_tokens={output_tokens}")
            return " | ".join(parts)
        return "model response complete"
    if event_type in {"content_block_stop", "message_start"}:
        return None
    return summarize_generic_event(event, f"stream:{event_type}", "")


def summarize_assistant_event(value: dict[str, Any]) -> str | None:
    message = value.get("message")
    content = message.get("content") if isinstance(message, dict) else value.get("content")
    lines: list[str] = []
    if isinstance(content, list):
        for block in content:
            line = summarize_content_block(block)
            if line:
                lines.append(line)
    elif isinstance(content, str) and content.strip():
        lines.append(f"assistant: {excerpt(content, 220)}")
    if lines:
        return " / ".join(lines[:3])
    delta = value.get("delta")
    if isinstance(delta, dict):
        text = delta.get("text") or delta.get("partial_json")
        if isinstance(text, str) and text.strip():
            return f"assistant: {excerpt(text, 220)}"
    return None


def summarize_user_event(value: dict[str, Any]) -> str | None:
    message = value.get("message")
    content = message.get("content") if isinstance(message, dict) else value.get("content")
    if not isinstance(content, list):
        return None
    lines = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            result = block.get("content")
            if isinstance(result, list):
                result = " ".join(str(item.get("text", "")) for item in result if isinstance(item, dict))
            if isinstance(result, str) and result.strip():
                lines.append(f"tool result: {excerpt(result, 220)}")
            else:
                lines.append("tool result received")
    return " / ".join(lines[:2]) if lines else None


def summarize_content_block(block: Any) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            return f"assistant: {excerpt(text, 220)}"
    if block_type == "tool_use":
        name = str(block.get("name") or "tool")
        return f"tool {name}: {summarize_tool_input(block.get('input'))}"
    if block_type in {"tool_result", "server_tool_use"}:
        return f"{block_type}: {summarize_tool_input(block)}"
    return None


def summarize_tool_input(value: Any) -> str:
    if not isinstance(value, dict):
        return excerpt(str(value), 220)
    for key in ("file_path", "path", "pattern", "cmd", "command", "description"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return f"{key}={excerpt(item, 180)}"
    if "edits" in value and isinstance(value["edits"], list):
        return f"edits={len(value['edits'])}"
    return excerpt(json.dumps(value, ensure_ascii=False, sort_keys=True), 220)


def summarize_generic_event(value: dict[str, Any], event_type: str, subtype: str) -> str | None:
    for key in ("message", "text", "error"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            label = event_type or key
            return f"{label}: {excerpt(item, 240)}"
    if event_type or subtype:
        return f"event: {event_type or 'unknown'} {subtype}".strip()
    return None


def excerpt(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def resolve_task(value: str, repo_root: Path, worktree: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = worktree / value
    if candidate.exists():
        return candidate.resolve()
    task_id = normalize_task_id(value)
    matches = sorted((repo_root / ".claude" / "tasks").glob(f"task_{task_id}_*.md"))
    if not matches:
        raise FileNotFoundError(f"could not resolve Claude task: {value}")
    return matches[0].resolve()


def normalize_task_id(value: str) -> str:
    match = re.search(r"(\d{1,2})", value)
    if not match:
        raise ValueError(f"task id must contain a number: {value}")
    return f"{int(match.group(1)):02d}"


def build_prompt(task_path: Path, *, commit_request: bool, extra: str) -> str:
    commit_text = "Commit your changes on this branch when the task is complete." if commit_request else ""
    extra_text = f"\n\nExtra instruction:\n{extra}" if extra else ""
    return (
        f"Read {task_path} and complete exactly that task in this worktree. "
        "Keep the PR focused, run the task's acceptance checks where feasible, "
        "and leave a concise summary in your final response. "
        "External method repositories under /home/robin_wang are read-only evidence sources: "
        "inspect them, but do not edit, create, delete, format, patch, checkout, reset, or commit files there. "
        "All file changes must stay inside the current worktree. "
        "Before your final response, run git status --short in every external method repo you inspected; "
        "if any external repo is dirty, stop and report it instead of continuing. "
        f"{commit_text}{extra_text}"
    )


def task_title(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("#"):
                return text.lstrip("#").strip()
    except OSError:
        return None
    return None


def git_value(worktree: Path, args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-")[:80] or "session"


def redact_command(command: list[str]) -> list[str]:
    redacted = []
    skip_next = False
    for item in command:
        if skip_next:
            redacted.append("<prompt>")
            skip_next = False
            continue
        redacted.append(item)
        if item == "-p":
            skip_next = True
    return redacted


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
