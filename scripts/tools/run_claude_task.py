#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


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
    metadata_path = log_dir / f"{session_id}.json"
    prompt = build_prompt(task_path, commit_request=not args.no_commit_request, extra=args.prompt_extra)
    command = [
        "claude",
        "-p",
        prompt,
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "text",
        "--max-budget-usd",
        str(args.budget),
    ]
    env = os.environ.copy()
    env[args.model_env] = "1"
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
        "summary": f"Running {task_title(task_path) or task_path.name}",
        "command": redact_command(command),
    }
    if args.dry_run:
        metadata["status"] = "dry_run"
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    with stdout_log.open("ab") as out, stderr_log.open("ab") as err:
        proc = subprocess.Popen(command, cwd=worktree, env=env, stdout=out, stderr=err, start_new_session=True)
        metadata["pid"] = proc.pid
        metadata["status"] = "running"
        write_json(metadata_path, metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
        if args.background:
            return 0
        return_code = proc.wait()
    metadata["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata["return_code"] = return_code
    metadata["status"] = "complete" if return_code == 0 else "failed"
    metadata["summary"] = f"Claude exited with code {return_code}"
    write_json(metadata_path, metadata)
    return return_code


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
