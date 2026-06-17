#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = Path(".claude/session_logs")
TASK_DIR = Path(".claude/tasks")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass
class WorktreeInfo:
    path: Path
    branch: str
    head: str | None


@dataclass
class ProcessInfo:
    pid: int
    stat: str
    etime: str
    command: str


@dataclass
class SessionCard:
    worktree: Path
    branch: str
    head: str | None
    status: str
    pid: int | None
    elapsed: str | None
    task_path: Path | None
    task_title: str | None
    session_json: Path | None
    stdout_log: Path | None
    stderr_log: Path | None
    summary: str
    changed_files: int
    last_commit: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor local Claude worktree sessions.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="main spatial-memory-evaluation repo root",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="refresh every N seconds in the terminal; 0 prints once",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--html", type=Path, default=None, help="write a self-refreshing HTML dashboard")
    parser.add_argument("--html-refresh", type=int, default=10)
    parser.add_argument("--log-lines", type=int, default=6, help="recent log lines used for each summary")
    parser.add_argument("--show-idle", action="store_true", help="include worktrees without a task/session")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    while True:
        cards = collect_cards(args.repo_root, log_lines=args.log_lines, show_idle=args.show_idle)
        if args.json:
            print(json.dumps([card_to_json(card) for card in cards], indent=2, sort_keys=True))
        else:
            print_terminal(cards)
        if args.html is not None:
            write_html(args.html, cards, refresh_seconds=args.html_refresh)
            print(f"\nHTML dashboard: {args.html.resolve()}")
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)
        if not args.json:
            print("\033[2J\033[H", end="")


def collect_cards(repo_root: Path, *, log_lines: int, show_idle: bool) -> list[SessionCard]:
    worktrees = git_worktrees(repo_root)
    task_root = repo_root / TASK_DIR
    cards = []
    for worktree in worktrees:
        session = latest_session(worktree.path)
        procs = claude_processes_for_worktree(worktree.path)
        process = procs[0] if procs else None
        task_path = resolve_task_path(worktree, session, task_root)
        task_title = read_task_title(task_path) if task_path else None
        stdout_log = path_from_session(session, "stdout_log")
        stderr_log = path_from_session(session, "stderr_log")
        status = session_status(session, process)
        if status == "idle" and task_path is None and not show_idle:
            continue
        summary = session_summary(session, stdout_log, stderr_log, task_title, log_lines)
        changed_files = git_changed_files(worktree.path)
        last_commit = git_last_commit(worktree.path)
        cards.append(
            SessionCard(
                worktree=worktree.path,
                branch=worktree.branch,
                head=worktree.head,
                status=status,
                pid=process.pid if process else int(session.get("pid")) if session and session.get("pid") else None,
                elapsed=process.etime if process else None,
                task_path=task_path,
                task_title=task_title,
                session_json=Path(session["_path"]) if session and session.get("_path") else None,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
                summary=summary,
                changed_files=changed_files,
                last_commit=last_commit,
            )
        )
    cards.sort(key=lambda card: (status_rank(card.status), card.branch))
    return cards


def git_worktrees(repo_root: Path) -> list[WorktreeInfo]:
    result = run(["git", "-C", str(repo_root), "worktree", "list", "--porcelain"])
    if result.returncode != 0:
        return [WorktreeInfo(path=repo_root, branch=current_branch(repo_root), head=None)]
    worktrees: list[WorktreeInfo] = []
    current: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                worktrees.append(worktree_from_porcelain(current))
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        worktrees.append(worktree_from_porcelain(current))
    return worktrees


def worktree_from_porcelain(value: dict[str, str]) -> WorktreeInfo:
    branch = value.get("branch", "")
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/") :]
    if not branch:
        branch = "(detached)"
    return WorktreeInfo(path=Path(value["worktree"]), branch=branch, head=value.get("HEAD"))


def latest_session(worktree: Path) -> dict[str, Any] | None:
    session_dir = worktree / SESSION_DIR
    if not session_dir.exists():
        return None
    candidates = sorted(session_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8") as f:
                value = json.load(f)
            if isinstance(value, dict):
                value["_path"] = str(path)
                return value
        except (OSError, json.JSONDecodeError):
            continue
    return None


def resolve_task_path(worktree: WorktreeInfo, session: dict[str, Any] | None, task_root: Path) -> Path | None:
    if session:
        task_value = session.get("task_path")
        if isinstance(task_value, str) and task_value:
            path = Path(task_value)
            if path.exists():
                return path
            candidate = worktree.path / task_value
            if candidate.exists():
                return candidate
    task_id = task_id_from_branch(worktree.branch)
    if task_id is None:
        return None
    matches = sorted(task_root.glob(f"task_{task_id}_*.md"))
    return matches[0] if matches else None


def task_id_from_branch(branch: str) -> str | None:
    patterns = (
        r"task[-_/]?(\d{1,2})",
        r"task_(\d{1,2})",
    )
    for pattern in patterns:
        match = re.search(pattern, branch)
        if match:
            return f"{int(match.group(1)):02d}"
    return None


def read_task_title(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text.startswith("#"):
                return text.lstrip("#").strip()
            if text:
                return text[:100]
    except OSError:
        return None
    return None


def claude_processes_for_worktree(worktree: Path) -> list[ProcessInfo]:
    processes = []
    proc_root = Path("/proc")
    if proc_root.exists():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
                if not is_claude_command(cmdline):
                    continue
                cwd = (entry / "cwd").resolve()
                if not is_relative_to(cwd, worktree):
                    continue
                stat = read_proc_stat(entry)
                etime = process_etime_from_proc(entry)
                processes.append(ProcessInfo(pid=int(entry.name), stat=stat, etime=etime, command=cmdline))
            except (OSError, RuntimeError):
                continue
        return sorted(processes, key=lambda proc: proc.pid)

    result = run(["ps", "-eo", "pid=,stat=,etime=,command="])
    if result.returncode != 0:
        return []
    for line in result.stdout.splitlines():
        if not is_claude_command(line):
            continue
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            processes.append(ProcessInfo(pid=int(parts[0]), stat=parts[1], etime=parts[2], command=parts[3]))
    return processes


def is_claude_command(command: str) -> bool:
    lowered = command.lower()
    if "claude_session_dashboard.py" in lowered or "run_claude_task.py" in lowered:
        return False
    tokens = command.split()
    if tokens and Path(tokens[0]).name == "claude":
        return True
    return bool(re.search(r"(^|\s)claude(\s|$)", command))


def read_proc_stat(proc_dir: Path) -> str:
    try:
        text = (proc_dir / "stat").read_text(encoding="utf-8", errors="ignore")
        # /proc/<pid>/stat: pid (comm) state ...
        after = text.rsplit(")", 1)[1].strip()
        return after.split()[0]
    except OSError:
        return "?"


def process_etime_from_proc(proc_dir: Path) -> str:
    try:
        ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        boot = 0.0
        with Path("/proc/stat").open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime "):
                    boot = float(line.split()[1])
                    break
        stat = (proc_dir / "stat").read_text(encoding="utf-8", errors="ignore")
        fields = stat.rsplit(")", 1)[1].strip().split()
        start_ticks = float(fields[19])
        start_time = boot + start_ticks / ticks
        return format_duration(max(0.0, time.time() - start_time))
    except (OSError, ValueError, IndexError, KeyError):
        return "?"


def session_status(session: dict[str, Any] | None, process: ProcessInfo | None) -> str:
    if process is not None:
        return "running"
    if not session:
        return "idle"
    status = str(session.get("status") or "")
    if status in {"complete", "failed", "running"}:
        if status == "running":
            return "stopped"
        return status
    return "stopped"


def session_summary(
    session: dict[str, Any] | None,
    stdout_log: Path | None,
    stderr_log: Path | None,
    task_title: str | None,
    log_lines: int,
) -> str:
    if session and isinstance(session.get("summary"), str) and session["summary"].strip():
        return clean_text(session["summary"])
    lines = []
    if stdout_log is not None:
        lines.extend(tail_nonempty(stdout_log, log_lines))
    if not lines and stderr_log is not None:
        lines.extend(tail_nonempty(stderr_log, log_lines))
    if lines:
        return clean_text(" / ".join(lines[-log_lines:]))
    if task_title:
        return task_title
    return "No active task summary yet."


def path_from_session(session: dict[str, Any] | None, key: str) -> Path | None:
    if not session:
        return None
    value = session.get(key)
    if isinstance(value, str) and value:
        path = Path(value)
        return path if path.exists() else path
    return None


def tail_nonempty(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    cleaned = [clean_text(line) for line in lines if clean_text(line)]
    return cleaned[-count:]


def clean_text(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def git_changed_files(worktree: Path) -> int:
    result = run(["git", "-C", str(worktree), "status", "--short"])
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def git_last_commit(worktree: Path) -> str | None:
    result = run(["git", "-C", str(worktree), "log", "-1", "--oneline", "--decorate"])
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def current_branch(worktree: Path) -> str:
    result = run(["git", "-C", str(worktree), "branch", "--show-current"])
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "(unknown)"


def print_terminal(cards: list[SessionCard]) -> None:
    print(f"Claude session dashboard  {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    if not cards:
        print("No Claude worktree sessions found. Use --show-idle to include idle worktrees.")
        return
    for card in cards:
        status = color_status(card.status)
        print(f"{status}  branch={card.branch}  pid={card.pid or '-'}  changed={card.changed_files}")
        print(f"  worktree: {card.worktree}")
        print(f"  task    : {card.task_title or '-'}")
        if card.elapsed:
            print(f"  elapsed : {card.elapsed}")
        print(f"  summary : {card.summary}")
        if card.stdout_log:
            print(f"  inspect : cd {card.worktree} && tail -f {relative_display(card.stdout_log, card.worktree)}")
        else:
            print(f"  inspect : cd {card.worktree} && git status")
        if card.last_commit:
            print(f"  commit  : {card.last_commit}")
        print("-" * 100)


def color_status(status: str) -> str:
    colors = {
        "running": "\033[32m",
        "stopped": "\033[33m",
        "failed": "\033[31m",
        "complete": "\033[36m",
        "idle": "\033[90m",
    }
    return f"{colors.get(status, '')}{status.upper():<8}\033[0m"


def write_html(path: Path, cards: list[SessionCard], *, refresh_seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(render_card_html(card) for card in cards) or "<p>No sessions found.</p>"
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>Claude Session Dashboard</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f7f7f5; color: #191919; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    .sub {{ color: #666; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .top {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; }}
    .status {{ border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 700; }}
    .running {{ background: #d9f8df; color: #126b2f; }}
    .stopped {{ background: #fff1c7; color: #815600; }}
    .failed {{ background: #ffd8d6; color: #8a1f17; }}
    .complete {{ background: #d9f0ff; color: #14577a; }}
    .idle {{ background: #ececec; color: #555; }}
    code {{ background: #f1f1ee; padding: 2px 4px; border-radius: 4px; }}
    .label {{ color: #666; font-size: 12px; text-transform: uppercase; margin-top: 10px; }}
    .summary {{ white-space: pre-wrap; line-height: 1.35; }}
  </style>
</head>
<body>
  <h1>Claude Session Dashboard</h1>
  <div class="sub">Updated {html.escape(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}; refreshes every {refresh_seconds}s.</div>
  <div class="grid">{body}</div>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def render_card_html(card: SessionCard) -> str:
    inspect = f"cd {card.worktree} && git status"
    if card.stdout_log:
        inspect = f"cd {card.worktree} && tail -f {relative_display(card.stdout_log, card.worktree)}"
    return f"""<div class="card">
  <div class="top">
    <strong>{html.escape(card.branch)}</strong>
    <span class="status {html.escape(card.status)}">{html.escape(card.status.upper())}</span>
  </div>
  <div class="label">Task</div>
  <div>{html.escape(card.task_title or "-")}</div>
  <div class="label">Worktree</div>
  <code>{html.escape(str(card.worktree))}</code>
  <div class="label">Summary</div>
  <div class="summary">{html.escape(card.summary)}</div>
  <div class="label">Inspect</div>
  <code>{html.escape(inspect)}</code>
  <div class="label">Meta</div>
  <div>pid={html.escape(str(card.pid or "-"))} changed={card.changed_files} elapsed={html.escape(card.elapsed or "-")}</div>
</div>"""


def card_to_json(card: SessionCard) -> dict[str, Any]:
    return {
        "worktree": str(card.worktree),
        "branch": card.branch,
        "head": card.head,
        "status": card.status,
        "pid": card.pid,
        "elapsed": card.elapsed,
        "task_path": str(card.task_path) if card.task_path else None,
        "task_title": card.task_title,
        "session_json": str(card.session_json) if card.session_json else None,
        "stdout_log": str(card.stdout_log) if card.stdout_log else None,
        "stderr_log": str(card.stderr_log) if card.stderr_log else None,
        "summary": card.summary,
        "changed_files": card.changed_files,
        "last_commit": card.last_commit,
    }


def status_rank(status: str) -> int:
    return {"running": 0, "stopped": 1, "failed": 2, "complete": 3, "idle": 4}.get(status, 9)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


if __name__ == "__main__":
    raise SystemExit(main())
