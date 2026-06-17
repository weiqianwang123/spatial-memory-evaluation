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

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_claude_task import summarize_stream_json_line


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
    stream_json_log: Path | None
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
        stream_json_log = path_from_session(session, "stream_json_log")
        status = session_status(session, process)
        if status == "idle" and task_path is None and not show_idle:
            continue
        summary = session_summary(session, stdout_log, stderr_log, stream_json_log, task_title, log_lines)
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
                stream_json_log=stream_json_log,
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
    sessions = []
    for path in session_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                value = json.load(f)
            if isinstance(value, dict):
                value["_path"] = str(path)
                sessions.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    if not sessions:
        return None
    return max(sessions, key=session_score)


def session_score(session: dict[str, Any]) -> tuple[int, int, int, float]:
    """Prefer a live session with visible activity over a newer silent one."""
    active = 1 if session_pid_is_alive(session) else 0
    visible = 1 if session_has_visible_activity(session) else 0
    stream = 1 if session.get("stream_json") else 0
    return (active and visible, active, stream or visible, session_activity_mtime(session))


def session_pid_is_alive(session: dict[str, Any]) -> bool:
    pid = session.get("pid")
    try:
        return pid is not None and Path("/proc", str(int(pid))).exists()
    except (TypeError, ValueError):
        return False


def session_has_visible_activity(session: dict[str, Any]) -> bool:
    for key in ("stdout_log", "stream_json_log", "stderr_log"):
        value = session.get(key)
        if isinstance(value, str) and has_content(Path(value)):
            return True
    return False


def session_activity_mtime(session: dict[str, Any]) -> float:
    mtimes = []
    for key in ("_path", "stdout_log", "stream_json_log", "stderr_log"):
        value = session.get(key)
        if isinstance(value, str):
            try:
                mtimes.append(Path(value).stat().st_mtime)
            except OSError:
                pass
    return max(mtimes) if mtimes else 0.0


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
    if status in {"complete", "failed", "running", "interrupted"}:
        if status == "running":
            return "stopped"
        return status
    return "stopped"


def session_summary(
    session: dict[str, Any] | None,
    stdout_log: Path | None,
    stderr_log: Path | None,
    stream_json_log: Path | None,
    task_title: str | None,
    log_lines: int,
) -> str:
    summary = ""
    if session and isinstance(session.get("summary"), str):
        summary = clean_text(session["summary"])
    lines = recent_activity_lines(stdout_log, stderr_log, stream_json_log, log_lines)
    if lines:
        if not summary or summary.startswith("Running "):
            return clean_text(" / ".join(lines[-log_lines:]))
    if summary:
        return summary
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


def recent_activity_lines(
    stdout_log: Path | None,
    stderr_log: Path | None,
    stream_json_log: Path | None,
    count: int,
) -> list[str]:
    lines = tail_nonempty(stdout_log, count) if stdout_log is not None else []
    if not lines and stream_json_log is not None:
        lines = tail_stream_json(stream_json_log, count)
    if not lines and stderr_log is not None:
        lines = tail_nonempty(stderr_log, count)
    return lines


def tail_stream_json(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    summaries = []
    for line in lines[-count * 8 :]:
        summary = summarize_stream_json_line(line)
        if summary:
            summaries.append(clean_text(summary))
    return summaries[-count:]


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
        inspect_log = preferred_inspect_log(card)
        if inspect_log:
            print(f"  inspect : cd {card.worktree} && tail -f {relative_display(inspect_log, card.worktree)}")
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
        "interrupted": "\033[35m",
        "complete": "\033[36m",
        "idle": "\033[90m",
    }
    return f"{colors.get(status, '')}{status.upper():<8}\033[0m"


def write_html(path: Path, cards: list[SessionCard], *, refresh_seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(render_card_html(card) for card in cards) or render_empty_html()
    counts = {status: 0 for status in ("running", "stopped", "failed", "interrupted", "complete", "idle")}
    for card in cards:
        counts[card.status] = counts.get(card.status, 0) + 1
    chips = "\n".join(
        f'<button class="chip" data-filter="{html.escape(status)}">{html.escape(status)} <span>{count}</span></button>'
        for status, count in counts.items()
        if count or status in {"running", "stopped"}
    )
    updated = html.escape(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    css = """
    :root {
      --ink: #202124;
      --muted: #686b73;
      --panel: rgba(255, 255, 255, 0.86);
      --line: rgba(42, 45, 52, 0.11);
      --shadow: 0 18px 50px rgba(30, 33, 40, 0.10);
      --green: #3aa76d;
      --yellow: #c78a13;
      --red: #d15b52;
      --blue: #4b84d8;
      --gray: #7c818c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 10%, rgba(255, 220, 175, 0.58), transparent 24rem),
        radial-gradient(circle at 90% 8%, rgba(185, 216, 255, 0.48), transparent 22rem),
        linear-gradient(135deg, #fbfaf6 0%, #f3f6fb 48%, #fbf8f2 100%);
    }
    .shell { max-width: 1280px; margin: 0 auto; padding: 28px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 18px;
    }
    h1 { margin: 0; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }
    .subtitle { margin: 8px 0 0; color: var(--muted); line-height: 1.45; }
    .refresh {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      box-shadow: var(--shadow);
      min-width: 230px;
    }
    .refresh strong { display: block; font-size: 13px; color: var(--muted); text-transform: uppercase; }
    .refresh span { display: block; margin-top: 4px; font-weight: 700; }
    .toolbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      margin: 0 0 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.76);
      backdrop-filter: blur(12px);
      box-shadow: var(--shadow);
    }
    .search {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      font: inherit;
      background: rgba(255, 255, 255, 0.9);
      outline: none;
    }
    .search:focus { border-color: rgba(75, 132, 216, 0.55); box-shadow: 0 0 0 3px rgba(75, 132, 216, 0.12); }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      background: white;
      color: var(--ink);
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease;
    }
    .chip:hover, .chip.active { transform: translateY(-1px); border-color: rgba(75, 132, 216, 0.55); }
    .chip span { margin-left: 6px; color: var(--muted); font-weight: 700; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; align-items: start; }
    .card {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .card:hover { transform: translateY(-2px); box-shadow: 0 22px 58px rgba(30, 33, 40, 0.14); border-color: rgba(75, 132, 216, 0.28); }
    .card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 5px; background: var(--gray); }
    .card.running::before { background: var(--green); }
    .card.stopped::before { background: var(--yellow); }
    .card.failed::before { background: var(--red); }
    .card.interrupted::before { background: #8c5fd4; }
    .card.complete::before { background: var(--blue); }
    .card-inner { padding: 16px 16px 15px 20px; }
    .card-top { display: flex; align-items: start; justify-content: space-between; gap: 12px; }
    .branch { font-weight: 800; line-height: 1.25; word-break: break-word; }
    .badge {
      flex: 0 0 auto;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.02em;
      background: #eceff3;
      color: var(--gray);
    }
    .badge.running { background: #def7e8; color: #166c42; }
    .badge.stopped { background: #fff1ce; color: #8a5d00; }
    .badge.failed { background: #ffe0dd; color: #9a2d25; }
    .badge.interrupted { background: #f0e4ff; color: #6f3bb2; }
    .badge.complete { background: #dfeeff; color: #245d99; }
    .task { margin-top: 10px; color: var(--muted); line-height: 1.35; }
    .summary {
      margin: 14px 0;
      padding: 12px;
      border-radius: 8px;
      background: rgba(247, 248, 250, 0.88);
      border: 1px solid var(--line);
      line-height: 1.42;
    }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
    .stat { border: 1px solid var(--line); border-radius: 8px; padding: 9px; background: rgba(255,255,255,0.62); }
    .stat label { display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }
    .stat strong { display: block; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    button.copy {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #202124;
      color: white;
      font: inherit;
      cursor: pointer;
    }
    details { margin-top: 12px; }
    summary { cursor: pointer; color: var(--muted); font-weight: 700; }
    .detail-grid { display: grid; gap: 9px; margin-top: 10px; }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      word-break: break-all;
      background: rgba(244, 245, 247, 0.9);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
    }
    .logs {
      white-space: pre-wrap;
      max-height: 180px;
      overflow: auto;
      line-height: 1.4;
    }
    .empty {
      grid-column: 1 / -1;
      padding: 24px;
      border: 1px dashed rgba(42,45,52,0.22);
      border-radius: 8px;
      background: rgba(255,255,255,0.66);
      text-align: center;
      color: var(--muted);
    }
    .hidden { display: none; }
    @media (max-width: 760px) {
      .shell { padding: 18px; }
      .hero, .toolbar { grid-template-columns: 1fr; }
      .chips { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
    }
    """
    script = """
    const buttons = Array.from(document.querySelectorAll('[data-filter]'));
    const cards = Array.from(document.querySelectorAll('.card'));
    const search = document.querySelector('#search');
    let active = localStorage.getItem('claude-dashboard-filter') || 'all';
    search.value = localStorage.getItem('claude-dashboard-search') || '';

    function applyFilters() {
      const q = (search.value || '').toLowerCase().trim();
      for (const card of cards) {
        const statusMatch = active === 'all' || card.dataset.status === active;
        const textMatch = !q || card.dataset.search.includes(q);
        card.classList.toggle('hidden', !(statusMatch && textMatch));
      }
    }

    for (const button of buttons) {
      button.addEventListener('click', () => {
        active = button.dataset.filter;
        localStorage.setItem('claude-dashboard-filter', active);
        buttons.forEach(b => b.classList.toggle('active', b === button));
        applyFilters();
      });
    }
    buttons.forEach(b => b.classList.toggle('active', b.dataset.filter === active));
    search.addEventListener('input', () => {
      localStorage.setItem('claude-dashboard-search', search.value || '');
      applyFilters();
    });
    applyFilters();

    document.querySelectorAll('[data-copy]').forEach(button => {
      button.addEventListener('click', async () => {
        const text = button.dataset.copy;
        try {
          await navigator.clipboard.writeText(text);
          const old = button.textContent;
          button.textContent = 'Copied';
          setTimeout(() => button.textContent = old, 900);
        } catch {
          window.prompt('Copy command', text);
        }
      });
    });
    """
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>Claude Session Dashboard</title>
  <style>{css}</style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <h1>Claude Sessions</h1>
        <p class="subtitle">A local dashboard for worktree agents: branch, task, logs, changes, and quick inspection commands.</p>
      </div>
      <div class="refresh">
        <strong>Updated</strong>
        <span>{updated}</span>
        <strong style="margin-top: 8px;">Auto refresh</strong>
        <span>{refresh_seconds}s</span>
      </div>
    </section>
    <section class="toolbar">
      <input id="search" class="search" placeholder="Search branch, task, worktree, summary..." />
      <div class="chips">
        <button class="chip active" data-filter="all">all <span>{len(cards)}</span></button>
        {chips}
      </div>
    </section>
    <section class="grid">{body}</section>
  </main>
  <script>{script}</script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def render_card_html(card: SessionCard) -> str:
    inspect = f"cd {card.worktree} && git status"
    inspect_log = preferred_inspect_log(card)
    if inspect_log:
        inspect = f"cd {card.worktree} && tail -f {relative_display(inspect_log, card.worktree)}"
    open_worktree = f"cd {card.worktree}"
    log_lines = recent_activity_lines(card.stdout_log, card.stderr_log, card.stream_json_log, 12)
    log_preview = "\n".join(log_lines[-12:]) if log_lines else "No recent log lines."
    search_text = " ".join(
        str(value or "")
        for value in (
            card.branch,
            card.status,
            card.task_title,
            card.worktree,
            card.summary,
            card.last_commit,
        )
    ).lower()
    task_path = str(card.task_path) if card.task_path else "-"
    session_path = str(card.session_json) if card.session_json else "-"
    stdout_path = str(card.stdout_log) if card.stdout_log else "-"
    stderr_path = str(card.stderr_log) if card.stderr_log else "-"
    stream_path = str(card.stream_json_log) if card.stream_json_log else "-"
    return f"""<article class="card {html.escape(card.status)}" data-status="{html.escape(card.status)}" data-search="{html.escape(search_text)}">
  <div class="card-inner">
    <div class="card-top">
      <div class="branch">{html.escape(card.branch)}</div>
      <span class="badge {html.escape(card.status)}">{html.escape(card.status.upper())}</span>
    </div>
    <div class="task">{html.escape(card.task_title or "No task detected yet")}</div>
    <div class="summary">{html.escape(card.summary)}</div>
    <div class="stats">
      <div class="stat"><label>pid</label><strong>{html.escape(str(card.pid or "-"))}</strong></div>
      <div class="stat"><label>changed</label><strong>{card.changed_files}</strong></div>
      <div class="stat"><label>elapsed</label><strong>{html.escape(card.elapsed or "-")}</strong></div>
    </div>
    <div class="actions">
      <button class="copy" data-copy="{html.escape(inspect, quote=True)}">Copy inspect</button>
      <button class="copy" data-copy="{html.escape(open_worktree, quote=True)}">Copy cd</button>
    </div>
    <details>
      <summary>Details and logs</summary>
      <div class="detail-grid">
        <div class="mono"><strong>worktree</strong><br>{html.escape(str(card.worktree))}</div>
        <div class="mono"><strong>task</strong><br>{html.escape(task_path)}</div>
        <div class="mono"><strong>session</strong><br>{html.escape(session_path)}</div>
        <div class="mono"><strong>stdout</strong><br>{html.escape(stdout_path)}</div>
        <div class="mono"><strong>stderr</strong><br>{html.escape(stderr_path)}</div>
        <div class="mono"><strong>stream json</strong><br>{html.escape(stream_path)}</div>
        <div class="mono"><strong>last commit</strong><br>{html.escape(card.last_commit or "-")}</div>
        <div class="mono logs"><strong>recent log</strong><br>{html.escape(log_preview)}</div>
      </div>
    </details>
  </div>
</article>"""


def render_empty_html() -> str:
    return """<div class="empty">
  <strong>No sessions visible.</strong>
  <div>Start a task with <code>scripts/tools/run_claude_task.py</code>, or rerun with <code>--show-idle</code>.</div>
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
        "stream_json_log": str(card.stream_json_log) if card.stream_json_log else None,
        "summary": card.summary,
        "changed_files": card.changed_files,
        "last_commit": card.last_commit,
    }


def status_rank(status: str) -> int:
    return {"running": 0, "stopped": 1, "failed": 2, "interrupted": 3, "complete": 4, "idle": 5}.get(status, 9)


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


def preferred_inspect_log(card: SessionCard) -> Path | None:
    if card.stdout_log is not None and has_content(card.stdout_log):
        return card.stdout_log
    if card.stream_json_log is not None:
        return card.stream_json_log
    return card.stdout_log or card.stderr_log


def has_content(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


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
