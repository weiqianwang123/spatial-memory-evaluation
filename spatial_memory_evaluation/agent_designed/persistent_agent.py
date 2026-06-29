"""A persistent Claude CLI agent process (one per scene), fed messages over stdin.

The per-scene self-eval launches ONE long-lived ``claude`` process per scene using
``--input-format stream-json --output-format stream-json`` and feeds it the scene's
queries one at a time over stdin. The process stays alive between queries, so only
the FIRST message pays the ~4 s process boot; subsequent turns are ~2 s (vs ~23 s
per cold-start CLI call with --resume). This is the "launch one agent per scene,
keep it alive, hand it queries one by one" model — and multiple scenes' agents run
concurrently.

Each call to ``ask(text)`` sends one user message and blocks until that turn's
``result`` event, returning the assistant's final text for that turn. Context
accumulates across calls within the process (true multi-turn session).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class PersistentAgent:
    def __init__(
        self,
        *,
        model: str,
        cwd: Path,
        bedrock: bool = True,
        region: str = "us-west-2",
        boot_timeout: float = 90.0,
        turn_timeout: float = 180.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.model = model
        self.cwd = Path(cwd)
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.turn_timeout = turn_timeout
        self.boot_timeout = boot_timeout
        env = dict(os.environ)
        if bedrock:
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            env["AWS_REGION"] = region
        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
            "--permission-mode", "bypassPermissions",
            "--model", model,
            "--add-dir", str(self.cwd),
        ] + (extra_args or [])
        self._proc = subprocess.Popen(
            cmd, cwd=str(self.cwd), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._results: "queue.Queue[str]" = queue.Queue()
        self._alive = True
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # one "result" event terminates each turn
            if obj.get("type") == "result":
                self._results.put(str(obj.get("result") or ""))
        self._alive = False

    def ask(self, text: str) -> str:
        """Send one user message; block until this turn's result. '' on failure."""
        if not self._alive or self._proc.stdin is None:
            return ""
        msg = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._alive = False
            return ""
        try:
            return self._results.get(timeout=self.turn_timeout)
        except queue.Empty:
            return ""

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def __enter__(self) -> "PersistentAgent":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
