"""Per-scene SESSION judge for the agent_designed SELF-eval.

The Track 3 evaluator calls ``judge(question, gt, predicted) -> float`` once per
question. Instead of one fresh CLI call per question (a cold agent each time) OR
one giant prompt with all questions at once, this judge launches ONE persistent
agent SESSION per scene and feeds it the scene's questions ONE AT A TIME (multi-
turn via the CLI's --session-id / --resume). Each turn grades a single question
1-5; the shared session context persists across the scene's questions.

Interface matches ``track3.judge.make_cli_judge``: build with a judge command
template, optionally ``prime(triples)`` once per scene to open the session and
grade all of that scene's questions in order, then ``judge(q, gt, pred)`` serves
the cached rating. (If un-primed, a single-turn fresh session grades on demand.)

The judge command template uses {prompt_path} like the base judge, AND may use
{session_args} where the session flags get injected (--session-id <uuid> for the
first turn, --resume <uuid> for later turns). If {session_args} is absent we fall
back to stateless per-question calls (still one question at a time).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable

_TURN_PROMPT = """You are grading answers to questions about ONE indoor scene. I
will give you the scene's questions one at a time. For each, rate how well the
candidate answer matches the correct (ground-truth) answer, 1-5:
5 = fully correct (same meaning), 4 = mostly correct, 3 = partially correct,
2 = mostly wrong, 1 = completely wrong or irrelevant.

Question: {question}
Correct answer: {gt_answer}
Candidate answer: {predicted}

Reply with ONLY a single integer 1-5 and nothing else."""


def make_batch_cli_judge(judge_command: str, *, timeout: int = 180) -> Callable[[str, str, str], float]:
    """Build a per-scene session judge. Name kept for drop-in compatibility.

    Returns ``judge(question, gt, predicted) -> float in [0,1]`` plus a
    ``judge.prime(triples)`` that opens ONE session for the scene and grades the
    triples in order (one turn per question), caching results.
    """

    cache: dict[tuple[str, str, str], float] = {}

    def _key(q: str, gt: str, pred: str) -> tuple[str, str, str]:
        return (q.strip(), gt.strip(), pred.strip())

    def _session_flags(session_id: str, first: bool) -> str:
        return f"--session-id {session_id}" if first else f"--resume {session_id}"

    def _run_turn(question: str, gt: str, pred: str, session_id: str, first: bool) -> float:
        if not gt.strip() or not pred.strip():
            return 0.0
        prompt = _TURN_PROMPT.format(question=question, gt_answer=gt, predicted=pred)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            p = Path(f.name)
        try:
            cmd = judge_command.replace("{prompt_path}", str(p))
            if "{session_args}" in cmd:
                cmd = cmd.replace("{session_args}", _session_flags(session_id, first))
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            out = proc.stdout if proc.returncode == 0 else ""
        except subprocess.TimeoutExpired:
            out = ""
        finally:
            p.unlink(missing_ok=True)
        r = _parse_rating(out)
        return ((r - 1) / 4.0) if r is not None else 0.0

    def prime(triples: list[tuple[str, str, str]]) -> None:
        """Open ONE session for this scene; grade triples in order (one turn each)."""
        session_id = str(uuid.uuid4())
        first = True
        for (q, gt, pred) in triples:
            val = _run_turn(q, gt, pred, session_id, first)
            cache[_key(q, gt, pred)] = val
            # only the FIRST scorable turn opens the session; the rest resume it
            if gt.strip() and pred.strip():
                first = False

    def _judge(question: str, gt_answer: str, predicted: str) -> float:
        k = _key(question, gt_answer, predicted)
        if k in cache:
            return cache[k]
        # un-primed fallback: a one-off single-turn session for this question
        val = _run_turn(question, gt_answer, predicted, str(uuid.uuid4()), first=True)
        cache[k] = val
        return val

    _judge.prime = prime  # type: ignore[attr-defined]
    return _judge


def _parse_rating(text: str) -> int | None:
    m = re.search(r"[1-5]", text)
    return int(m.group(0)) if m else None
