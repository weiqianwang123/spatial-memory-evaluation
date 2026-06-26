"""Batched LLM-Match judge for the agent_designed SELF-eval (one call per scene).

The Track 3 evaluator calls a ``judge(question, gt, predicted) -> float`` once per
question. With a Bedrock CLI judge that is N serial calls per scene (~13-14 ×
~15-20 s) — the slow part of a round. For the SELF-reference dev score that
precision isn't needed: this judge scores ALL of a scene's questions in ONE LLM
call ("one agent per scene, not per query"), then serves the cached per-question
ratings through the same callable signature the evaluator expects.

It is a drop-in for ``track3.judge.make_cli_judge`` in the dev-eval path only; the
real per-question judge stays available for held-out scoring if we ever want it.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

_BATCH_PROMPT = """You are grading answers to questions about an indoor scene.
For EACH item below, rate how well the candidate answer matches the correct
(ground-truth) answer on a 1-5 scale:
5 = fully correct (same meaning), 4 = mostly correct, 3 = partially correct,
2 = mostly wrong, 1 = completely wrong or irrelevant.

Return ONLY a JSON array of integers (one rating per item, in order), e.g. [5,3,1].
No prose, no keys — just the array.

Items:
{items}"""


def make_batch_cli_judge(judge_command: str, *, timeout: int = 600) -> Callable[[str, str, str], float]:
    """Judge that scores a scene's questions in one CLI call, then serves cache.

    Returns ``judge(question, gt, predicted) -> float in [0,1]`` with the same
    signature the evaluator uses. The FIRST call triggers no work; instead, call
    ``judge.prime(triples)`` once with the full list of (question, gt, predicted)
    for a scene to do the single batched LLM call. Any ``judge(...)`` lookups then
    hit the cache. If ``prime`` was not called (or a triple is missing), it falls
    back to a single-item CLI call for that triple.
    """

    cache: dict[tuple[str, str, str], float] = {}

    def _key(q: str, gt: str, pred: str) -> tuple[str, str, str]:
        return (q.strip(), gt.strip(), pred.strip())

    def _run_cli(prompt: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            p = Path(f.name)
        try:
            cmd = judge_command.replace("{prompt_path}", str(p))
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return proc.stdout if proc.returncode == 0 else ""
        except subprocess.TimeoutExpired:
            return ""
        finally:
            p.unlink(missing_ok=True)

    def prime(triples: list[tuple[str, str, str]]) -> None:
        scorable = [t for t in triples if t[1].strip() and t[2].strip()]
        for t in triples:
            if not (t[1].strip() and t[2].strip()):
                cache[_key(*t)] = 0.0  # empty gt or prediction -> 0, no LLM needed
        if not scorable:
            return
        items = "\n".join(
            f"{i+1}. Question: {q}\n   Correct: {gt}\n   Candidate: {pred}"
            for i, (q, gt, pred) in enumerate(scorable)
        )
        out = _run_cli(_BATCH_PROMPT.format(items=items))
        ratings = _parse_array(out, expected=len(scorable))
        for t, r in zip(scorable, ratings):
            cache[_key(*t)] = ((r - 1) / 4.0) if r is not None else 0.0

    def _judge(question: str, gt_answer: str, predicted: str) -> float:
        k = _key(question, gt_answer, predicted)
        if k in cache:
            return cache[k]
        if not gt_answer.strip() or not predicted.strip():
            return 0.0
        # Fallback: single-item call (prime wasn't used for this triple).
        out = _run_cli(_BATCH_PROMPT.format(
            items=f"1. Question: {question}\n   Correct: {gt_answer}\n   Candidate: {predicted}"))
        ratings = _parse_array(out, expected=1)
        val = ((ratings[0] - 1) / 4.0) if ratings and ratings[0] is not None else 0.0
        cache[k] = val
        return val

    _judge.prime = prime  # type: ignore[attr-defined]
    return _judge


def _parse_array(text: str, expected: int) -> list[int | None]:
    """Parse a JSON array of 1-5 ints; tolerate stray prose by regex fallback."""
    text = text.strip()
    try:
        arr = json.loads(text[text.index("["): text.rindex("]") + 1])
        out = [int(x) if isinstance(x, (int, float)) and 1 <= int(x) <= 5 else None for x in arr]
    except (ValueError, TypeError):
        out = [int(m) for m in re.findall(r"[1-5]", text)]
    # pad/trim to expected length
    out = (out + [None] * expected)[:expected]
    return out
