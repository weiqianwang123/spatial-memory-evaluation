"""LLM-Match judge for Track 3 (OpenEQA general QA).

OpenEQA scores open-ended answers with an LLM judge that rates the predicted
answer against the ground-truth answer. This module provides a judge backed by a
command-line LLM transport (e.g. the local Claude CLI), kept separate from the
answering LLM so the judge is an independent grader.

The judge returns a score in [0, 1]. Following OpenEQA, the LLM is asked for an
integer correctness rating in 1..5; we map it to (rating-1)/4.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


JUDGE_PROMPT = """You are grading an answer to a question about an indoor scene.
You are given the question, the correct (ground-truth) answer, and a candidate
answer. Rate how well the candidate answer matches the correct answer on a scale
of 1 to 5:

5 = fully correct (same meaning as the ground truth)
4 = mostly correct
3 = partially correct
2 = mostly wrong
1 = completely wrong or irrelevant

Question: {question}
Correct answer: {gt_answer}
Candidate answer: {predicted}

Reply with ONLY a single integer from 1 to 5 and nothing else."""


def make_cli_judge(judge_command: str, *, timeout: int = 120) -> Callable[[str, str, str], float]:
    """Build a judge that scores via a CLI LLM transport.

    ``judge_command`` is a shell template with a ``{prompt_path}`` placeholder; the
    command must print the rating to stdout. Example::

        claude -p "$(cat {prompt_path})" --output-format text

    Returns a ``judge(question, gt_answer, predicted) -> float in [0,1]``.
    """

    def _judge(question: str, gt_answer: str, predicted: str) -> float:
        if not gt_answer.strip() or not predicted.strip():
            return 0.0
        prompt = JUDGE_PROMPT.format(question=question, gt_answer=gt_answer, predicted=predicted)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            prompt_path = Path(f.name)
        try:
            command = judge_command.replace("{prompt_path}", str(prompt_path))
            proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0
        finally:
            prompt_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            return 0.0
        rating = _parse_rating(proc.stdout)
        if rating is None:
            return 0.0
        return (rating - 1) / 4.0

    return _judge


def _parse_rating(text: str) -> int | None:
    match = re.search(r"[1-5]", text)
    if match is None:
        return None
    return int(match.group(0))
