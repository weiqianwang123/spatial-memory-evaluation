"""Track 4: OC-NaVQA long-horizon trajectory QA (CODa robot sequences).

Mirrors Track 3's per-query tool_llm agent loop (SAME answer-agent stack), but the
questions are OC-NaVQA's and scoring is TYPE-SPECIFIC (binary / position / time /
duration / text), matching ReMEmbR's NaVQA evaluation:
  - binary   : predicted yes/no == GT                       -> accuracy
  - position : ||predicted_xyz - GT_xyz||                    -> mean L2 (m) + acc@{0.5,1,3}m
  - time     : |predicted_minutes_ago - GT|                 -> mean abs error (min)
  - duration : |predicted_minutes - GT|                     -> mean abs error (min)
  - text     : LLM-Match judge (like Track 3)               -> llm_match

The answer-agent returns a short natural-language answer (response_kind="answer",
the same contract as Track 3); we PARSE the typed prediction out of that text
(numbers / yes-no / 3-vector) so any method that can answer in words works without
a new tool contract. The package's native retrieval tools are used unchanged.

A combined headline ``navqa_score`` in [0,1] aggregates per-type sub-scores
(binary acc, position acc@1m, time/duration within-tolerance rate, text llm_match)
so the track yields one comparable number alongside the per-type detail.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.matching import mean, safe_div
from spatial_memory_evaluation.common.package_io import load_package
from spatial_memory_evaluation.tool_llm import run_tool_llm_query

TRACK_KEY = "track4_oc_navqa"
QUESTIONS_FILE = "questions.jsonl"
ANSWERS_FILE = "answers.jsonl"
JudgeFn = Callable[[str, str, str], float]

# tolerances for the aggregate "within-tolerance" sub-scores
POSITION_ACC_THRESH_M = (0.5, 1.0, 3.0)
TIME_TOL_MIN = 1.0       # |pred - gt| <= 1 min counts as correct (NaVQA-style)
DURATION_TOL_MIN = 1.0


# --------------------------- typed answer parsing ---------------------------
_NUM = r"[-+]?\d+(?:\.\d+)?"


def parse_binary(text: str) -> str | None:
    t = text.strip().lower()
    # look for a clear yes/no token
    if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t):
        return "yes"
    if re.search(r"\bno\b", t) and not re.search(r"\byes\b", t):
        return "no"
    # leading token
    m = re.match(r"\s*(yes|no)\b", t)
    return m.group(1) if m else None


def parse_position(text: str) -> list[float] | None:
    """Pull the first 3-number vector from the answer (e.g. '[-79.6, 78.7, -0.5]')."""
    nums = re.findall(_NUM, text.replace(",", " "))
    if len(nums) >= 3:
        return [float(x) for x in nums[:3]]
    return None


def parse_minutes(text: str) -> float | None:
    """First number in the answer, interpreted as minutes."""
    m = re.search(_NUM, text)
    return float(m.group(0)) if m else None


# --------------------------- per-type scoring ---------------------------
def score_item(qtype: str, gt: dict[str, Any], pred_text: str,
               judge: JudgeFn | None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": qtype, "pred_text": pred_text}
    if qtype == "binary":
        p = parse_binary(pred_text)
        out["pred"] = p
        out["correct"] = 1.0 if (p is not None and p == gt.get("answer")) else 0.0
    elif qtype == "position":
        p = parse_position(pred_text)
        gtp = gt.get("answer_position")
        if p is None or gtp is None:
            out["pred"] = p; out["distance_m"] = None
        else:
            d = sum((a - b) ** 2 for a, b in zip(p, gtp)) ** 0.5
            out["pred"] = p; out["distance_m"] = d
            for thr in POSITION_ACC_THRESH_M:
                out[f"acc@{thr}m"] = 1.0 if d <= thr else 0.0
    elif qtype in ("time", "duration"):
        p = parse_minutes(pred_text)
        gtv = gt.get("minutes_ago") if qtype == "time" else gt.get("minutes")
        tol = TIME_TOL_MIN if qtype == "time" else DURATION_TOL_MIN
        if p is None or gtv is None:
            out["pred"] = p; out["abs_error_min"] = None
        else:
            out["pred"] = p; out["abs_error_min"] = abs(p - gtv)
            out["within_tol"] = 1.0 if abs(p - gtv) <= tol else 0.0
    elif qtype == "text":
        gta = str(gt.get("answer") or "")
        if judge is not None:
            try:
                out["llm_match"] = float(judge(gt.get("_question", ""), gta, pred_text))
            except Exception as e:
                out["llm_match"] = 0.0; out["judge_error"] = str(e)
        else:
            # transparent fallback: normalized substring overlap (flagged elsewhere)
            a, b = gta.lower().strip(), pred_text.lower().strip()
            out["llm_match"] = 1.0 if (a and (a in b or b in a)) else 0.0
    return out


def summarize(scored: list[dict[str, Any]], llm_judge_available: bool) -> dict[str, Any]:
    by = lambda t: [s for s in scored if s["type"] == t]
    m = {}
    # binary
    b = by("binary")
    m["binary_n"] = len(b)
    m["binary_accuracy"] = mean([s["correct"] for s in b]) if b else None
    # position
    p = by("position")
    dists = [s["distance_m"] for s in p if s.get("distance_m") is not None]
    m["position_n"] = len(p)
    m["position_answered"] = len(dists)
    m["position_mean_dist_m"] = mean(dists) if dists else None
    for thr in POSITION_ACC_THRESH_M:
        vals = [s.get(f"acc@{thr}m", 0.0) for s in p]
        m[f"position_acc@{thr}m"] = mean(vals) if p else None
    # time / duration
    for t in ("time", "duration"):
        g = by(t)
        errs = [s["abs_error_min"] for s in g if s.get("abs_error_min") is not None]
        m[f"{t}_n"] = len(g)
        m[f"{t}_mean_abs_error_min"] = mean(errs) if errs else None
        m[f"{t}_within_tol"] = mean([s.get("within_tol", 0.0) for s in g]) if g else None
    # text
    tx = by("text")
    m["text_n"] = len(tx)
    m["text_llm_match"] = mean([s.get("llm_match", 0.0) for s in tx]) if tx else None
    m["llm_judge_available"] = llm_judge_available
    # aggregate headline: mean of available per-type primary sub-scores in [0,1]
    parts = []
    if m["binary_accuracy"] is not None: parts.append(m["binary_accuracy"])
    if m.get("position_acc@1.0m") is not None: parts.append(m["position_acc@1.0m"])
    if m["time_within_tol"] is not None: parts.append(m["time_within_tol"])
    if m["duration_within_tol"] is not None: parts.append(m["duration_within_tol"])
    if m["text_llm_match"] is not None: parts.append(m["text_llm_match"])
    m["navqa_score"] = mean(parts) if parts else None
    return m


def evaluate_track4(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    mode: str = "tool_llm",
    output: Path | None = None,
    llm_command: str | None = None,
    judge: JudgeFn | None = None,
    max_tool_iterations: int = 5,
) -> dict[str, Any]:
    """Score one CODa sequence's OC-NaVQA questions via the per-query agent loop."""
    manifest, _cap = load_package(package_dir)
    questions = read_jsonl(benchmark_dir / QUESTIONS_FILE)
    gt_by = {str(a["question_id"]): a for a in read_jsonl(benchmark_dir / ANSWERS_FILE)}
    if mode != "tool_llm":
        return {"status": "error", "message": f"track4 supports mode=tool_llm (got {mode})"}
    if not llm_command:
        return {"status": "error", "message": "tool_llm mode requires --llm-command"}

    work_dir = (output.parent if output else benchmark_dir) / "tool_llm_traces"
    work_dir.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []
    latency_by_q: dict[str, float] = {}
    started_all = time.perf_counter()
    for q in questions:
        qid = str(q["question_id"]); qtype = q.get("type", "text")
        try:
            r = run_tool_llm_query(
                package_dir=package_dir, manifest=manifest,
                query={"query_id": qid, "question_id": qid,
                       "question": q.get("question"), "query": q.get("question"),
                       "episode_id": q.get("episode_id")},
                llm_command=llm_command, work_dir=work_dir,
                max_tool_iterations=max_tool_iterations, response_kind="answer")
        except Exception as exc:
            r = {"status": "error", "answer": "", "latency_seconds": 0.0, "message": str(exc)}
        if r.get("status") == "invalid":
            return {"status": "invalid", "reason_code": r.get("reason_code"), "message": r.get("message")}
        pred = str(r.get("answer") or "")
        latency_by_q[qid] = float(r.get("latency_seconds") or 0.0)
        gt = dict(gt_by.get(qid, {})); gt["_question"] = q.get("raw_question") or q.get("question") or ""
        scored.append({**score_item(qtype, gt, pred, judge), "question_id": qid})

    metrics = summarize(scored, llm_judge_available=judge is not None)
    metrics["mean_query_latency_ms"] = (mean(list(latency_by_q.values())) or 0.0) * 1000.0
    metrics["n"] = len(scored)
    result = {"status": "ok", "track": TRACK_KEY, "mode": mode,
              "metrics": metrics, "per_item": scored}
    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    return result
