"""Track 3: OpenEQA general spatial QA (ScanNet + HM3D).

Evaluates open-ended spatial QA. Supports ``fixed_api`` (package declares a native
``answer_question`` entrypoint) and ``tool_llm`` (per-query LLM + method-native
QA/retrieval tools). Answers are scored by an LLM-Match judge that must be a
different model/config from the memory-construction and answering LLMs.

Status: skeleton. Control flow, capability gating, invalid/data_unavailable
results, and answer-collection are implemented. The LLM judge is pluggable: when
no judge command is provided it falls back to a transparent normalized
exact/substring match and flags ``llm_judge_available=false`` so scores are not
mistaken for real LLM-Match.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.labels import normalize_text
from spatial_memory_evaluation.common.matching import mean, safe_div
from spatial_memory_evaluation.common.package_io import (
    fixed_api_capability,
    invalid_result,
    load_entrypoint,
    load_package,
)
from spatial_memory_evaluation.common.reporting import (
    evaluation_output_paths,
    render_evaluation_report,
    report_metadata,
    report_title,
    write_evaluation_outputs,
)
from spatial_memory_evaluation.output_paths import timestamped_result_dir
from spatial_memory_evaluation.tool_llm import run_tool_llm_query

from .data import ANSWERS_FILE, QUESTIONS_FILE, track3_data_status


TRACK_KEY = "track3_openeqa"

# A judge maps (question, gt_answer, predicted_answer) -> score in [0, 1].
JudgeFn = Callable[[str, str, str], float]


def evaluate_track3(
    *,
    package_dir: Path,
    benchmark_dir: Path,
    mode: str,
    output: Path | None,
    llm_command: str | None = None,
    max_tool_iterations: int = 5,
    judge: JudgeFn | None = None,
) -> dict[str, Any]:
    manifest, capabilities = load_package(package_dir)
    method = str(manifest["method"]["name"])
    if output is None:
        output = timestamped_result_dir(method, f"track3-{mode}") / "eval_summary.json"

    base_summary: dict[str, Any] = {
        "status": "ok",
        "track": TRACK_KEY,
        "mode": mode,
        "package_dir": str(package_dir),
        "method": method,
        "dataset": manifest.get("dataset"),
        "explicit_memory": manifest.get("explicit_memory"),
    }

    data_status = track3_data_status(benchmark_dir)
    if data_status.get("status") != "ok":
        result = {"status": "data_unavailable", **data_status}
        return _finalize(base_summary, result, output)

    if mode == "fixed_api":
        result = _run_fixed_api(package_dir, manifest, capabilities, benchmark_dir, method)
    elif mode == "tool_llm":
        result = _run_tool_llm(
            package_dir=package_dir,
            manifest=manifest,
            benchmark_dir=benchmark_dir,
            output=output,
            llm_command=llm_command,
            max_tool_iterations=max_tool_iterations,
        )
    else:
        raise ValueError(f"unknown Track 3 mode: {mode}")

    if result["status"] == "ok":
        questions = read_jsonl(benchmark_dir / QUESTIONS_FILE)
        gt_answers = {str(row["question_id"]): row for row in read_jsonl(benchmark_dir / ANSWERS_FILE)}
        judge_fn = judge or _default_exact_judge()
        metrics = _score(
            questions=questions,
            gt_answers=gt_answers,
            answers_by_question=result["answers_by_question"],
            evidence_by_question=result.get("evidence_by_question", {}),
            latency_seconds_by_question=result.get("latency_seconds_by_question", {}),
            judge=judge_fn,
            llm_judge_available=judge is not None,
        )
        summary = {**base_summary, "metrics": _summary_metrics(metrics)}
        details = {**base_summary, "metrics": metrics}
        if result.get("tool_traces_by_question") is not None:
            details["tool_traces_by_question"] = result["tool_traces_by_question"]
    else:
        summary = {**base_summary, "status": result["status"], "result": result}
        details = dict(summary)

    return _finalize(base_summary, result, output, summary=summary, details=details)


def _finalize(
    base_summary: dict[str, Any],
    result: dict[str, Any],
    output: Path,
    *,
    summary: dict[str, Any] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if summary is None:
        summary = {**base_summary, "status": result["status"], "result": result}
    if details is None:
        details = dict(summary)
    paths = evaluation_output_paths(output)
    report = render_evaluation_report(
        title=report_title(summary),
        metadata=report_metadata(summary),
        metrics=summary.get("metrics") if isinstance(summary.get("metrics"), dict) else None,
        status=str(summary["status"]),
        summary_path=paths.summary,
        details_path=paths.details,
        result=result if result.get("status") != "ok" else None,
    )
    write_evaluation_outputs(summary_path=output, summary=summary, details=details, report_markdown=report)
    return summary


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "question_count",
        "llm_match",
        "llm_judge_available",
        "mean_query_latency_ms",
        "answered_rate",
    )
    return {key: metrics.get(key) for key in keys}


def _run_fixed_api(
    package_dir: Path,
    manifest: dict[str, Any],
    capabilities: dict[str, Any],
    benchmark_dir: Path,
    method: str,
) -> dict[str, Any]:
    cap = fixed_api_capability(capabilities, TRACK_KEY)
    if cap.get("status") != "supported":
        method_meta = manifest.get("method") if isinstance(manifest.get("method"), dict) else {}
        return invalid_result(
            method=method,
            package_dir=package_dir,
            track_key=TRACK_KEY,
            reason=str(cap.get("reason") or ""),
            explicit_memory=manifest.get("explicit_memory"),
            method_family=method_meta.get("family"),
        )
    answer_question = load_entrypoint(package_dir, str(cap["entrypoint"]))
    answers_by_question: dict[str, str] = {}
    evidence_by_question: dict[str, Any] = {}
    latency_seconds_by_question: dict[str, float] = {}
    for question in read_jsonl(benchmark_dir / QUESTIONS_FILE):
        question_id = str(question["question_id"])
        started = time.perf_counter()
        result = answer_question(
            str(package_dir),
            {
                "question_id": question_id,
                "question": question.get("question"),
                "episode_id": question.get("episode_id"),
            },
        )
        latency_seconds_by_question[question_id] = time.perf_counter() - started
        if isinstance(result, dict):
            answers_by_question[question_id] = str(result.get("answer") or "")
            evidence_by_question[question_id] = result.get("evidence", [])
        else:
            answers_by_question[question_id] = ""
    return {
        "status": "ok",
        "answers_by_question": answers_by_question,
        "evidence_by_question": evidence_by_question,
        "latency_seconds_by_question": latency_seconds_by_question,
    }


def _run_tool_llm(
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    benchmark_dir: Path,
    output: Path,
    llm_command: str | None,
    max_tool_iterations: int,
) -> dict[str, Any]:
    if not llm_command:
        return {"status": "error", "message": "tool_llm mode requires --llm-command"}
    work_dir = output.parent / "tool_llm_traces"
    work_dir.mkdir(parents=True, exist_ok=True)
    answers_by_question: dict[str, str] = {}
    evidence_by_question: dict[str, Any] = {}
    latency_seconds_by_question: dict[str, float] = {}
    tool_traces_by_question: dict[str, Any] = {}

    for question in read_jsonl(benchmark_dir / QUESTIONS_FILE):
        question_id = str(question["question_id"])
        try:
            query_result = run_tool_llm_query(
                package_dir=package_dir,
                manifest=manifest,
                query={
                    "query_id": question_id,
                    "question_id": question_id,
                    "question": question.get("question"),
                    "query": question.get("question"),
                    "episode_id": question.get("episode_id"),
                },
                llm_command=llm_command,
                work_dir=work_dir,
                max_tool_iterations=max_tool_iterations,
                response_kind="answer",
            )
        except Exception as exc:
            query_result = {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "latency_seconds": 0.0,
                "trace": [],
            }
        if query_result.get("status") == "invalid":
            return {
                "status": "invalid",
                "reason_code": query_result.get("reason_code"),
                "message": query_result.get("message"),
            }
        answers_by_question[question_id] = str(query_result.get("answer") or "")
        evidence_by_question[question_id] = query_result.get("evidence", [])
        latency_seconds_by_question[question_id] = float(query_result.get("latency_seconds") or 0.0)
        tool_traces_by_question[question_id] = query_result.get("trace", [])

    return {
        "status": "ok",
        "answers_by_question": answers_by_question,
        "evidence_by_question": evidence_by_question,
        "latency_seconds_by_question": latency_seconds_by_question,
        "tool_traces_by_question": tool_traces_by_question,
    }


def _score(
    *,
    questions: list[dict[str, Any]],
    gt_answers: dict[str, dict[str, Any]],
    answers_by_question: dict[str, str],
    evidence_by_question: dict[str, Any],
    latency_seconds_by_question: dict[str, float],
    judge: JudgeFn,
    llm_judge_available: bool,
) -> dict[str, Any]:
    scores: list[float] = []
    answered = 0
    per_category: dict[str, list[float]] = {}
    per_question = []

    for question in questions:
        question_id = str(question["question_id"])
        category = str(question.get("category") or "uncategorized")
        gt = gt_answers.get(question_id, {})
        gt_answer = str(gt.get("answer") or "")
        predicted = str(answers_by_question.get(question_id) or "")
        if predicted.strip():
            answered += 1
        score = judge(str(question.get("question") or ""), gt_answer, predicted)
        scores.append(score)
        per_category.setdefault(category, []).append(score)
        per_question.append(
            {
                "question_id": question_id,
                "category": category,
                "score": score,
                "has_evidence": bool(evidence_by_question.get(question_id)),
                "latency_ms": latency_seconds_by_question.get(question_id, 0.0) * 1000.0,
            }
        )

    latencies = [latency_seconds_by_question.get(str(q["question_id"]), 0.0) for q in questions]
    return {
        "question_count": len(questions),
        "llm_match": mean(scores),
        "llm_judge_available": llm_judge_available,
        "answered_rate": safe_div(answered, len(questions)),
        "mean_query_latency_ms": (mean(latencies) or 0.0) * 1000.0,
        "by_category": {cat: mean(cat_scores) for cat, cat_scores in per_category.items()},
        "per_question": per_question,
    }


def _default_exact_judge() -> JudgeFn:
    """Transparent fallback judge used only when no LLM judge is configured.

    Returns 1.0 when the GT answer and the prediction normalize to a containment
    match, else 0.0. This is NOT LLM-Match; the evaluator flags
    ``llm_judge_available=false`` so the number is not over-interpreted.
    """

    def _judge(_question: str, gt_answer: str, predicted: str) -> float:
        gt = normalize_text(gt_answer)
        pred = normalize_text(predicted)
        if not gt or not pred:
            return 0.0
        return 1.0 if (gt in pred or pred in gt) else 0.0

    return _judge
