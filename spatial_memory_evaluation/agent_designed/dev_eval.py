"""Score a designed package on the agent's OWN dev test cases.

The self-improvement loop optimizes against the designer's dev tests, scored by the
UNCHANGED Track 1/2/3 evaluators (so dev numbers are on the same scale as the final
held-out report). This module is the thin adapter: given a built package and a set
of dev benchmark dirs (laid out exactly like the real benchmarks), it runs each
track's evaluator and reduces the per-track primary metrics to one scalar the loop
maximizes.

Dev benchmark layout (authored by the designer in auto_research, or harness-seeded
in loop_fixed_tests), per dev scene:
    <dev_tests>/track1/<scene>/{detector_coverable.jsonl, queries_detector_coverable.jsonl, ...}
    <dev_tests>/track2/<scene>/{referring_queries.jsonl, scene_objects.jsonl, ...}
    <dev_tests>/track3/<scene>/{questions.jsonl, answers.jsonl, ...}

These are the same files ``build_track{1,2,3}_data.py`` emit, so the designer
generates them with the provided GT-derivation tooling on the DEV scenes only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from spatial_memory_evaluation.track1.evaluator import evaluate_track1
from spatial_memory_evaluation.track2.evaluator import evaluate_track2
from spatial_memory_evaluation.track3.evaluator import evaluate_track3


# The primary metric per track that the loop optimizes (each roughly in [0, 1]).
# T1 uses success@1 (top-1 prediction must be near GT) rather than success@5: @5
# is gamed by returning many objects (spatial recall), since each query passes
# target_label so the label-match is free — see the gaming caveat in the prompt.
PRIMARY_METRIC: dict[str, str] = {
    "track1_object_location": "success@1",
    "track2_scanrefer": "acc@0.5m",
    "track3_openeqa": "llm_match",
}

# Build-cost budget for the weighted loop objective. Accuracy is primary; we only
# want to discourage memory bloat and loss of real-time. A SOFT hinge: zero penalty
# while within budget, growing linearly above it. Budgets are generous (the current
# design is ~2.2MB / ~0.7s/frame, well under), so a normal compact/fast build pays
# nothing — only regressions (bigger memory or slower builds) are penalized.
MEMORY_BUDGET_BYTES = 50_000_000      # 50 MB/scene: ample for an object map; bloat beyond pays
# REAL-TIME budget: a memory that can't keep up with the camera is not deployable on
# a robot. Real-time perception references on this layout: DAAAM ~0.012 s/frame,
# ClawS ~0.095 s/frame. We set the budget at 0.2 s/frame (~5 fps, generous for
# "real-time") and penalize HARD above it, so the loop must keep builds near-real-
# time rather than buying coverage with a slow per-frame pipeline (the current
# detect-every-frame + caption design runs ~0.4-0.7 s/frame = NOT real-time, and
# should now feel pressure to speed up: subsample frames, lighter describe, etc.).
TPF_BUDGET_SECONDS = 0.2              # 0.2 s/frame (~5 fps) = real-time floor
# Weights. Memory stays a mild LINEAR penalty. TPF is a SATURATING penalty: it ramps
# up steeply once you exceed real-time (so the loop genuinely feels pressure to be
# real-time) but CAPS at MAX_TPF_PENALTY so it can never dominate accuracy_sum
# (~1.8-2.0). Net: accuracy stays primary, but a non-real-time build forfeits up to
# ~0.5 objective points — meaningful (≈ one track's typical gain), not crushing.
COST_WEIGHT_MEMORY = 0.20             # -0.20 per (1x budget) of memory over 50MB/scene
COST_WEIGHT_TPF = 0.50               # TPF penalty slope (per 1x over budget, before cap)
MAX_TPF_PENALTY = 0.50               # hard cap: losing real-time costs at most 0.5 pts


def _cost_penalty(build_cost: dict) -> tuple[float, dict]:
    """Build-cost penalty (>=0), subtracted from loop_objective. Zero within budget.

    Memory: mild linear hinge above MEMORY_BUDGET. TPF: saturating hinge above the
    real-time TPF_BUDGET, capped at MAX_TPF_PENALTY so speed pressure is strong but
    never overwhelms accuracy.
    """
    mem = build_cost.get("mean_native_memory_size_bytes")
    tpf = build_cost.get("mean_time_per_frame_seconds")
    mem_over = max(0.0, (mem / MEMORY_BUDGET_BYTES) - 1.0) if isinstance(mem, (int, float)) and mem else 0.0
    tpf_over = max(0.0, (tpf / TPF_BUDGET_SECONDS) - 1.0) if isinstance(tpf, (int, float)) and tpf else 0.0
    pen_mem = COST_WEIGHT_MEMORY * mem_over
    pen_tpf = min(MAX_TPF_PENALTY, COST_WEIGHT_TPF * tpf_over)
    return pen_mem + pen_tpf, {
        "memory_over_budget_ratio": round(mem_over, 3),
        "tpf_over_budget_ratio": round(tpf_over, 3),
        "penalty_memory": round(pen_mem, 4),
        "penalty_tpf": round(pen_tpf, 4),
        "tpf_penalty_capped": pen_tpf >= MAX_TPF_PENALTY,
    }


# Secondary RELAXED-localization (proximity) metric per track — reported alongside
# the strict primary so caption/coarse memories still show partial credit. T1's
# top-1 nearest-prediction proximity, T2's top-1 center-distance proximity, both at
# the 3 m threshold. Track 3 has no spatial proximity notion.
PROXIMITY_METRIC: dict[str, str] = {
    "track1_object_location": "proximity_top1@3.0m",
    "track2_scanrefer": "proximity@3.0m",
}


@dataclass
class DevEvalResult:
    """Per-track dev results — reported separately, NOT collapsed to one headline.

    The reported signal is ``per_track`` (T1/T2/T3 each scored on its own scale) +
    ``build_cost``. We deliberately do NOT publish a single mean: averaging over
    "supported" tracks penalizes breadth (a weak third track drags the headline
    below a strong two-track design), which pushes a score-maximizer to DROP tracks
    rather than attempt them. ``loop_objective`` is a breadth-friendly target the
    self-improvement loop maximizes (sum of per-track means): adding a track can
    only ever raise it, so the loop is rewarded for attempting more tracks well.
    """

    per_track: dict[str, Any] = field(default_factory=dict)
    per_eval: list[dict[str, Any]] = field(default_factory=list)
    build_cost: dict[str, Any] = field(default_factory=dict)
    accuracy_sum: float | None = None
    loop_objective: float | None = None
    status: str = "ok"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "per_track": self.per_track,
            "accuracy_sum": self.accuracy_sum,
            "loop_objective": self.loop_objective,
            "per_eval": self.per_eval,
            "build_cost": self.build_cost,
        }


def _metric(summary: dict[str, Any], key: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _prime_batch_judge(judge: Callable, package_dir: Path, t3_dir: Path) -> None:
    """Collect (question, gt, predicted) for a scene and prime a batched judge.

    Runs the package's answer_question once per question (local qwen, cheap) so the
    batched judge can score the whole scene in ONE LLM call. Best-effort: any error
    leaves the judge un-primed and it falls back to per-item judging.
    """

    from spatial_memory_evaluation.common.jsonl import read_jsonl
    from spatial_memory_evaluation.common.package_io import (
        fixed_api_capability,
        load_entrypoint,
        load_package,
    )

    try:
        manifest, capabilities = load_package(package_dir)
        cap = fixed_api_capability(capabilities, "track3_openeqa")
        if cap.get("status") != "supported":
            return
        answer_question = load_entrypoint(package_dir, str(cap["entrypoint"]))
        questions = read_jsonl(t3_dir / "questions.jsonl")
        answers = {str(a["question_id"]): str(a.get("answer") or "")
                   for a in read_jsonl(t3_dir / "answers.jsonl")}
        triples = []
        for q in questions:
            qid = str(q["question_id"])
            try:
                res = answer_question(str(package_dir), {
                    "question_id": qid, "question": q.get("question"),
                    "episode_id": q.get("episode_id")})
                pred = str(res.get("answer") or "") if isinstance(res, dict) else ""
            except Exception:
                pred = ""
            triples.append((str(q.get("question") or ""), answers.get(qid, ""), pred))
        judge.prime(triples)  # type: ignore[attr-defined]
    except Exception:
        return  # un-primed -> evaluator falls back to per-item judge calls


def _read_build_cost(package_dir: Path, scene: str) -> dict[str, Any]:
    """Read memory size + time-per-frame from a scene's package build_log.json.

    A package is built per dev scene under <pkg parent>/<scene>/; we accept either
    ``package_dir`` itself (single scene) or a sibling ``<scene>`` dir, so the
    scorer surfaces build cost the same way the main eval records it (mandatory
    secondary metrics alongside accuracy).
    """

    candidates = [package_dir, package_dir.parent / scene, package_dir / scene]
    for cand in candidates:
        bl = cand / "build_log.json"
        if not bl.exists():
            continue
        try:
            log = json.loads(bl.read_text())
        except (OSError, ValueError):
            continue
        return {
            "scene": scene,
            "package_dir": str(cand),
            "native_memory_size_bytes": log.get("native_memory_size_bytes"),
            "package_size_bytes": log.get("package_size_bytes"),
            "frame_count": log.get("frame_count"),
            "build_runtime_seconds": log.get("build_runtime_seconds"),
            "time_per_frame_seconds": log.get("time_per_frame_seconds"),
            "peak_ram_bytes": log.get("peak_ram_bytes"),
            "peak_vram_bytes": log.get("peak_vram_bytes"),
        }
    return {"scene": scene, "package_dir": None, "note": "no build_log.json found"}


def _aggregate_build_cost(per_scene: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str) -> float | None:
        vals = [r[key] for r in per_scene if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "per_scene": per_scene,
        "mean_native_memory_size_bytes": _mean("native_memory_size_bytes"),
        "mean_time_per_frame_seconds": _mean("time_per_frame_seconds"),
        "total_frame_count": sum(r["frame_count"] for r in per_scene if isinstance(r.get("frame_count"), int)),
    }


_TRACK_EVALUATOR = {
    "track1_object_location": evaluate_track1,
    "track2_scanrefer": evaluate_track2,
    "track3_openeqa": evaluate_track3,
}


def _eval_track(
    track: str, package_dir: Path, bench_dir: Path, scene: str, mode: str,
    llm_command: str | None, judge: Callable | None, output_root: Path | None,
) -> dict[str, Any]:
    """Score one track/scene, routing by mode. Returns a summary with 'metrics'.

    - per_scene_session: ONE agent session per scene, queries one at a time
      (the auto-design self-eval) -> session_eval.score_scene_session.
    - fixed_api / tool_llm: the unchanged Track 1/2/3 evaluators.
    Proximity + judge-priming behavior is preserved in both paths.
    """

    tkey = track.split("_")[0]  # track1/2/3 for filenames
    out = (output_root / f"{tkey}-{scene}.json") if output_root else None

    if mode == "per_scene_session":
        from .session_eval import score_scene_session
        if llm_command is None:
            return {"status": "error", "message": "per_scene_session requires --llm-command"}
        work_dir = (output_root or bench_dir.parent) / "_session" / f"{tkey}-{scene}"
        return score_scene_session(
            track=track, package_dir=package_dir, benchmark_dir=bench_dir,
            scene_id=scene, llm_command=llm_command, work_dir=work_dir, judge=judge)

    # fixed_api / tool_llm: unchanged evaluators.
    if track == "track3_openeqa":
        scene_judge = judge
        # batched/session judge: prime once per scene in fixed_api mode.
        if judge is not None and hasattr(judge, "prime") and mode == "fixed_api":
            _prime_batch_judge(judge, package_dir, bench_dir)
        return evaluate_track3(
            package_dir=package_dir, benchmark_dir=bench_dir, mode=mode,
            output=out, llm_command=llm_command, judge=scene_judge)
    return _TRACK_EVALUATOR[track](
        package_dir=package_dir, benchmark_dir=bench_dir, mode=mode,
        output=out, llm_command=llm_command)


def evaluate_dev(
    *,
    package_dir: Path,
    dev_tests_root: Path,
    dev_scene_ids: list[str],
    mode: str = "fixed_api",
    llm_command: str | None = None,
    judge: Callable[[str, str, str], float] | None = None,
    output_root: Path | None = None,
) -> DevEvalResult:
    """Run the agent's dev tests for each track/scene and reduce to a dev score.

    Missing track/scene dirs are skipped (the designer may not support every
    track). The dev score is the mean of the per-track mean primary metrics over
    whatever dev evals actually ran; ``None`` if nothing ran.
    """

    per_eval: list[dict[str, Any]] = []
    track_scores: dict[str, list[float]] = {t: [] for t in PRIMARY_METRIC}
    track_prox: dict[str, list[float]] = {t: [] for t in PROXIMITY_METRIC}
    build_cost_rows: list[dict[str, Any]] = []

    for scene in dev_scene_ids:
        # Build cost (memory size + time-per-frame) from the scene's build_log.json
        # — mandatory secondary metrics reported alongside accuracy.
        build_cost_rows.append(_read_build_cost(package_dir, scene))

        # Track 1
        t1_dir = dev_tests_root / "track1" / scene
        if (t1_dir / "queries_detector_coverable.jsonl").exists():
            summary = _eval_track(
                "track1_object_location", package_dir, t1_dir, scene, mode,
                llm_command, judge, output_root)
            s = _metric(summary, PRIMARY_METRIC["track1_object_location"])
            px = _metric(summary, PROXIMITY_METRIC["track1_object_location"])
            per_eval.append({"track": "track1_object_location", "scene": scene,
                             "status": summary.get("status"), "metric": s, "proximity": px})
            if s is not None:
                track_scores["track1_object_location"].append(s)
            if px is not None:
                track_prox["track1_object_location"].append(px)

        # Track 2
        t2_dir = dev_tests_root / "track2" / scene
        if (t2_dir / "referring_queries.jsonl").exists():
            summary = _eval_track(
                "track2_scanrefer", package_dir, t2_dir, scene, mode,
                llm_command, judge, output_root)
            s = _metric(summary, PRIMARY_METRIC["track2_scanrefer"])
            px = _metric(summary, PROXIMITY_METRIC["track2_scanrefer"])
            per_eval.append({"track": "track2_scanrefer", "scene": scene,
                             "status": summary.get("status"), "metric": s, "proximity": px})
            if s is not None:
                track_scores["track2_scanrefer"].append(s)
            if px is not None:
                track_prox["track2_scanrefer"].append(px)

        # Track 3
        t3_dir = dev_tests_root / "track3" / scene
        if (t3_dir / "questions.jsonl").exists():
            summary = _eval_track(
                "track3_openeqa", package_dir, t3_dir, scene, mode,
                llm_command, judge, output_root)
            s = _metric(summary, PRIMARY_METRIC["track3_openeqa"])
            per_eval.append({"track": "track3_openeqa", "scene": scene,
                             "status": summary.get("status"), "metric": s, "proximity": None})
            if s is not None:
                track_scores["track3_openeqa"].append(s)

    per_track: dict[str, Any] = {}
    track_means: list[float] = []
    for track, scores in track_scores.items():
        if scores:
            mean = sum(scores) / len(scores)
            entry = {"metric_key": PRIMARY_METRIC[track], "mean": mean, "n": len(scores)}
            prox = track_prox.get(track) or []
            if prox:
                entry["proximity_key"] = PROXIMITY_METRIC[track]
                entry["proximity_mean"] = sum(prox) / len(prox)
            per_track[track] = entry
            track_means.append(mean)

    # Breadth-friendly accuracy term: SUM (not mean) of per-track means, so adding a
    # track can only raise it -> the loop is rewarded for attempting more tracks.
    accuracy_sum = sum(track_means) if track_means else None
    build_cost = _aggregate_build_cost(build_cost_rows)
    # Weighted objective: accuracy is primary; subtract a soft build-cost penalty so
    # the loop won't bloat memory or lose real-time for a tiny accuracy gain. Within
    # budget the penalty is 0 (objective == accuracy_sum).
    penalty, penalty_breakdown = _cost_penalty(build_cost)
    loop_objective = (accuracy_sum - penalty) if accuracy_sum is not None else None
    build_cost["cost_penalty"] = round(penalty, 4)
    build_cost["cost_penalty_breakdown"] = penalty_breakdown
    return DevEvalResult(
        per_track=per_track, per_eval=per_eval,
        build_cost=build_cost,
        accuracy_sum=accuracy_sum,
        loop_objective=loop_objective,
        status="ok" if track_means else "no_dev_evals_ran",
    )
