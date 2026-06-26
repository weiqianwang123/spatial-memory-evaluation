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
PRIMARY_METRIC: dict[str, str] = {
    "track1_object_location": "success@5",
    "track2_scanrefer": "acc@0.5m",
    "track3_openeqa": "llm_match",
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
    loop_objective: float | None = None
    status: str = "ok"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "per_track": self.per_track,
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
    build_cost_rows: list[dict[str, Any]] = []

    for scene in dev_scene_ids:
        # Build cost (memory size + time-per-frame) from the scene's build_log.json
        # — mandatory secondary metrics reported alongside accuracy.
        build_cost_rows.append(_read_build_cost(package_dir, scene))

        # Track 1
        t1_dir = dev_tests_root / "track1" / scene
        if (t1_dir / "queries_detector_coverable.jsonl").exists():
            out = (output_root / f"track1-{scene}.json") if output_root else None
            summary = evaluate_track1(
                package_dir=package_dir, benchmark_dir=t1_dir, mode=mode,
                output=out, llm_command=llm_command,
            )
            s = _metric(summary, PRIMARY_METRIC["track1_object_location"])
            per_eval.append({"track": "track1_object_location", "scene": scene,
                             "status": summary.get("status"), "metric": s})
            if s is not None:
                track_scores["track1_object_location"].append(s)

        # Track 2
        t2_dir = dev_tests_root / "track2" / scene
        if (t2_dir / "referring_queries.jsonl").exists():
            out = (output_root / f"track2-{scene}.json") if output_root else None
            summary = evaluate_track2(
                package_dir=package_dir, benchmark_dir=t2_dir, mode=mode,
                output=out, llm_command=llm_command,
            )
            s = _metric(summary, PRIMARY_METRIC["track2_scanrefer"])
            per_eval.append({"track": "track2_scanrefer", "scene": scene,
                             "status": summary.get("status"), "metric": s})
            if s is not None:
                track_scores["track2_scanrefer"].append(s)

        # Track 3
        t3_dir = dev_tests_root / "track3" / scene
        if (t3_dir / "questions.jsonl").exists():
            out = (output_root / f"track3-{scene}.json") if output_root else None
            summary = evaluate_track3(
                package_dir=package_dir, benchmark_dir=t3_dir, mode=mode,
                output=out, llm_command=llm_command, judge=judge,
            )
            s = _metric(summary, PRIMARY_METRIC["track3_openeqa"])
            per_eval.append({"track": "track3_openeqa", "scene": scene,
                             "status": summary.get("status"), "metric": s})
            if s is not None:
                track_scores["track3_openeqa"].append(s)

    per_track: dict[str, Any] = {}
    track_means: list[float] = []
    for track, scores in track_scores.items():
        if scores:
            mean = sum(scores) / len(scores)
            per_track[track] = {"metric_key": PRIMARY_METRIC[track], "mean": mean, "n": len(scores)}
            track_means.append(mean)

    # Breadth-friendly loop target: SUM (not mean) of per-track means, so adding a
    # track can only raise it -> the loop is rewarded for attempting more tracks.
    # Reported output stays per-track; this is just the scalar the loop maximizes.
    loop_objective = sum(track_means) if track_means else None
    return DevEvalResult(
        per_track=per_track, per_eval=per_eval,
        build_cost=_aggregate_build_cost(build_cost_rows),
        loop_objective=loop_objective,
        status="ok" if track_means else "no_dev_evals_ran",
    )
