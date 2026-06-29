"""Per-scene-SESSION self-eval using ONE persistent agent process per scene.

For the auto-design self-reference score, each scene gets ONE long-lived ``claude``
process (stream-json stdin, kept alive — see ``persistent_agent.py``); the scene's
queries are fed to it ONE AT A TIME. Only the first query pays process boot; the
rest are fast (no per-turn cold start). The three dev scenes' agents run
CONCURRENTLY (one thread each).

Within a query we run the SAME method-native tool loop as the main eval (render a
prompt describing the package's declared tools -> the agent replies with a
tool_call or a final answer -> we execute the tool and feed the observation back as
the next message), but all turns go to the live per-scene process instead of a
fresh CLI call each time. Scoring reuses the UNCHANGED Track 1/2/3 metric functions
(incl. relaxed proximity), so dev numbers stay on the benchmark scale.

Public API: ``score_scene_session(...)`` -> the per-track metrics dict the
evaluators emit. ``score_all_scenes_concurrent(...)`` runs the three scenes in
parallel and returns per-(track,scene) results.
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from spatial_memory_evaluation.common.jsonl import read_jsonl
from spatial_memory_evaluation.common.labels import load_aliases
from spatial_memory_evaluation.common.package_io import load_package
from spatial_memory_evaluation.tool_llm.native_tools import NativeToolExecutor
from spatial_memory_evaluation.tool_llm.runner import (
    _extract_final,
    _extract_tool_call,
    _load_json_value,
    _normalize_prediction,
    _prepare_tool_llm_sandbox,
    _render_prompt,
)

from .persistent_agent import PersistentAgent

# Reuse the EXACT scoring math from the unchanged evaluators.
from spatial_memory_evaluation.track1.evaluator import (
    SPLITS as T1_SPLITS,
    _score_split as t1_score_split,
    _track1_summary_metrics,
)
from spatial_memory_evaluation.track2.evaluator import (
    _scene_objects_by_id as t2_scene_objects_by_id,
    _score as t2_score,
    _summary_metrics as t2_summary_metrics,
)
from spatial_memory_evaluation.track2.data import REFERRING_QUERIES_FILE
from spatial_memory_evaluation.track3.evaluator import (
    _default_exact_judge,
    _score as t3_score,
    _summary_metrics as t3_summary_metrics,
)
from spatial_memory_evaluation.track3.data import ANSWERS_FILE, QUESTIONS_FILE

DEFAULT_ANSWER_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _run_one_query(
    *,
    agent: PersistentAgent,
    manifest: Mapping[str, Any],
    executor: NativeToolExecutor,
    tool_specs: list[dict[str, Any]],
    sandbox_context: Mapping[str, Any],
    query: Mapping[str, Any],
    response_kind: str,
    max_tool_iterations: int,
    scratch_dir: Path,
) -> dict[str, Any]:
    """One query through the live per-scene agent: render -> ask -> tool-loop."""
    observations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(max(0, max_tool_iterations) + 1):
        prompt = _render_prompt(
            manifest=manifest, query=query, tool_specs=tool_specs,
            observations=observations, sandbox_context=sandbox_context,
            max_tool_iterations=max_tool_iterations, response_kind=response_kind,
            final_step=(step >= max_tool_iterations),
        )
        raw = agent.ask(prompt)
        # parse JSON out of the assistant text (tolerant; same as the CLI path)
        out_path = scratch_dir / f"{_safe(query.get('query_id'))}_step{step}.json"
        try:
            scratch_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(raw, encoding="utf-8")
        except Exception:
            pass
        try:
            value = _load_json_value(raw, out_path)
        except Exception:
            value = {}
        final = _extract_final(value, response_kind=response_kind)
        if final is not None:
            if response_kind == "answer":
                ev = final.get("evidence")
                return {"status": "ok", "answer": str(final.get("answer") or ""),
                        "evidence": ev if isinstance(ev, list) else [],
                        "latency_seconds": time.perf_counter() - started}
            preds = final.get("predictions")
            preds = preds if isinstance(preds, list) else []
            return {"status": "ok",
                    "predictions": [_normalize_prediction(p) for p in preds],
                    "latency_seconds": time.perf_counter() - started}
        tool_call = _extract_tool_call(value)
        if tool_call is None or step >= max_tool_iterations:
            # no actionable output; return empty (scored as a miss, not a crash)
            empty = {"status": "ok", "latency_seconds": time.perf_counter() - started}
            return {**empty, "answer": ""} if response_kind == "answer" else {**empty, "predictions": []}
        name = str(tool_call.get("name") or "")
        args = tool_call.get("arguments")
        observation = executor.execute(name, args if isinstance(args, Mapping) else {})
        observations.append({"tool_call": {"name": name, "arguments": dict(args or {})},
                             "observation": observation})
    empty = {"status": "ok", "latency_seconds": time.perf_counter() - started}
    return {**empty, "answer": ""} if response_kind == "answer" else {**empty, "predictions": []}


def _safe(x: Any) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(x or "q"))[:80]


def score_scene_session(
    *,
    track: str,
    package_dir: Path,
    benchmark_dir: Path,
    scene_id: str,
    llm_command: str | None = None,   # kept for signature compat; ignored
    answer_model: str = DEFAULT_ANSWER_MODEL,
    work_dir: Path,
    judge: Callable[[str, str, str], float] | None = None,
    max_tool_iterations: int = 2,
    agent: PersistentAgent | None = None,
) -> dict[str, Any]:
    """Score one scene/track via a persistent per-scene agent; reuse evaluator math.

    If ``agent`` is given it is reused (so all tracks of a scene share ONE process);
    otherwise a process is launched for this call and closed at the end.
    """
    manifest, _cap = load_package(package_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = NativeToolExecutor(package_dir, manifest)
    tool_specs = executor.tool_specs()
    if not tool_specs:
        return {"status": "invalid", "reason_code": "no_method_native_llm_tools"}
    sandbox_context = _prepare_tool_llm_sandbox(
        package_dir=package_dir, manifest=manifest, tool_specs=tool_specs, work_dir=work_dir)

    own_agent = agent is None
    if own_agent:
        agent = PersistentAgent(model=answer_model, cwd=work_dir)
    try:
        return _score_track(
            track=track, agent=agent, manifest=manifest, executor=executor,
            tool_specs=tool_specs, sandbox_context=sandbox_context,
            benchmark_dir=benchmark_dir, judge=judge,
            max_tool_iterations=max_tool_iterations, work_dir=work_dir)
    finally:
        if own_agent and agent is not None:
            agent.close()


def _score_track(*, track, agent, manifest, executor, tool_specs, sandbox_context,
                 benchmark_dir, judge, max_tool_iterations, work_dir) -> dict[str, Any]:
    def run(queries, response_kind, mti=None):
        results, lat = {}, {}
        for q in queries:
            r = _run_one_query(
                agent=agent, manifest=manifest, executor=executor, tool_specs=tool_specs,
                sandbox_context=sandbox_context, query=q, response_kind=response_kind,
                max_tool_iterations=(mti or max_tool_iterations), scratch_dir=work_dir / "turns")
            results[str(q["query_id"])] = r
            lat[str(q["query_id"])] = float(r.get("latency_seconds") or 0.0)
        return results, lat

    if track == "track1_object_location":
        queries = [{"query_id": str(q["query_id"]), "query": q.get("query"),
                    "target_label": q.get("target_label"), "canonical_label": q.get("canonical_label"),
                    "top_k": int(q.get("top_k", 10))}
                   for q in read_jsonl(benchmark_dir / "queries_detector_coverable.jsonl")]
        results, lat = run(queries, "predictions")
        preds = {qid: (r.get("predictions") or []) for qid, r in results.items()}
        aliases = load_aliases(benchmark_dir / "label_aliases.json")
        splits = {s: t1_score_split(
            gt_objects=read_jsonl(benchmark_dir / f"{s}.jsonl"),
            queries=read_jsonl(benchmark_dir / f"queries_{s}.jsonl"),
            predictions_by_query=preds, latency_seconds_by_query=lat, aliases=aliases)
            for s in T1_SPLITS}
        return {"status": "ok", "metrics": _track1_summary_metrics(splits["detector_coverable"])}

    if track == "track2_scanrefer":
        raw = read_jsonl(benchmark_dir / REFERRING_QUERIES_FILE)
        queries = [{"query_id": str(q["query_id"]), "scene_id": q.get("scene_id"),
                    "query": q.get("utterance"), "utterance": q.get("utterance"),
                    "top_k": int(q.get("top_k", 10))} for q in raw]
        results, lat = run(queries, "predictions")
        preds = {qid: (r.get("predictions") or []) for qid, r in results.items()}
        metrics = t2_score(queries=raw, scene_objects=t2_scene_objects_by_id(benchmark_dir),
                           predictions_by_query=preds, latency_seconds_by_query=lat)
        return {"status": "ok", "metrics": t2_summary_metrics(metrics)}

    if track == "track3_openeqa":
        questions = read_jsonl(benchmark_dir / QUESTIONS_FILE)
        gt = {str(r["question_id"]): r for r in read_jsonl(benchmark_dir / ANSWERS_FILE)}
        queries = [{"query_id": str(q["question_id"]), "question": q.get("question"),
                    "episode_id": q.get("episode_id")} for q in questions]
        results, lat = run(queries, "answer", mti=max(max_tool_iterations, 3))
        answers = {qid: str(r.get("answer") or "") for qid, r in results.items()}
        evidence = {qid: r.get("evidence", []) for qid, r in results.items()}
        active_judge = judge or _default_exact_judge()
        if hasattr(active_judge, "prime"):
            triples = [(str(q.get("question") or ""),
                        str(gt.get(str(q["question_id"]), {}).get("answer") or ""),
                        answers.get(str(q["question_id"]), "")) for q in questions]
            try:
                active_judge.prime(triples)  # type: ignore[attr-defined]
            except Exception:
                pass
        metrics = t3_score(questions=questions, gt_answers=gt, answers_by_question=answers,
                          evidence_by_question=evidence, latency_seconds_by_question=lat,
                          judge=active_judge, llm_judge_available=judge is not None)
        return {"status": "ok", "metrics": t3_summary_metrics(metrics)}

    raise ValueError(f"unknown track: {track!r}")


def score_all_scenes_concurrent(
    *,
    tracks: list[str],
    package_parent: Path,
    dev_tests_root: Path,
    dev_scene_ids: list[str],
    answer_model: str = DEFAULT_ANSWER_MODEL,
    judge_factory: Callable[[], Callable[[str, str, str], float] | None] | None = None,
    work_root: Path,
    max_tool_iterations: int = 2,
) -> dict[str, dict[str, Any]]:
    """Run the dev scenes CONCURRENTLY: one persistent agent per scene, all tracks
    on that scene share its process. Returns {scene: {track: summary}}.

    ``judge_factory`` builds a FRESH judge per scene (judges hold per-scene caches),
    or None for the containment fallback.
    """
    def do_scene(scene: str) -> tuple[str, dict[str, Any]]:
        pkg = package_parent / scene
        wd = work_root / scene
        wd.mkdir(parents=True, exist_ok=True)
        out: dict[str, Any] = {}
        manifest, _ = load_package(pkg)
        executor = NativeToolExecutor(pkg, manifest)
        tool_specs = executor.tool_specs()
        if not tool_specs:
            return scene, {t: {"status": "invalid", "reason_code": "no_method_native_llm_tools"} for t in tracks}
        sandbox_context = _prepare_tool_llm_sandbox(
            package_dir=pkg, manifest=manifest, tool_specs=tool_specs, work_dir=wd)
        agent = PersistentAgent(model=answer_model, cwd=wd)
        try:
            for track in tracks:
                bench = dev_tests_root / track.split("_")[0] / scene
                exists = (bench / ("queries_detector_coverable.jsonl" if track == "track1_object_location"
                                   else REFERRING_QUERIES_FILE if track == "track2_scanrefer"
                                   else QUESTIONS_FILE)).exists()
                if not exists:
                    continue
                judge = judge_factory() if (judge_factory and track == "track3_openeqa") else None
                out[track] = _score_track(
                    track=track, agent=agent, manifest=manifest, executor=executor,
                    tool_specs=tool_specs, sandbox_context=sandbox_context,
                    benchmark_dir=bench, judge=judge,
                    max_tool_iterations=max_tool_iterations, work_dir=wd)
        finally:
            agent.close()
        return scene, out

    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dev_scene_ids)) as ex:
        futs = [ex.submit(do_scene, s) for s in dev_scene_ids]
        for f in concurrent.futures.as_completed(futs):
            scene, out = f.result()
            results[scene] = out
    return results
