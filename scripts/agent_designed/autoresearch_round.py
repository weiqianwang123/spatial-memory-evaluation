#!/usr/bin/env python
"""AutoResearch round controller — the git-driven keep/revert loop (Karpathy-style).

Run this AFTER each code change you make to the design. It:
  1. builds the memory for every dev scene (runs your build_memory.py),
  2. scores all packages on the FIXED dev tests (real Track 1/2/3 evaluators + judge),
  3. compares the loop_objective to the best-so-far committed version,
  4. if IMPROVED  -> `git commit` (your diff is recorded; best advances),
     if NOT       -> `git revert` to the last good commit (your change is rolled back),
  5. appends a row to history.jsonl and redraws progress.png
     (x = round/time, y = best-so-far loop_objective + per-track lines).

So the git log IS the experiment journal: every commit is a genuine improvement,
and reverts undo regressions automatically. Loop until your turn/time budget.

Usage (from inside the sandbox):
    python autoresearch_round.py --build-cmd "python starter/build_memory.py" \
        --message "round N: <what you changed>"

Notes:
- The sandbox must be a git repo (init_sandbox_git.py does this once).
- --build-cmd is run once per dev scene with --layout-dir/--scene-id/--out appended,
  OR, if your builder loops scenes itself, pass --build-once and it's run verbatim.
- Reads sandbox_config.json for repo_root, python, dev scenes, judge_command.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SANDBOX = Path(__file__).resolve().parent
CFG = json.loads((SANDBOX / "sandbox_config.json").read_text())
REPO_ROOT = Path(CFG["repo_root"])
PY = CFG.get("python", sys.executable)
DEV_SCENES = CFG.get("dev_scene_ids", [])
JUDGE = CFG.get("judge_command")
MEM_ROOT = SANDBOX / "memories"
HISTORY = SANDBOX / "history.jsonl"
PROGRESS_PNG = SANDBOX / "progress.png"


def _run(cmd: list[str] | str, cwd: Path = SANDBOX, shell: bool = False) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, shell=shell, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def _git(*args: str) -> tuple[int, str]:
    return _run(["git", *args])


def _best_objective_so_far() -> float | None:
    """Highest committed loop_objective recorded in history.jsonl (kept rows only)."""
    if not HISTORY.exists():
        return None
    best = None
    for line in HISTORY.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("decision") == "keep" and isinstance(row.get("loop_objective"), (int, float)):
            best = row["loop_objective"] if best is None else max(best, row["loop_objective"])
    return best


def _round_index() -> int:
    if not HISTORY.exists():
        return 0
    return sum(1 for l in HISTORY.read_text().splitlines() if l.strip())


def build_all_scenes(build_cmd: str, build_once: bool) -> dict:
    """Run the agent's builder for each dev scene; return per-scene build status."""
    status = {}
    if build_once:
        rc, out = _run(build_cmd, shell=True)
        status["__once__"] = {"rc": rc, "tail": out[-400:]}
        return status
    for scene in DEV_SCENES:
        layout = SANDBOX / "dev_scenes" / scene
        out_pkg = MEM_ROOT / "design" / scene
        out_pkg.mkdir(parents=True, exist_ok=True)
        cmd = f"{build_cmd} --layout-dir {layout} --scene-id {scene} --out {out_pkg}"
        rc, out = _run(cmd, shell=True)
        status[scene] = {"rc": rc, "pkg": str(out_pkg), "tail": out[-300:]}
    return status


def score_all_scenes() -> dict:
    """Score all dev scenes on the FIXED dev tests, CONCURRENTLY (one persistent
    per-scene agent each), and aggregate per-track + loop_objective + build cost.

    This is the agent SELF-eval path only (per-scene sessions); the main/held-out
    eval uses the independent-per-query evaluators, never this.
    """
    from importlib import import_module
    sys.path.insert(0, str(REPO_ROOT))
    SE = import_module("spatial_memory_evaluation.agent_designed.session_eval")
    dev_eval = import_module("spatial_memory_evaluation.agent_designed.dev_eval")
    PRIMARY = dev_eval.PRIMARY_METRIC
    PROX = dev_eval.PROXIMITY_METRIC
    ANSWER_MODEL = CFG.get("answer_model", SE.DEFAULT_ANSWER_MODEL)

    judge_factory = None
    if JUDGE:
        mk = import_module("spatial_memory_evaluation.agent_designed.batch_judge").make_batch_cli_judge
        judge_factory = lambda: mk(JUDGE)  # FRESH judge per scene (per-scene cache)

    pkg_parent = MEM_ROOT / "design"
    tracks = list(PRIMARY.keys())
    per_scene = SE.score_all_scenes_concurrent(
        tracks=tracks, package_parent=pkg_parent,
        dev_tests_root=Path(CFG["dev_tests_root"]), dev_scene_ids=DEV_SCENES,
        answer_model=ANSWER_MODEL, judge_factory=judge_factory,
        work_root=pkg_parent / "_session", max_tool_iterations=int(CFG.get("max_tool_iterations", 1)),
    )

    # Aggregate to dev_eval's reported shape (per_track means + proximity + sum loop_objective).
    track_scores = {t: [] for t in PRIMARY}
    track_prox = {t: [] for t in PROX}
    per_eval = []
    for scene, by_track in per_scene.items():
        for track, summary in by_track.items():
            m = summary.get("metrics") if isinstance(summary, dict) else None
            s = (m or {}).get(PRIMARY[track]) if m else None
            px = (m or {}).get(PROX[track]) if (m and track in PROX) else None
            per_eval.append({"track": track, "scene": scene,
                             "status": summary.get("status"), "metric": s, "proximity": px})
            if isinstance(s, (int, float)):
                track_scores[track].append(float(s))
            if isinstance(px, (int, float)):
                track_prox[track].append(float(px))

    per_track, means = {}, []
    for t, vals in track_scores.items():
        if vals:
            entry = {"metric_key": PRIMARY[t], "mean": sum(vals) / len(vals), "n": len(vals)}
            pv = track_prox.get(t) or []
            if pv:
                entry["proximity_key"] = PROX[t]
                entry["proximity_mean"] = sum(pv) / len(pv)
            per_track[t] = entry
            means.append(entry["mean"])

    build_rows = [dev_eval._read_build_cost(pkg_parent / s, s) for s in DEV_SCENES]
    build_cost = dev_eval._aggregate_build_cost(build_rows)
    accuracy_sum = sum(means) if means else None
    # Weighted objective: accuracy primary minus a soft build-cost penalty (memory
    # bloat / loss of real-time). Same logic as dev_eval so loop + manual agree.
    penalty, breakdown = dev_eval._cost_penalty(build_cost)
    build_cost["cost_penalty"] = round(penalty, 4)
    build_cost["cost_penalty_breakdown"] = breakdown
    return {
        "status": "ok" if means else "no_dev_evals_ran",
        "per_track": per_track,
        "accuracy_sum": accuracy_sum,
        "loop_objective": (accuracy_sum - penalty) if accuracy_sum is not None else None,
        "per_eval": per_eval,
        "build_cost": build_cost,
    }


def append_history(row: dict) -> None:
    with HISTORY.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def redraw_progress() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [json.loads(l) for l in HISTORY.read_text().splitlines() if l.strip()] if HISTORY.exists() else []
    if not rows:
        return
    xs = list(range(len(rows)))
    objs = [r.get("loop_objective") or 0 for r in rows]
    best, cur = [], None
    for o in objs:
        cur = o if cur is None else max(cur, o)
        best.append(cur)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, objs, "o-", alpha=0.4, label="round loop_objective")
    ax.step(xs, best, "g-", where="post", lw=2, label="best-so-far")
    for track, style in (("track1_object_location", "C0--"),
                         ("track2_scanrefer", "C1--"),
                         ("track3_openeqa", "C3--")):
        ys = [(r.get("per_track", {}).get(track, {}) or {}).get("mean") for r in rows]
        if any(y is not None for y in ys):
            ax.plot(xs, [y if y is not None else float("nan") for y in ys], style,
                    alpha=0.6, label=track.split("_")[0])
    # mark kept (commit) vs reverted rounds
    for i, r in enumerate(rows):
        ax.annotate("●" if r.get("decision") == "keep" else "×",
                    (i, objs[i]), fontsize=8,
                    color="green" if r.get("decision") == "keep" else "red")
    ax.set_xlabel("round"); ax.set_ylabel("score")
    ax.set_title("AutoResearch progress (best-so-far = green step; ●keep ×revert)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PROGRESS_PNG, dpi=110); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="One AutoResearch keep/revert round.")
    ap.add_argument("--build-cmd", required=True, help="builder command (per-scene args appended unless --build-once)")
    ap.add_argument("--build-once", action="store_true", help="run --build-cmd verbatim (builder loops scenes itself)")
    ap.add_argument("--message", required=True, help="what you changed this round (becomes the commit message)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="min loop_objective gain to count as improvement")
    args = ap.parse_args()

    rnd = _round_index()
    best_before = _best_objective_so_far()

    print(f"[round {rnd}] building {len(DEV_SCENES)} dev scenes ...")
    build_status = build_all_scenes(args.build_cmd, args.build_once)
    print(f"[round {rnd}] scoring on fixed dev tests ...")
    scored = score_all_scenes()
    obj = scored.get("loop_objective")

    improved = (obj is not None) and (best_before is None or obj > best_before + args.min_delta)
    decision = "keep" if improved else "revert"

    if improved:
        _git("add", "-A")
        rc, out = _git("commit", "-m", f"[r{rnd}] {args.message} | loop_objective={obj:.4f}")
        commit = _git("rev-parse", "--short", "HEAD")[1].strip()
    else:
        # roll back uncommitted changes to the last good commit
        _git("reset", "--hard", "HEAD")
        _git("clean", "-fd", "memories", "starter")
        commit = _git("rev-parse", "--short", "HEAD")[1].strip()

    row = {
        "round": rnd,
        "message": args.message,
        "loop_objective": obj,
        "best_before": best_before,
        "decision": decision,
        "commit": commit,
        "per_track": scored.get("per_track", {}),
        "build_cost": {
            "mean_native_memory_size_bytes": scored.get("build_cost", {}).get("mean_native_memory_size_bytes"),
            "mean_time_per_frame_seconds": scored.get("build_cost", {}).get("mean_time_per_frame_seconds"),
        },
    }
    append_history(row)
    redraw_progress()

    print("=" * 64)
    print(f"[round {rnd}] loop_objective={obj}  best_before={best_before}  -> {decision.upper()} (commit {commit})")
    for t, info in (scored.get("per_track") or {}).items():
        extra = f"  [{info.get('proximity_key')}={info.get('proximity_mean'):.3f}]" if info.get("proximity_key") else ""
        print(f"    {t:26s} {info['metric_key']}={info['mean']:.3f}{extra}")
    print(f"history -> {HISTORY.name}   plot -> {PROGRESS_PNG.name}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
