#!/usr/bin/env python
"""Annotated run4 auto-design progress chart (English step labels).

Reads run4's history.jsonl and renders an annotated PNG:
  x = round, y = loop_objective (+ per-track T1/T2/T3), best-so-far step,
  each round labeled with a short English note on what the agent changed and
  whether it was kept (●) or reverted (×).

Usage: python plot_run4_progress.py <history.jsonl> <out.png>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Short English annotation per round (what the agent did).
NOTE = {
    0:  "v1: object-centric 3D map\n(YOLO-World + depth\nbackproject + qwen desc)",
    1:  "T2: spatial-relation\nre-ranker (above/near/\nbetween + anchors)",
    2:  "T1: synonym-expanded\nlabel coverage",
    3:  "T3: richer grounded\ncontext (neighbor labels)",
    4:  "caption from largest\nhi-conf crop",
    5:  "T3: 8 keyframes\n→ hurt, revert",
    6:  "T3: clean inventory\n→ hurt, revert",
    7:  "T3: clean inventory +\nroom-type inference",
    8:  "predict BBOX CENTER\n(matches GT) → T1+T2 up",
    9:  "T3: adaptive answer\nlength → revert",
    10: "T3 adaptive (retry)\n→ revert",
    11: "T1 synonym (over-broad)\n→ revert",
    12: "T1 synonym only\n→ revert (noise)",
    13: "T1 toilet-paper synonym\n(deterministic) → BEST",
    14: "T3 adaptive on top\n→ revert",
}


def main() -> int:
    hist = Path(sys.argv[1]); out = Path(sys.argv[2])
    rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
    xs = list(range(len(rows)))
    obj = [r.get("loop_objective") or 0 for r in rows]
    dec = [r.get("decision") for r in rows]

    def track(key):
        return [((r.get("per_track", {}).get(key, {}) or {}).get("mean")) for r in rows]
    t1 = track("track1_object_location"); t2 = track("track2_scanrefer"); t3 = track("track3_openeqa")

    best, cur = [], None
    for o in obj:
        cur = o if cur is None else max(cur, o)
        best.append(cur)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(xs, obj, "o-", color="#4C78A8", alpha=0.55, lw=1.5, label="round loop_objective")
    ax.step(xs, best, where="post", color="#2CA02C", lw=2.5, label="best-so-far")
    ax.plot(xs, t1, "--", color="#4C78A8", alpha=0.7, label="T1 success@1")
    ax.plot(xs, t2, "--", color="#F58518", alpha=0.7, label="T2 acc@0.5m")
    ax.plot(xs, t3, "--", color="#E45756", alpha=0.7, label="T3 llm_match")

    # keep/revert markers + per-round annotations
    for i, r in enumerate(rows):
        kept = dec[i] == "keep"
        ax.scatter([i], [obj[i]], s=90, zorder=5,
                   marker="o" if kept else "x",
                   color="#2CA02C" if kept else "#D62728")
        note = NOTE.get(i, "")
        # alternate annotation above/below to reduce overlap
        up = (i % 2 == 0)
        yoff = 0.055 if up else -0.11
        ax.annotate(f"r{i} {'keep' if kept else 'revert'}\n{note}",
                    (i, obj[i]), fontsize=6.6, ha="center",
                    va="bottom" if up else "top",
                    xytext=(i, obj[i] + yoff), color="#333",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="#eaffea" if kept else "#ffecec",
                              ec="#bbb", lw=0.5, alpha=0.9))

    ax.set_xlabel("auto-research round"); ax.set_ylabel("score")
    ax.set_title("MemForge — run4 auto-design progress (● keep, × revert; green = best-so-far)\n"
                 "obj 1.60 → 1.86; each round: read the metric, diagnose, change one thing, keep/revert on real evals",
                 fontsize=11)
    ax.set_ylim(0.30, 2.02)
    ax.set_xticks(xs)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
