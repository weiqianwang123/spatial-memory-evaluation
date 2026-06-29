#!/usr/bin/env python
"""Annotated AutoResearch progress chart: each point labeled with what the agent did.

Reads a run's history.jsonl (+ optional extra rounds from git commit messages) and
renders progress_annotated.png: x = round, y = loop_objective and per-track means,
with a Chinese annotation per point describing the agent's change and keep/revert.

Usage: python plot_progress_annotated.py <history.jsonl> <out.png>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Per-round Chinese summary of what the agent did (keyed by the round message prefix).
ANNOT = {
    "r0": "v1基线\n3D物体图+qwen描述",
    "r1": "T1描述一致性加分\n+T2外观/嵌入排序",
    "r3_revert": "重启T2颜色排序\n→T3下降,回退",
    "r4": "T2属性=4\n+更丰富T3帧描述",
    "r5": "扩大大场景\nT3帧描述覆盖",
}


def _cn_font():
    # Prefer registering a known CJK font file directly (most reliable).
    import glob
    candidates = []
    for pat in ["/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
                "/usr/share/fonts/opentype/noto/NotoSerifCJK*.ttc",
                "/usr/share/fonts/truetype/wqy/*.ttc",
                "/usr/share/fonts/**/*CJK*.tt?", "/usr/share/fonts/**/wqy*.tt?"]:
        candidates += glob.glob(pat, recursive=True)
    for f in candidates:
        try:
            font_manager.fontManager.addfont(f)
            name = font_manager.FontProperties(fname=f).get_name()
            # sanity: must resolve to itself, not fall back
            return name
        except Exception:
            continue
    return None


def main() -> int:
    hist = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else hist.parent / "progress_annotated.png"
    rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]

    cn = _cn_font()
    if cn:
        plt.rcParams["font.sans-serif"] = [cn]
        plt.rcParams["axes.unicode_minus"] = False
    use_cn = cn is not None

    # English fallback labels if no CJK font
    annot_en = {
        "r0": "v1 baseline\n3D obj-map + qwen desc",
        "r1": "T1 desc-consistency\n+T2 appearance/embed rank",
        "r3_revert": "re-enable T2 color rank\n-> T3 down, REVERT",
        "r4": "T2 attr=4\n+richer T3 captions",
        "r5": "widen T3 caption\ncoverage (large scenes)",
    }
    A = ANNOT if use_cn else annot_en

    xs, objs, t1, t2, t3, labels, decisions, keys = [], [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        pt = r.get("per_track", {})
        g = lambda k: (pt.get(k, {}) or {}).get("mean")
        xs.append(i)
        objs.append(r.get("loop_objective") or 0)
        t1.append(g("track1_object_location"))
        t2.append(g("track2_scanrefer"))
        t3.append(g("track3_openeqa"))
        decisions.append(r.get("decision"))
        msg = r.get("message", "")
        key = "r0" if msg.startswith("r0") else \
              "r1" if msg.startswith("r1") else \
              "r3_revert" if (msg.startswith("r3") and r.get("decision") == "revert") else \
              "r4" if (msg.startswith("r4") or "attr=4" in msg) else \
              "r5" if msg.startswith("r5") else "r?"
        keys.append(key)
        labels.append(A.get(key, msg[:24]))

    fig, ax = plt.subplots(figsize=(13, 7.5))
    # best-so-far step (kept rounds only)
    best, bestline = None, []
    for o, d in zip(objs, decisions):
        if d == "keep":
            best = o if best is None else max(best, o)
        bestline.append(best if best is not None else (objs[0]))
    ax.step(xs, bestline, where="post", color="#2ca02c", lw=2.5, label="best-so-far loop_objective", zorder=2)
    ax.plot(xs, objs, "o-", color="#1f77b4", lw=1.4, ms=9, label="loop_objective (this round)", zorder=3)
    ax.plot(xs, t1, "^--", color="#ff7f0e", alpha=.75, label="T1 success@1")
    ax.plot(xs, t2, "s--", color="#9467bd", alpha=.75, label="T2 acc@0.5m")
    ax.plot(xs, t3, "d--", color="#d62728", alpha=.75, label="T3 llm_match")

    for i in xs:
        keep = decisions[i] == "keep"
        ax.scatter([i], [objs[i]], s=190, marker=("o" if keep else "X"),
                   color=("#2ca02c" if keep else "#d62728"), zorder=4,
                   edgecolor="black", linewidth=0.6)
        tag = ("KEEP" if keep else "REVERT")
        ax.annotate(f"{tag}\n{labels[i]}",
                    (i, objs[i]), textcoords="offset points",
                    xytext=(0, 22 if i % 2 == 0 else -52), ha="center", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc=("#e8f5e9" if keep else "#ffebee"),
                              ec=("#2ca02c" if keep else "#d62728"), alpha=.9))
        ax.annotate(f"{objs[i]:.3f}", (i, objs[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, color="#1f77b4")

    title = "Agent-Designed 记忆 · 自改进进程 (run2)" if use_cn else \
            "Agent-Designed memory · AutoResearch progress (run2)"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("轮次 round" if use_cn else "round")
    ax.set_ylabel("分数 score" if use_cn else "score")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"r{r.get('round')}\n({d})" for r, d in zip(rows, decisions)])
    ax.grid(alpha=.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.3, 2.1)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  (CJK font: {cn or 'NONE -> English labels'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
