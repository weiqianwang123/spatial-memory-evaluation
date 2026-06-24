#!/usr/bin/env python
"""Aggregate the 10-scene ScanNet eval results into per-method/track means.

Reads results/<method>/track<N>-<mode>/scannet-<scene>/eval_summary.json and
prints a compact per-track table (mean over scenes) plus a JSON blob suitable
for pasting into RESULTS.md. Scenes with status != ok are reported as skipped.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METHODS = ["daaam", "claws", "remembr", "remembr_captions", "multiframe_vlm"]
SCENES = [
    "scene0015_00", "scene0050_00", "scene0077_00", "scene0084_00", "scene0131_00",
    "scene0193_00", "scene0207_00", "scene0222_00", "scene0256_00", "scene0314_00",
]

# (track, mode) -> metric keys to average
TRACK_METRICS = {
    ("1", "tool_llm"): ["success@5", "success@1", "mean_first_hit_distance_m", "mrr", "mean_query_latency_ms"],
    ("1", "fixed_api"): ["success@5", "success@1", "mean_first_hit_distance_m", "mrr", "mean_query_latency_ms"],
    ("2", "tool_llm"): ["acc@0.25m", "acc@0.5m", "mean_center_distance_m", "mean_query_latency_ms"],
    ("2", "fixed_api"): ["acc@0.25m", "acc@0.5m", "mean_center_distance_m", "mean_query_latency_ms"],
    ("3", "tool_llm"): ["llm_match", "answered_rate", "mean_query_latency_ms"],
}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> int:
    out = {}
    for (track, mode), keys in TRACK_METRICS.items():
        for method in METHODS:
            per_scene = {}
            agg = defaultdict(list)
            qtotal = 0
            for scene in SCENES:
                f = REPO / "results" / method / f"track{track}-{mode}" / f"scannet-{scene}" / "eval_summary.json"
                if not f.exists():
                    continue
                try:
                    d = json.loads(f.read_text())
                except Exception:
                    continue
                status = d.get("status")
                metrics = d.get("metrics") if isinstance(d.get("metrics"), dict) else {}
                if status != "ok" or not metrics:
                    per_scene[scene] = {"status": status}
                    continue
                qtotal += int(metrics.get("query_count") or metrics.get("question_count") or 0)
                row = {}
                for k in keys:
                    v = metrics.get(k)
                    row[k] = v
                    if isinstance(v, (int, float)):
                        agg[k].append(v)
                per_scene[scene] = row
            if not per_scene:
                continue
            means = {k: _mean(agg[k]) for k in keys}
            scenes_ok = sum(1 for v in per_scene.values() if "status" not in v)
            out[f"track{track}-{mode}:{method}"] = {
                "scenes_ok": scenes_ok,
                "scenes_total": len(per_scene),
                "query_total": qtotal,
                "means": means,
            }

    # human-readable
    by_track = defaultdict(list)
    for key, v in out.items():
        tk, method = key.split(":")
        by_track[tk].append((method, v))
    for tk in sorted(by_track):
        print(f"\n===== {tk} =====")
        for method, v in by_track[tk]:
            m = v["means"]
            ms = "  ".join(f"{k}={m[k]}" for k in m)
            print(f"  {method:18s} [{v['scenes_ok']}/{v['scenes_total']} scenes, {v['query_total']} q]  {ms}")

    print("\n----- JSON -----")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
