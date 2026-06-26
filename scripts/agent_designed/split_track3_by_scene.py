#!/usr/bin/env python
"""Split the flat OpenEQA-ScanNet Track 3 build into per-scene benchmark dirs.

``scripts/build_track3_data.py --dataset scannet`` writes one flat directory
(``benchmarks/track3/openeqa/scannet/{questions,answers}.jsonl``). The Track 3
evaluator + ``eval_all_scannet.sh`` consume PER-SCENE dirs
(``benchmarks/track3/openeqa/<scene>/``). This regroups the flat build by the
scene id parsed from each question's ``episode_id`` and writes the same per-scene
layout the held-out 10 scenes already use (questions.jsonl + answers.jsonl +
metadata.json).

Usage:
    python scripts/agent_designed/split_track3_by_scene.py scene0527_00 [scene...]
    python scripts/agent_designed/split_track3_by_scene.py --all
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.jsonl import read_jsonl, write_json, write_jsonl

SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
FLAT_DIR = REPO_ROOT / "benchmarks" / "track3" / "openeqa" / "scannet"
OUT_ROOT = REPO_ROOT / "benchmarks" / "track3" / "openeqa"


def _scene_of(row: dict) -> str | None:
    m = SCENE_RE.search(str(row.get("episode_id") or ""))
    return m.group(0) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Split flat OpenEQA-ScanNet T3 build into per-scene dirs.")
    ap.add_argument("scenes", nargs="*", help="scene ids to extract (e.g. scene0527_00)")
    ap.add_argument("--all", action="store_true", help="extract every scene present in the flat build")
    args = ap.parse_args()

    questions = read_jsonl(FLAT_DIR / "questions.jsonl")
    answers = {r["question_id"]: r for r in read_jsonl(FLAT_DIR / "answers.jsonl")}

    by_scene: dict[str, list[dict]] = {}
    for q in questions:
        scene = _scene_of(q)
        if scene:
            by_scene.setdefault(scene, []).append(q)

    targets = sorted(by_scene) if args.all else args.scenes
    if not targets:
        print("no scenes given; use --all or pass scene ids", file=sys.stderr)
        return 2

    for scene in targets:
        qs = by_scene.get(scene, [])
        if not qs:
            print(f"[skip] {scene}: 0 questions in flat build")
            continue
        out = OUT_ROOT / scene
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "questions.jsonl", qs)
        write_jsonl(out / "answers.jsonl", [answers[q["question_id"]] for q in qs if q["question_id"] in answers])
        write_json(out / "metadata.json", {"dataset": "scannet", "scene_id": scene, "question_count": len(qs)})
        print(f"[ok] {scene}: {len(qs)} questions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
