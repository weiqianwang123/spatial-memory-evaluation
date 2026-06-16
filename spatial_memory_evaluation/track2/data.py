from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_json, read_jsonl, write_json, write_jsonl
from spatial_memory_evaluation.common.labels import write_default_alias_file


def build_track2_queries(
    *,
    track1_benchmark_dir: Path,
    output_dir: Path,
    top_k: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split in ("all_annotated", "detector_coverable"):
        gt_objects = read_jsonl(track1_benchmark_dir / f"{split}.jsonl")
        queries = _queries_for_split(split, gt_objects, top_k=top_k)
        write_jsonl(output_dir / f"queries_{split}.jsonl", queries)
        summaries[split] = len(queries)
    write_default_alias_file(output_dir / "label_aliases.json")
    track1_metadata_path = track1_benchmark_dir / "metadata.json"
    track1_metadata = read_json(track1_metadata_path) if track1_metadata_path.exists() else {}
    summary = {
        "output_dir": str(output_dir),
        "track1_benchmark_dir": str(track1_benchmark_dir),
        "scene_id": track1_metadata.get("scene_id", "unknown") if isinstance(track1_metadata, dict) else "unknown",
        "query_counts": summaries,
        "top_k": top_k,
    }
    write_json(output_dir / "metadata.json", summary)
    return summary


def _queries_for_split(split: str, gt_objects: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in gt_objects:
        grouped[str(obj["canonical_label"])].append(obj)
    scene_id = str(gt_objects[0]["scene_id"]) if gt_objects else "unknown"
    queries = []
    for index, label in enumerate(sorted(grouped)):
        targets = grouped[label]
        queries.append(
            {
                "query_id": f"{scene_id}_{split}_{index:04d}_{label.replace(' ', '_')}",
                "scene_id": scene_id,
                "split": split,
                "canonical_label": label,
                "query": f"where is the {label}?",
                "top_k": top_k,
                "target_gt_ids": [str(obj["gt_id"]) for obj in targets],
            }
        )
    return queries
