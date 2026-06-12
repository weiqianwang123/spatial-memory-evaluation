from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from adapters.claws_spatial_rag import (
    DEFAULT_CLAWS_ROOT,
    DEFAULT_MEMORY_DB,
    DEFAULT_SCENE_ID,
    create_method,
)
from spatial_memory_evaluation import RGBDSequence
from spatial_memory_evaluation.output_paths import timestamped_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-run ClawS SpatialRAG on the current ScanNet++ scene."
    )
    parser.add_argument("--spatial-rag-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--memory-db", type=Path, default=DEFAULT_MEMORY_DB)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory for result outputs. Default: results/claws/current-scene-smoke/<timestamp>",
    )
    parser.add_argument(
        "--object-queries",
        type=Path,
        default=Path(
            "examples/claws_current_scene_object_queries.json"
        ),
    )
    parser.add_argument(
        "--memory-questions",
        type=Path,
        default=Path(
            "examples/claws_current_scene_memory_questions.json"
        ),
    )
    parser.add_argument(
        "--object-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metric-output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--metric-output-md",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/mondo-training-dataset/semantic_mapping/scannetpp"),
    )
    parser.add_argument("--embedding-provider", default="mock", choices=("mock", "ollama", "vllm"))
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--embedding-endpoint", default="http://localhost:11434")
    parser.add_argument("--skip-metric", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    _resolve_output_paths(args)
    args.object_output.parent.mkdir(parents=True, exist_ok=True)
    args.memory_output.parent.mkdir(parents=True, exist_ok=True)
    args.metric_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.metric_output_md.parent.mkdir(parents=True, exist_ok=True)

    sequence = RGBDSequence(
        episode_history=f"scannetpp/{args.scene_id}",
        root=args.memory_db.parent,
        frames=[],
        metadata={"source": "prebuilt_claws_memory_db"},
    )
    method = create_method(
        sequence=sequence,
        spatial_rag_root=str(args.spatial_rag_root),
        memory_db=str(args.memory_db),
        scene_id=args.scene_id,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_endpoint=args.embedding_endpoint,
    )

    object_results = []
    for item in _load_list(args.object_queries):
        objects = [obj.to_json() for obj in method.get_object(str(item["query"]))]
        object_results.append(
            {
                "query_id": item["query_id"],
                "episode_history": item["episode_history"],
                "query": item["query"],
                "objects": objects,
            }
        )
    _write_json(args.object_output, object_results)
    print(f"wrote object predictions: {args.object_output}")

    memory_results = []
    for item in _load_list(args.memory_questions):
        memory_results.append(
            {
                "question_id": item["question_id"],
                "episode_history": item["episode_history"],
                "question": item["question"],
                "answer": method.get_memory_text(str(item["question"])),
            }
        )
    _write_json(args.memory_output, memory_results)
    print(f"wrote memory contexts: {args.memory_output}")

    method.close()

    if not args.skip_metric:
        _run_current_scene_metric(args)


def _resolve_output_paths(args: argparse.Namespace) -> None:
    if args.run_dir is None:
        args.run_dir = timestamped_result_dir("claws", "current-scene-smoke")
    if args.object_output is None:
        args.object_output = args.run_dir / "object-predictions.json"
    if args.memory_output is None:
        args.memory_output = args.run_dir / "memory-contexts.json"
    if args.metric_output_json is None:
        args.metric_output_json = args.run_dir / "object-metrics.json"
    if args.metric_output_md is None:
        args.metric_output_md = args.run_dir / "object-metrics.md"


def _run_current_scene_metric(args: argparse.Namespace) -> None:
    evaluator = args.spatial_rag_root / "scripts" / "evaluate_scannetpp_spatial_rag.py"
    command = [
        sys.executable,
        str(evaluator),
        "--dataset_root",
        str(args.dataset_root),
        "--scene_id",
        args.scene_id,
        "--memory_db",
        str(args.memory_db),
        "--class_mapping",
        str(args.spatial_rag_root / "configs" / "scannetpp_label_mapping.yaml"),
        "--output_json",
        str(args.metric_output_json),
        "--output_md",
        str(args.metric_output_md),
    ]
    print("running current-scene ScanNet++ metric:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _load_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
