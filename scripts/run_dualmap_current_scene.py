from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from adapters.dualmap import (
    DEFAULT_DUALMAP_ROOT,
    DEFAULT_MAP_DIR,
    DEFAULT_SCENE_ID,
    create_method,
)
from spatial_memory_evaluation import RGBDSequence
from spatial_memory_evaluation.output_paths import (
    run_timestamp,
    timestamped_memory_dir,
    timestamped_result_dir,
)


DEFAULT_KWARGS = Path(
    "configs/dualmap_current_scene_method_kwargs.json"
)
DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DualMap get_object on the current ScanNet++ scene."
    )
    parser.add_argument("--dualmap-root", type=Path, default=DEFAULT_DUALMAP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--method-kwargs", type=Path, default=DEFAULT_KWARGS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory for result outputs. Default: results/dualmap/current-scene-smoke/<timestamp>",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help="directory for generated memory. Default: memories/dualmap/current-scene-smoke/<timestamp>",
    )
    parser.add_argument(
        "--object-queries",
        type=Path,
        default=Path(
            "examples/claws_current_scene_object_queries.json"
        ),
    )
    parser.add_argument(
        "--object-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=None,
        help="Temporary sqlite DB written from DualMap objects for the existing metric.",
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
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--skip-metric", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _resolve_output_paths(args)
    args.object_output.parent.mkdir(parents=True, exist_ok=True)
    args.memory_db.parent.mkdir(parents=True, exist_ok=True)
    args.metric_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.metric_output_md.parent.mkdir(parents=True, exist_ok=True)

    kwargs = _load_kwargs(args.method_kwargs)
    kwargs["dualmap_root"] = str(args.dualmap_root)
    kwargs["map_dir"] = str(args.map_dir)
    kwargs["scene_id"] = args.scene_id

    sequence = RGBDSequence(
        episode_history=f"scannetpp/{args.scene_id}",
        root=args.map_dir,
        frames=[],
        metadata={"source": "prebuilt_dualmap_map"},
    )
    method = create_method(sequence=sequence, **kwargs)

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

    memory_db = method.export_spatial_memory_db(args.memory_db)
    print(f"wrote DualMap metric DB: {memory_db}")

    if not args.skip_metric:
        _run_current_scene_metric(args, memory_db)

    return 0


def _resolve_output_paths(args: argparse.Namespace) -> None:
    timestamp = run_timestamp()
    if args.run_dir is None:
        args.run_dir = timestamped_result_dir(
            "dualmap", "current-scene-smoke", timestamp=timestamp
        )
    if args.memory_dir is None:
        args.memory_dir = timestamped_memory_dir(
            "dualmap", "current-scene-smoke", timestamp=timestamp
        )
    if args.object_output is None:
        args.object_output = args.run_dir / "object-predictions.json"
    if args.memory_db is None:
        args.memory_db = args.memory_dir / "memory.db"
    if args.metric_output_json is None:
        args.metric_output_json = args.run_dir / "object-metrics.json"
    if args.metric_output_md is None:
        args.metric_output_md = args.run_dir / "object-metrics.md"


def _run_current_scene_metric(args: argparse.Namespace, memory_db: Path) -> None:
    evaluator = args.claws_root / "scripts" / "evaluate_scannetpp_spatial_rag.py"
    command = [
        sys.executable,
        str(evaluator),
        "--dataset_root",
        str(args.dataset_root),
        "--scene_id",
        args.scene_id,
        "--memory_db",
        str(memory_db),
        "--class_mapping",
        str(args.claws_root / "configs" / "scannetpp_label_mapping.yaml"),
        "--output_json",
        str(args.metric_output_json),
        "--output_md",
        str(args.metric_output_md),
    ]
    print("running current-scene ScanNet++ metric:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _load_kwargs(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


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
    raise SystemExit(main(parse_args()))
