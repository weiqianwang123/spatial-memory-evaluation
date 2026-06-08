from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from adapters.hovsg import (
    DEFAULT_HOVSG_RESULT_PATH,
    DEFAULT_HOVSG_ROOT,
    DEFAULT_SCENE_ID,
    create_method,
)
from spatial_memory_evaluation import RGBDSequence


DEFAULT_KWARGS = Path(
    "spatial-memory-evaluation/configs/hovsg_current_scene_method_kwargs.json"
)
DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HOV-SG get_object recall on the current ScanNet++ scene."
    )
    parser.add_argument("--hovsg-root", type=Path, default=DEFAULT_HOVSG_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--hovsg-result-path", type=Path, default=DEFAULT_HOVSG_RESULT_PATH)
    parser.add_argument("--method-kwargs", type=Path, default=DEFAULT_KWARGS)
    parser.add_argument(
        "--object-queries",
        type=Path,
        default=Path(
            "spatial-memory-evaluation/examples/claws_current_scene_object_queries.json"
        ),
    )
    parser.add_argument(
        "--object-output",
        type=Path,
        default=Path("spatial-memory-evaluation/results/hovsg-current-scene-objects.json"),
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=Path("spatial-memory-evaluation/results/hovsg-full-recall-memory.db"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("spatial-memory-evaluation/results/hovsg-full-recall.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("spatial-memory-evaluation/results/hovsg-full-recall.md"),
    )
    parser.add_argument("--skip-query-output", action="store_true")
    parser.add_argument("--skip-metric", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    kwargs = _load_kwargs(args.method_kwargs)
    kwargs["hovsg_root"] = str(args.hovsg_root)
    kwargs["result_path"] = str(_resolve_result_path(args.hovsg_result_path, args.scene_id))
    kwargs["scene_id"] = args.scene_id

    sequence = RGBDSequence(
        episode_history=f"scannetpp/{args.scene_id}",
        root=Path(kwargs["result_path"]),
        frames=[],
        metadata={"source": "prebuilt_hovsg_map"},
    )
    method = create_method(sequence=sequence, **kwargs)

    if not args.skip_query_output:
        _write_query_predictions(args, method)

    memory_db = method.export_spatial_memory_db(args.memory_db)
    print(f"exported all HOV-SG objects to metric DB: {memory_db}")

    if not args.skip_metric:
        _run_scannetpp_metric(args, memory_db)
        _print_recall_summary(args.output_json)
    return 0


def _write_query_predictions(args: argparse.Namespace, method: Any) -> None:
    args.object_output.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"wrote HOV-SG get_object predictions: {args.object_output}")


def _run_scannetpp_metric(args: argparse.Namespace, memory_db: Path) -> None:
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
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
        str(args.output_json),
        "--output_md",
        str(args.output_md),
    ]
    print("running full ScanNet++ object recall metric:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _print_recall_summary(path: Path) -> None:
    with path.open("r") as f:
        report = json.load(f)
    coverage = report["coverage"]
    anchor = report["anchor_quality"]
    print("full recall summary:")
    print(
        "  detector-coverable recall: "
        f"{coverage['object_memory_recall_detector_coverable']:.4f} "
        f"({coverage['matched_detector_coverable_gt']}/"
        f"{coverage['total_detector_coverable_gt']})"
    )
    print(
        "  all-valid recall: "
        f"{coverage['object_memory_recall_all_valid']:.4f} "
        f"({coverage['matched_all_valid_gt']}/{coverage['total_all_valid_gt']})"
    )
    print(f"  mean anchor error: {anchor['mean_error_m']:.4f} m")
    print(f"  median anchor error: {anchor['median_error_m']:.4f} m")


def _resolve_result_path(path: Path, scene_id: str) -> Path:
    if path.exists():
        return path

    candidates = [
        path,
        Path(f"/home/robin_wang/HOV-SG/output/sem_seg_new_scannet_mobileclip/{scene_id}/scannet"),
        Path(f"/home/robin_wang/HOV-SG/data/sem_seg/{scene_id}/scannet"),
        Path(f"/data/mondo-training-dataset/semantic_mapping/hovsg/{scene_id}/scannet"),
        Path(f"/data/mondo-training-dataset/semantic_mapping/hovsg/scannetpp_{scene_id}/scannet"),
    ]
    for candidate in candidates:
        if (candidate / "mask_feats.pt").exists() and (candidate / "objects").exists():
            return candidate
    return path


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
