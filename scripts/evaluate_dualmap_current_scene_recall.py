from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

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
        description="Evaluate full scene-level DualMap object recall on ScanNet++."
    )
    parser.add_argument("--dualmap-root", type=Path, default=DEFAULT_DUALMAP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--method-kwargs", type=Path, default=DEFAULT_KWARGS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="directory for result outputs. Default: results/dualmap/object-recall/<timestamp>",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help="directory for generated memory. Default: memories/dualmap/object-recall/<timestamp>",
    )
    parser.add_argument(
        "--memory-db",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _resolve_output_paths(args)
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

    memory_db = method.export_spatial_memory_db(args.memory_db)
    print(f"exported all DualMap objects to metric DB: {memory_db}")

    _run_scannetpp_metric(args, memory_db)
    _print_recall_summary(args.output_json)
    return 0


def _resolve_output_paths(args: argparse.Namespace) -> None:
    timestamp = run_timestamp()
    if args.run_dir is None:
        args.run_dir = timestamped_result_dir("dualmap", "object-recall", timestamp=timestamp)
    if args.memory_dir is None:
        args.memory_dir = timestamped_memory_dir("dualmap", "object-recall", timestamp=timestamp)
    if args.memory_db is None:
        args.memory_db = args.memory_dir / "memory.db"
    if args.output_json is None:
        args.output_json = args.run_dir / "metrics.json"
    if args.output_md is None:
        args.output_md = args.run_dir / "metrics.md"


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


def _load_kwargs(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
