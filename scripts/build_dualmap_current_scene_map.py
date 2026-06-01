from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUALMAP_ROOT = Path("/home/robin_wang/DualMap")
DEFAULT_CLAWS_ROOT = Path("/home/robin_wang/ClawS-SpatialRAG")
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_DUALMAP_DATASET_ROOT = Path(
    "/data/mondo-training-dataset/semantic_mapping/dualmap/scannetpp_dataset"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/mondo-training-dataset/semantic_mapping/dualmap/scannetpp_036bce3393"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the current ScanNet++ scene and build a DualMap map."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--dualmap-root", type=Path, default=DEFAULT_DUALMAP_ROOT)
    parser.add_argument("--claws-root", type=Path, default=DEFAULT_CLAWS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--dualmap-dataset-root", type=Path, default=DEFAULT_DUALMAP_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--export-stride", type=int, default=1)
    parser.add_argument("--run-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--yolo-model", default="yolov8s-world.pt")
    parser.add_argument("--sam-model", default="sam_b.pt")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("extra_overrides", nargs="*")
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.dualmap_root),
            str(args.claws_root),
            str(args.repo_root / "spatial-memory-evaluation"),
            env.get("PYTHONPATH", ""),
        ]
    )

    if not args.skip_export:
        export_script = args.repo_root / "spatial-memory-evaluation/scripts/export_scannetpp_for_dualmap.py"
        export_cmd = [
            sys.executable,
            str(export_script),
            "--claws-root",
            str(args.claws_root),
            "--dataset-root",
            str(args.dataset_root),
            "--scene-id",
            args.scene_id,
            "--output-root",
            str(args.dualmap_dataset_root),
            "--max-frames",
            str(args.max_frames),
            "--stride",
            str(args.export_stride),
            "--overwrite",
        ]
        _run(export_cmd, cwd=args.repo_root, env=env)

    map_dir = args.output_dir / "map"
    detection_dir = args.output_dir / "detections"
    run_cmd = [
        sys.executable,
        "-m",
        "applications.runner_dataset",
        "dataset_name=scannet",
        f"scene_id={args.scene_id}",
        f"dataset_path={args.dualmap_dataset_root}",
        f"dataset_conf_path={args.dualmap_root / 'config/data_config/dataset/scannet.yaml'}",
        f"dataset_gt_path={args.dataset_root}",
        f"output_path={args.output_dir}",
        f"map_save_path={map_dir}",
        f"detection_path={detection_dir}",
        "use_rerun=false",
        "visualize_detection=false",
        "use_parallel=false",
        "use_fastsam=false",
        "skip_refinement=true",
        "save_layout=false",
        f"yolo.model_path={args.yolo_model}",
        f"sam.model_path={args.sam_model}",
        "clip.model_name=ViT-B-32",
        "clip.pretrained=laion2b_s34b_b79k",
        "clip.clip_length=512",
        f"device={args.device}",
        f"stride={args.run_stride}",
        *args.extra_overrides,
    ]
    _run(run_cmd, cwd=args.dualmap_root, env=env)
    return 0


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
