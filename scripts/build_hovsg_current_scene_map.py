from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_HOVSG_ROOT = Path("/home/robin_wang/HOV-SG")
DEFAULT_SCENE_ID = "036bce3393"
DEFAULT_DATASET_PATH = Path(
    "/data/mondo-training-dataset/semantic_mapping/dualmap/"
    "scannetpp_dataset/exported/036bce3393"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/mondo-training-dataset/semantic_mapping/hovsg/scannetpp_036bce3393"
)
DEFAULT_SAM_CHECKPOINT = Path("/home/robin_wang/DualMap/sam_b.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a HOV-SG object feature map for the current ScanNet++ scene."
    )
    parser.add_argument("--hovsg-root", type=Path, default=DEFAULT_HOVSG_ROOT)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--sam-type", default="vit_b")
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--points-per-side", type=int, default=8)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument(
        "--merge-type",
        choices=("hierarchical", "sequential"),
        default="hierarchical",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    _validate_paths(args)

    command = [
        sys.executable,
        "application/semantic_segmentation.py",
        "main.dataset=scannet",
        f"main.scene_id={args.scene_id}",
        f"main.dataset_path={args.dataset_path}",
        f"main.save_path={args.output_root}",
        f"models.clip.type={args.clip_model}",
        f"models.clip.checkpoint={args.clip_pretrained}",
        f"models.sam.type={args.sam_type}",
        f"models.sam.checkpoint={args.sam_checkpoint}",
        f"models.sam.points_per_side={args.points_per_side}",
        f"models.sam.points_per_batch={args.points_per_batch}",
        f"pipeline.skip_frames={args.skip_frames}",
        f"pipeline.merge_type={args.merge_type}",
    ]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = f"{args.hovsg_root}:{env.get('PYTHONPATH', '')}"
    env["HYDRA_FULL_ERROR"] = "1"

    print("running HOV-SG build:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(args.hovsg_root), env=env, check=True)
    print(f"HOV-SG result path: {args.output_root / 'scannet'}")
    return 0


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.hovsg_root.exists():
        raise FileNotFoundError(f"HOV-SG root does not exist: {args.hovsg_root}")
    if not args.dataset_path.exists():
        raise FileNotFoundError(f"RGB-D dataset path does not exist: {args.dataset_path}")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {args.sam_checkpoint}")


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
