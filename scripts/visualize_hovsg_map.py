from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d

from adapters.hovsg import DEFAULT_HOVSG_RESULT_PATH, create_method
from spatial_memory_evaluation import RGBDSequence


PALETTE = [
    [1.0, 0.15, 0.10],
    [0.10, 0.55, 1.0],
    [0.20, 0.85, 0.30],
    [1.0, 0.75, 0.10],
    [0.85, 0.25, 1.0],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved HOV-SG object map.")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_HOVSG_RESULT_PATH)
    parser.add_argument(
        "--mode",
        choices=("full", "masked", "objects", "query"),
        default="masked",
        help="full=raw RGB point cloud, masked=all HOV-SG masks, objects=individual object PLYs, query=highlight get_object results",
    )
    parser.add_argument("--query", default=None, help="text query used in query mode")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-objects", type=int, default=200)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--show-full-context", action="store_true")
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    result_path = args.result_path.expanduser()
    _validate_result_path(result_path)

    if args.mode == "full":
        geometries = [_load_point_cloud(result_path / "full_pcd.ply", args.voxel_size)]
    elif args.mode == "masked":
        geometries = [_load_point_cloud(result_path / "masked_pcd.ply", args.voxel_size)]
    elif args.mode == "objects":
        geometries = _load_colored_objects(result_path, args.max_objects, args.voxel_size)
    else:
        if not args.query:
            raise ValueError("--query is required when --mode=query")
        geometries = _load_query_geometries(args, result_path)

    print(f"opening Open3D viewer with {len(geometries)} geometries")
    print("controls: mouse rotate/pan/zoom, q or Esc to close")
    _draw(geometries, point_size=args.point_size)
    return 0


def _load_query_geometries(args: argparse.Namespace, result_path: Path) -> list[o3d.geometry.Geometry]:
    sequence = RGBDSequence(
        episode_history="scannetpp/036bce3393",
        root=result_path,
        frames=[],
        metadata={"source": "prebuilt_hovsg_map"},
    )
    method = create_method(
        sequence=sequence,
        result_path=result_path,
        top_k=args.top_k,
        clip_model_name="ViT-B-32",
        clip_pretrained="laion2b_s34b_b79k",
    )
    predictions = list(method.get_object(args.query))
    if not predictions:
        print(f"no HOV-SG objects matched query: {args.query}")
        return []

    print(f"query: {args.query}")
    geometries = []
    if args.show_full_context:
        context = _load_point_cloud(result_path / "full_pcd.ply", max(args.voxel_size, 0.03))
        context.paint_uniform_color([0.55, 0.55, 0.55])
        geometries.append(context)

    objects_dir = result_path / "objects"
    for rank, prediction in enumerate(predictions):
        color = PALETTE[rank % len(PALETTE)]
        object_id = str(prediction.object_id)
        object_path = objects_dir / f"{object_id}.ply"
        if not object_path.exists():
            print(f"missing object PLY for {object_id}: {object_path}")
            continue

        pcd = _load_point_cloud(object_path, args.voxel_size)
        pcd.paint_uniform_color(color)
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = color
        geometries.extend([pcd, bbox])
        print(
            f"{rank + 1}. {prediction.label} "
            f"score={prediction.score:.4f} id={object_id} path={object_path}"
        )
    return geometries


def _load_colored_objects(
    result_path: Path, max_objects: int, voxel_size: float
) -> list[o3d.geometry.Geometry]:
    object_paths = sorted((result_path / "objects").glob("*.ply"), key=_natural_key)
    if max_objects > 0:
        object_paths = object_paths[:max_objects]
    geometries = []
    for idx, path in enumerate(object_paths):
        pcd = _load_point_cloud(path, voxel_size)
        color = _index_color(idx)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)
    print(f"loaded {len(geometries)} object point clouds")
    return geometries


def _load_point_cloud(path: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    if not path.exists():
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def _draw(geometries: Iterable[o3d.geometry.Geometry], point_size: float) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="HOV-SG Map", width=1600, height=1000)
    render = vis.get_render_option()
    render.point_size = point_size
    render.background_color = np.asarray([0.02, 0.02, 0.025])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()


def _validate_result_path(path: Path) -> None:
    required = [path / "full_pcd.ply", path / "masked_pcd.ply", path / "mask_feats.pt", path / "objects"]
    missing = [item for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(
            "HOV-SG result path is incomplete:\n" + "\n".join(str(item) for item in missing)
        )


def _index_color(index: int) -> list[float]:
    rng = np.random.default_rng(index)
    return rng.uniform(0.15, 1.0, size=3).tolist()


def _natural_key(path: Path) -> list[int | str]:
    parts = []
    number = ""
    text = ""
    for char in path.name:
        if char.isdigit():
            if text:
                parts.append(text)
                text = ""
            number += char
        else:
            if number:
                parts.append(int(number))
                number = ""
            text += char
    if number:
        parts.append(int(number))
    if text:
        parts.append(text)
    return parts


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
