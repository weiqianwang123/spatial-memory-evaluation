#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCANNETPP_ROOT = Path("/data/mondo-training-dataset/semantic_mapping/scannetpp")
DEFAULT_LABEL_COLORS = REPO_ROOT / "spatial_memory_evaluation" / "assets" / "class_lists" / "detector_coverable_colors.json"
POINT_CLOUD_SUFFIXES = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a memory package object map over a 3D scene point cloud.",
    )
    parser.add_argument("package_dir", type=Path, help="memory package directory")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=None,
        help="explicit point cloud or mesh path; otherwise auto-discovered from package/ScanNet++ scene",
    )
    parser.add_argument("--scannetpp-root", type=Path, default=DEFAULT_SCANNETPP_ROOT)
    parser.add_argument("--scene-id", default=None, help="override scene id used for ScanNet++ point cloud discovery")
    parser.add_argument("--object-table", type=Path, default=None, help="default: <package>/memory/object_table.jsonl")
    parser.add_argument("--label-colors", type=Path, default=DEFAULT_LABEL_COLORS)
    parser.add_argument("--label", action="append", default=None, help="filter to one or more labels; repeatable")
    parser.add_argument("--max-objects", type=int, default=0, help="0 means no object limit after filtering")
    parser.add_argument("--max-points", type=int, default=350000, help="randomly downsample scene points; 0 disables")
    parser.add_argument("--voxel-size", type=float, default=0.03, help="voxel downsample size for scene point cloud; 0 disables")
    parser.add_argument(
        "--ply-loader",
        choices=("auto", "fast", "open3d"),
        default="auto",
        help="fast reads binary PLY vertices only and skips mesh faces",
    )
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--center-radius", type=float, default=0.06)
    parser.add_argument("--hide-bboxes", action="store_true")
    parser.add_argument("--hide-centers", action="store_true")
    parser.add_argument("--hide-point-cloud", action="store_true")
    parser.add_argument("--show-labels", action="store_true", help="use Open3D GUI labels instead of legacy viewer")
    parser.add_argument("--print-objects", action="store_true", help="print object table rows selected for visualization")
    parser.add_argument("--dry-run", action="store_true", help="load inputs and print summary without opening a viewer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    manifest = _read_json(package_dir / "manifest.json")
    object_table = args.object_table or (package_dir / "memory" / "object_table.jsonl")
    objects = _load_objects(object_table)
    objects = _filter_objects(objects, labels=args.label, max_objects=args.max_objects)
    label_colors = _load_label_colors(args.label_colors)
    point_cloud_path = _resolve_point_cloud_path(args, package_dir, manifest)

    summary = {
        "package_dir": str(package_dir),
        "method": _get_nested(manifest, ("method", "name")),
        "scene_id": args.scene_id or _get_nested(manifest, ("dataset", "scene_id")),
        "object_table": str(object_table),
        "object_count": len(objects),
        "labels": dict(sorted(Counter(_label_of(obj) for obj in objects).items())),
        "point_cloud": str(point_cloud_path) if point_cloud_path is not None else None,
        "show_point_cloud": not args.hide_point_cloud,
        "show_bboxes": not args.hide_bboxes,
        "show_centers": not args.hide_centers,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_objects:
        for obj in objects:
            print(
                json.dumps(
                    {
                        "object_id": obj.get("object_id"),
                        "label": obj.get("label"),
                        "position_3d": obj.get("position_3d"),
                        "bbox_3d": obj.get("bbox_3d"),
                        "num_points": obj.get("num_points"),
                    },
                    sort_keys=True,
                )
            )
    if args.dry_run:
        return 0

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for visualization") from exc

    geometries: list[Any] = []
    labels: list[tuple[list[float], str]] = []
    if not args.hide_point_cloud:
        if point_cloud_path is None:
            print("warning: no point cloud found; visualizing objects only", file=sys.stderr)
        else:
            scene_geometry = _load_point_cloud(
                o3d=o3d,
                path=point_cloud_path,
                voxel_size=args.voxel_size,
                max_points=args.max_points,
                ply_loader=args.ply_loader,
            )
            geometries.append(scene_geometry)

    if not args.hide_bboxes:
        geometries.extend(_make_bboxes(o3d, objects, label_colors))
    if not args.hide_centers:
        centers, center_labels = _make_centers(o3d, objects, label_colors, radius=args.center_radius)
        geometries.extend(centers)
        labels.extend(center_labels)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6))

    if not geometries:
        raise RuntimeError("nothing to visualize")
    if args.show_labels:
        _draw_with_labels(o3d, geometries, labels, point_size=args.point_size)
    else:
        _draw_legacy(o3d, geometries, point_size=args.point_size)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _load_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"object table not found: {path}")
    objects = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    objects.append(value)
    return objects


def _filter_objects(
    objects: list[dict[str, Any]],
    *,
    labels: list[str] | None,
    max_objects: int,
) -> list[dict[str, Any]]:
    if labels:
        wanted = {_normalize_label(label) for label in labels}
        objects = [obj for obj in objects if _normalize_label(_label_of(obj)) in wanted]
    if max_objects > 0:
        objects = objects[:max_objects]
    return objects


def _load_label_colors(path: Path) -> dict[str, tuple[float, float, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    colors = {}
    if isinstance(value, dict):
        for label, color in value.items():
            if isinstance(color, list) and len(color) >= 3:
                colors[_normalize_label(label)] = tuple(float(channel) for channel in color[:3])
    return colors


def _resolve_point_cloud_path(args: argparse.Namespace, package_dir: Path, manifest: dict[str, Any]) -> Path | None:
    if args.point_cloud is not None:
        return args.point_cloud.resolve()

    candidates = list(_manifest_point_cloud_candidates(package_dir, manifest))
    scene_id = args.scene_id or _get_nested(manifest, ("dataset", "scene_id"))
    if scene_id:
        scene_dir = args.scannetpp_root / "data" / str(scene_id) / "scans"
        candidates.extend(
            [
                scene_dir / "mesh_aligned_0.05.ply",
                scene_dir / "mesh_aligned_0.05_semantic.ply",
                scene_dir / "mesh_aligned.ply",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in POINT_CLOUD_SUFFIXES:
            return candidate.resolve()
    return None


def _manifest_point_cloud_candidates(package_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for section in ("memory_artifacts", "evidence_artifacts", "raw_links"):
        values = manifest.get(section)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if not raw_path:
                continue
            path = Path(str(raw_path))
            paths = [path] if path.is_absolute() else [package_dir / path, REPO_ROOT / path]
            for candidate in paths:
                if candidate.suffix.lower() in POINT_CLOUD_SUFFIXES:
                    candidates.append(candidate)
                elif candidate.exists() and candidate.is_dir():
                    candidates.extend(_find_point_clouds(candidate))
    return candidates


def _find_point_clouds(root: Path) -> list[Path]:
    preferred_names = ("mesh_aligned_0.05.ply", "mesh_aligned.ply", "scene.ply", "pointcloud.ply")
    found = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in POINT_CLOUD_SUFFIXES]
    found.sort(key=lambda path: (preferred_names.index(path.name) if path.name in preferred_names else 99, len(path.parts), str(path)))
    return found


def _load_point_cloud(
    *,
    o3d: Any,
    path: Path,
    voxel_size: float,
    max_points: int,
    ply_loader: str,
) -> Any:
    pcd = None
    if path.suffix.lower() == ".ply" and ply_loader in ("auto", "fast"):
        pcd = _try_load_binary_ply_vertices(o3d=o3d, path=path, max_points=max_points)
        if pcd is None and ply_loader == "fast":
            raise RuntimeError(f"fast PLY loader does not support this file: {path}")
    if pcd is None:
        pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_vertices():
            raise RuntimeError(f"failed to read point cloud or mesh: {path}")
        sample_count = max_points if max_points > 0 else min(len(mesh.vertices), 500000)
        pcd = mesh.sample_points_uniformly(number_of_points=max(1, sample_count))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if max_points > 0 and len(pcd.points) > max_points:
        pcd = pcd.random_down_sample(max_points / len(pcd.points))
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.62, 0.62, 0.62])
    return pcd


def _try_load_binary_ply_vertices(o3d: Any, path: Path, max_points: int) -> Any | None:
    import numpy as np

    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                return None
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        header_size = f.tell()

    if not any(line == "format binary_little_endian 1.0" for line in header_lines):
        return None

    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            vertex_properties.append((parts[2], parts[1]))
    if vertex_count <= 0:
        return None
    property_dtypes = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    try:
        dtype = np.dtype([(name, property_dtypes[property_type]) for name, property_type in vertex_properties])
    except KeyError:
        return None
    required = {"x", "y", "z"}
    if not required.issubset(dtype.names or ()):
        return None

    with path.open("rb") as f:
        f.seek(header_size)
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
    if len(vertices) == 0:
        return None

    if max_points > 0 and len(vertices) > max_points:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(len(vertices), size=max_points, replace=False))
        vertices = vertices[indices]

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64, copy=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if {"red", "green", "blue"}.issubset(dtype.names or ()):
        colors = np.column_stack([vertices["red"], vertices["green"], vertices["blue"]]).astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _make_bboxes(o3d: Any, objects: list[dict[str, Any]], colors: dict[str, tuple[float, float, float]]) -> list[Any]:
    geometries = []
    for obj in objects:
        bbox = _bbox_of(obj)
        if bbox is None:
            continue
        box = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:3], max_bound=bbox[3:])
        box.color = _color_for(_label_of(obj), colors)
        geometries.append(box)
    return geometries


def _make_centers(
    o3d: Any,
    objects: list[dict[str, Any]],
    colors: dict[str, tuple[float, float, float]],
    radius: float,
) -> tuple[list[Any], list[tuple[list[float], str]]]:
    geometries = []
    labels = []
    for obj in objects:
        center = _center_of(obj)
        if center is None:
            continue
        label = _label_of(obj)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh.translate(center)
        mesh.paint_uniform_color(_color_for(label, colors))
        geometries.append(mesh)
        labels.append((center, f"{label}:{obj.get('object_id')}"))
    return geometries, labels


def _draw_legacy(o3d: Any, geometries: list[Any], point_size: float) -> None:
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="memory package viewer", width=1400, height=900)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    render_options = visualizer.get_render_option()
    render_options.point_size = float(point_size)
    render_options.background_color = [0.04, 0.04, 0.04]
    visualizer.run()
    visualizer.destroy_window()


def _draw_with_labels(o3d: Any, geometries: list[Any], labels: list[tuple[list[float], str]], point_size: float) -> None:
    gui = o3d.visualization.gui
    app = gui.Application.instance
    app.initialize()
    visualizer = o3d.visualization.O3DVisualizer("memory package viewer", 1400, 900)
    visualizer.show_settings = True
    for index, geometry in enumerate(geometries):
        visualizer.add_geometry(f"geometry_{index:04d}", geometry)
    for position, text in labels:
        visualizer.add_3d_label(position, text)
    try:
        material = visualizer.get_geometry_material("geometry_0000")
        if material is not None:
            material.point_size = float(point_size)
    except Exception:
        pass
    app.add_window(visualizer)
    app.run()


def _bbox_of(obj: dict[str, Any]) -> list[float] | None:
    bbox = obj.get("bbox_3d")
    if isinstance(bbox, list) and len(bbox) >= 6:
        values = [float(value) for value in bbox[:6]]
        mins = [min(values[i], values[i + 3]) for i in range(3)]
        maxs = [max(values[i], values[i + 3]) for i in range(3)]
        return mins + maxs
    return None


def _center_of(obj: dict[str, Any]) -> list[float] | None:
    position = obj.get("position_3d")
    if isinstance(position, list) and len(position) >= 3:
        return [float(value) for value in position[:3]]
    bbox = _bbox_of(obj)
    if bbox is None:
        return None
    return [(bbox[i] + bbox[i + 3]) / 2.0 for i in range(3)]


def _label_of(obj: dict[str, Any]) -> str:
    return str(obj.get("label") or "object")


def _color_for(label: str, colors: dict[str, tuple[float, float, float]]) -> tuple[float, float, float]:
    normalized = _normalize_label(label)
    if normalized in colors:
        return colors[normalized]
    rng = random.Random(normalized)
    return (0.25 + 0.7 * rng.random(), 0.25 + 0.7 * rng.random(), 0.25 + 0.7 * rng.random())


def _normalize_label(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _get_nested(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    raise SystemExit(main())
