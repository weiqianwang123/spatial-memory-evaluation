"""Resolve ScanNet instance ids -> axis-aligned 3D bounding boxes.

ScanEnts3D / ScanRefer reference objects by ScanNet *instance id* (the
``objectId`` in ``<scene>.aggregation.json``) but carry no 3D geometry in their
JSON. The GT bbox is derivable from the ScanNet scan: aggregation maps
``objectId -> segments``, ``<scene>_vh_clean_2.0.010000.segs.json`` maps each
mesh vertex to a segment, and ``<scene>_vh_clean_2.ply`` holds the vertices. An
instance's bbox is the axis-aligned min/max of its vertices -- the same
construction ScanRefer uses in ``data/scannet/load_scannet_data.export``.

Two conventions matter:

* **ID indexing.** ScanEnts3D ``object_id`` equals the aggregation
  ``segGroups[].objectId`` directly (verified: scene0207_00 objectId 12 ==
  "window"). ScanRefer's loader internally adds 1 to make ids 1-indexed, then
  subtracts it back when writing the box -- so the externally visible id is the
  raw ``objectId``. We key by the raw ``objectId`` and never shift it.
* **Coordinate frame.** ScanNet ships an ``axisAlignment`` matrix in
  ``<scene>.txt`` that rotates the mesh into a gravity-aligned frame. The poses
  our memory builders consume (``.sens`` -> ``SensorData.export_poses``) live in
  the *original, unaligned* mesh frame, so GT boxes are emitted unaligned by
  default to share that frame with method predictions. Pass
  ``apply_axis_align=True`` for the gravity-aligned box (the ScanRefer training
  convention).

Output bbox is corner-form ``[xmin, ymin, zmin, xmax, ymax, zmax]`` to match the
Track 2 evaluator (``_bbox_center``; the evaluator scores distance to this
center, not IoU). The PLY reader is dependency-free (``struct`` only) so the
Track 2 builder runs in any env.
"""

from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_SCANNET_SCANS_ROOT = Path(
    "/data/mondo-training-dataset/semantic_mapping/scannet/scans"
)

# PLY scalar type -> (struct format char, byte size).
_PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


class ScanNetSceneNotFound(FileNotFoundError):
    """Raised when a scene's ScanNet annotation files are not present."""


def scene_paths(scene_id: str, scans_root: Path | str | None = None) -> dict[str, Path]:
    """Return the four ScanNet files for ``scene_id`` (existence not checked)."""

    root = Path(scans_root) if scans_root is not None else DEFAULT_SCANNET_SCANS_ROOT
    scene_dir = root / scene_id
    return {
        "mesh": scene_dir / f"{scene_id}_vh_clean_2.ply",
        "aggregation": scene_dir / f"{scene_id}.aggregation.json",
        "segs": scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "meta": scene_dir / f"{scene_id}.txt",
    }


def scene_available(scene_id: str, scans_root: Path | str | None = None) -> bool:
    """True if the mesh + aggregation + segs files exist for ``scene_id``."""

    paths = scene_paths(scene_id, scans_root)
    return all(paths[k].exists() for k in ("mesh", "aggregation", "segs"))


def _read_ply_vertices_xyz(mesh_file: Path) -> list[tuple[float, float, float]]:
    """Read (x, y, z) for every vertex of a binary/ascii ScanNet PLY.

    Parses the header generically (any property order/types) and extracts only
    the x/y/z columns of the ``vertex`` element. Avoids a numpy/plyfile
    dependency so the Track 2 builder runs in the base env.
    """

    with open(mesh_file, "rb") as handle:
        # --- header ---
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError(f"not a PLY file: {mesh_file}")
        fmt = None
        elements: list[dict[str, Any]] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"unexpected EOF in PLY header: {mesh_file}")
            tokens = line.split()
            if not tokens:
                continue
            keyword = tokens[0]
            if keyword == b"format":
                fmt = tokens[1].decode()
            elif keyword == b"element":
                elements.append(
                    {"name": tokens[1].decode(), "count": int(tokens[2]), "props": []}
                )
            elif keyword == b"property":
                if not elements:
                    raise ValueError(f"property before element in {mesh_file}")
                if tokens[1] == b"list":
                    elements[-1]["props"].append(
                        {
                            "name": tokens[4].decode(),
                            "is_list": True,
                            "count_type": tokens[2].decode(),
                            "item_type": tokens[3].decode(),
                        }
                    )
                else:
                    elements[-1]["props"].append(
                        {"name": tokens[2].decode(), "is_list": False, "type": tokens[1].decode()}
                    )
            elif keyword == b"end_header":
                break

        if fmt is None:
            raise ValueError(f"PLY missing format line: {mesh_file}")
        if fmt == "ascii":
            return _read_ply_vertices_xyz_ascii(handle, elements)
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"unsupported PLY format {fmt!r} in {mesh_file}")
        endian = "<" if fmt == "binary_little_endian" else ">"

        # --- vertex element ---
        vertex = next((e for e in elements if e["name"] == "vertex"), None)
        if vertex is None:
            raise ValueError(f"PLY has no vertex element: {mesh_file}")

        # Per-vertex layout: byte offset of x/y/z within the fixed-size record.
        offset = 0
        coord_offsets: dict[str, tuple[int, str]] = {}
        for prop in vertex["props"]:
            if prop["is_list"]:
                raise ValueError("list property in vertex element not supported")
            sfmt, size = _PLY_TYPES[prop["type"]]
            if prop["name"] in ("x", "y", "z"):
                coord_offsets[prop["name"]] = (offset, sfmt)
            offset += size
        record_size = offset
        for axis in ("x", "y", "z"):
            if axis not in coord_offsets:
                raise ValueError(f"PLY vertex missing {axis}: {mesh_file}")

        count = vertex["count"]
        buf = handle.read(record_size * count)
        if len(buf) < record_size * count:
            raise ValueError(f"PLY truncated: {mesh_file}")

        ox, fx = coord_offsets["x"]
        oy, fy = coord_offsets["y"]
        oz, fz = coord_offsets["z"]
        ux = struct.Struct(endian + fx)
        uy = struct.Struct(endian + fy)
        uz = struct.Struct(endian + fz)
        verts: list[tuple[float, float, float]] = []
        base = 0
        for _ in range(count):
            x = ux.unpack_from(buf, base + ox)[0]
            y = uy.unpack_from(buf, base + oy)[0]
            z = uz.unpack_from(buf, base + oz)[0]
            verts.append((x, y, z))
            base += record_size
        return verts


def _read_ply_vertices_xyz_ascii(handle: Any, elements: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    vertex = next((e for e in elements if e["name"] == "vertex"), None)
    if vertex is None:
        raise ValueError("PLY has no vertex element")
    idx = {p["name"]: i for i, p in enumerate(vertex["props"]) if not p["is_list"]}
    for axis in ("x", "y", "z"):
        if axis not in idx:
            raise ValueError(f"PLY vertex missing {axis}")
    verts: list[tuple[float, float, float]] = []
    for _ in range(vertex["count"]):
        parts = handle.readline().split()
        verts.append((float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])))
    return verts


def _read_aggregation(agg_file: Path) -> dict[int, list[int]]:
    """objectId -> list of segment ids (raw objectId, not shifted)."""

    data = json.loads(Path(agg_file).read_text(encoding="utf-8"))
    object_id_to_segs: dict[int, list[int]] = {}
    for group in data["segGroups"]:
        object_id_to_segs[int(group["objectId"])] = list(group["segments"])
    return object_id_to_segs


def _read_segmentation(seg_file: Path) -> dict[int, list[int]]:
    """segment id -> list of vertex indices."""

    data = json.loads(Path(seg_file).read_text(encoding="utf-8"))
    seg_to_verts: dict[int, list[int]] = {}
    for vert_idx, seg_id in enumerate(data["segIndices"]):
        seg_to_verts.setdefault(int(seg_id), []).append(vert_idx)
    return seg_to_verts


def _read_axis_align_matrix(meta_file: Path) -> list[list[float]] | None:
    if not Path(meta_file).exists():
        return None
    for line in Path(meta_file).read_text(encoding="utf-8").splitlines():
        if "axisAlignment" in line:
            values = [float(x) for x in line.split("=", 1)[1].split()]
            if len(values) == 16:
                return [values[i * 4 : i * 4 + 4] for i in range(4)]
    return None


def _apply_matrix(matrix: list[list[float]], pt: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = pt
    hx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    hy = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    hz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    return (hx, hy, hz)


@lru_cache(maxsize=8)
def _scene_instance_bboxes(
    scene_id: str, scans_root_str: str, apply_axis_align: bool
) -> dict[int, list[float]]:
    """objectId -> corner bbox [xmin,ymin,zmin,xmax,ymax,zmax] for one scene.

    Cached per (scene, root, alignment) since reading the PLY is the costly step.
    """

    paths = scene_paths(scene_id, scans_root_str)
    for key in ("mesh", "aggregation", "segs"):
        if not paths[key].exists():
            raise ScanNetSceneNotFound(f"{key} file missing for {scene_id}: {paths[key]}")

    verts = _read_ply_vertices_xyz(paths["mesh"])
    if apply_axis_align:
        matrix = _read_axis_align_matrix(paths["meta"])
        if matrix is not None:
            verts = [_apply_matrix(matrix, v) for v in verts]

    object_id_to_segs = _read_aggregation(paths["aggregation"])
    seg_to_verts = _read_segmentation(paths["segs"])

    bboxes: dict[int, list[float]] = {}
    for object_id, segs in object_id_to_segs.items():
        vert_indices: list[int] = []
        for seg in segs:
            vert_indices.extend(seg_to_verts.get(int(seg), ()))
        if not vert_indices:
            continue
        xs = [verts[i][0] for i in vert_indices]
        ys = [verts[i][1] for i in vert_indices]
        zs = [verts[i][2] for i in vert_indices]
        bboxes[object_id] = [
            min(xs), min(ys), min(zs), max(xs), max(ys), max(zs),
        ]
    return bboxes


def resolve_instance_bbox(
    scene_id: str,
    object_id: int | str,
    *,
    scans_root: Path | str | None = None,
    apply_axis_align: bool = False,
) -> list[float] | None:
    """Corner bbox ``[xmin,ymin,zmin,xmax,ymax,zmax]`` for a ScanNet instance.

    Returns ``None`` if the object id is absent from the scene's aggregation.
    Raises :class:`ScanNetSceneNotFound` if the scene files are missing.
    """

    root_str = str(scans_root) if scans_root is not None else str(DEFAULT_SCANNET_SCANS_ROOT)
    bboxes = _scene_instance_bboxes(scene_id, root_str, apply_axis_align)
    return bboxes.get(int(object_id))


def resolve_scene_bboxes(
    scene_id: str,
    *,
    scans_root: Path | str | None = None,
    apply_axis_align: bool = False,
) -> dict[int, list[float]]:
    """All instance corner bboxes for a scene, keyed by objectId."""

    root_str = str(scans_root) if scans_root is not None else str(DEFAULT_SCANNET_SCANS_ROOT)
    return dict(_scene_instance_bboxes(scene_id, root_str, apply_axis_align))
