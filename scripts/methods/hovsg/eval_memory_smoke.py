from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.memory_package_validator import validate_package
from spatial_memory_evaluation.output_paths import timestamped_result_dir


DEFAULT_SCENE_ID = "036bce3393"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a HOV-SG memory package and run a smoke eval.")
    parser.add_argument("package_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--query", action="append", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    package_dir = args.package_dir or _latest_package(args.scene_id)
    report = validate_package(package_dir)
    if not report.valid:
        print(json.dumps(report.to_json(), indent=2))
        return 1

    manifest = _read_json(package_dir / "manifest.json")
    capabilities = _read_json(package_dir / "capabilities.json")

    summary: dict[str, Any] = {
        "status": "ok",
        "package_dir": str(package_dir),
        "method": manifest["method"]["name"],
        "dataset": manifest["dataset"],
        "validation": report.to_json(),
        "track1_memory_construction": _eval_track1(package_dir, capabilities),
        "track2_object_location": _eval_track2(package_dir, capabilities, args.query or ["object"], args.top_k),
    }

    output = args.output
    if output is None:
        output = timestamped_result_dir("hovsg", "hovsg-memory-smoke") / "eval_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, summary)
    print(json.dumps({"status": "ok", "output": str(output), "summary": summary}, indent=2))
    return 0


def _eval_track1(package_dir: Path, capabilities: dict[str, Any]) -> dict[str, Any]:
    cap = capabilities["fixed_api"]["track1_memory_construction"]
    if cap["status"] != "supported":
        return {"status": "invalid", "reason": cap.get("reason")}
    result = _load_entrypoint(package_dir, cap["entrypoint"])(str(package_dir), {})
    objects = result.get("objects", [])
    bbox_count = sum(1 for obj in objects if obj.get("bbox_3d"))
    position_count = sum(1 for obj in objects if obj.get("position_3d"))
    return {
        "status": result.get("status", "ok"),
        "object_count": len(objects),
        "bbox_count": bbox_count,
        "position_count": position_count,
        "preview": objects[:3],
    }


def _eval_track2(
    package_dir: Path,
    capabilities: dict[str, Any],
    queries: list[str],
    top_k: int,
) -> dict[str, Any]:
    cap = capabilities["fixed_api"]["track2_object_location"]
    if cap["status"] != "supported":
        return {"status": "invalid", "reason": cap.get("reason")}
    query_object = _load_entrypoint(package_dir, cap["entrypoint"])
    results = []
    for query in queries:
        result = query_object(str(package_dir), {"query": query, "top_k": top_k})
        results.append(
            {
                "query": query,
                "status": result.get("status", "ok"),
                "prediction_count": len(result.get("predictions", [])),
                "predictions": result.get("predictions", [])[:top_k],
            }
        )
    return {"status": "ok", "queries": results}


def _load_entrypoint(package_dir: Path, entrypoint: str) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    module_path_text, function_name = entrypoint.split(":", 1)
    module_path = package_dir / module_path_text
    module_name = f"memory_package_{module_path.stem}_{datetime.now().timestamp()}".replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load entrypoint module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, function_name)
    if not callable(func):
        raise TypeError(f"entrypoint is not callable: {entrypoint}")
    return func


def _latest_package(scene_id: str) -> Path:
    root = REPO_ROOT / "memories" / "hovsg" / "scannetpp" / scene_id
    candidates = sorted([path for path in root.glob("*") if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no HOV-SG packages found under {root}")
    return candidates[-1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
