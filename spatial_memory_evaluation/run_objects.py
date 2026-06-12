from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x

from .interfaces import ObjectPrediction
from .method_loader import load_method, parse_method_kwargs
from .output_paths import method_name_from_spec, timestamped_result_dir
from .rgbd import load_rgbd_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run get_object on ScanNet-style object queries."
    )
    parser.add_argument(
        "--method",
        required=True,
        help="method adapter as 'module:attribute', e.g. examples.dummy_method:create_method",
    )
    parser.add_argument(
        "--method-kwargs",
        default=None,
        help="JSON object or path to JSON passed to the adapter factory",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="object-query JSON file with episode_history and query fields",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("data/frames"),
        help="root containing extracted episode histories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output object predictions JSON. Default: "
            "results/<method>/object-query/<timestamp>/predictions.json"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="where an external ScanNet evaluator should write metrics. Default: output dir/metrics.json",
    )
    parser.add_argument(
        "--scannet-evaluator-cmd",
        default=None,
        help=(
            "optional command for your existing ScanNet evaluator. "
            "Placeholders: {predictions}, {queries}, {output}"
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="maximum RGB-D frames passed to the method for each episode",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="keep every Nth frame from each RGB-D sequence",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 object queries",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue after per-query method errors",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    method_kwargs = parse_method_kwargs(args.method_kwargs)
    if args.output is None:
        method_name = method_name_from_spec(args.method, method_kwargs)
        args.output = timestamped_result_dir(method_name, "object-query") / "predictions.json"
    if args.metrics_output is None:
        args.metrics_output = args.output.parent / "metrics.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(args.queries)
    if args.dry_run:
        queries = queries[:5]

    results = _load_existing_results(args.output)
    completed = {item["query_id"] for item in results}

    print(f"found {len(queries):,} object queries")
    print(f"found {len(completed):,} existing object predictions")

    for episode_history, episode_queries in tqdm(_group_by_episode(queries).items()):
        pending_queries = [
            query for query in episode_queries if _query_id(query) not in completed
        ]
        if not pending_queries:
            continue

        sequence = load_rgbd_sequence(
            frames_root=args.frames_root,
            episode_history=episode_history,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )
        method = load_method(args.method, sequence=sequence, method_kwargs=method_kwargs)

        for query in pending_queries:
            query_id = _query_id(query)
            query_text = _query_text(query)
            objects = _call_get_object(method, query_text=query_text, force=args.force)
            results.append(
                {
                    "query_id": query_id,
                    "episode_history": episode_history,
                    "query": query_text,
                    "objects": [_object_to_json(obj) for obj in objects],
                }
            )
            completed.add(query_id)
            _write_json(args.output, results)

    _write_json(args.output, results)
    print(f"saving {len(results):,} object predictions to {args.output}")

    if args.scannet_evaluator_cmd:
        _run_external_evaluator(args)


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of object queries in {path}")

    normalized = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Object query #{idx} must be a JSON object")
        if "episode_history" not in item:
            raise ValueError(f"Object query #{idx} missing episode_history")
        _query_text(item)
        if "query_id" not in item:
            item = dict(item)
            item["query_id"] = item.get("id") or f"{item['episode_history']}::{idx}"
        normalized.append(item)
    return normalized


def _group_by_episode(queries: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = {}
    for query in queries:
        grouped.setdefault(query["episode_history"], []).append(query)
    return grouped


def _query_id(query: Mapping[str, Any]) -> str:
    return str(query.get("query_id") or query.get("id"))


def _query_text(query: Mapping[str, Any]) -> str:
    value = query.get("query") or query.get("label") or query.get("object")
    if value is None:
        raise ValueError(f"Object query missing query/label/object: {query}")
    return str(value)


def _call_get_object(method: Any, query_text: str, force: bool) -> List[Any]:
    try:
        return _normalize_objects(method.get_object(query=query_text))
    except TypeError:
        try:
            return _normalize_objects(method.get_object(query_text))
        except Exception as exc:
            return _handle_method_error(exc, force=force)
    except Exception as exc:
        return _handle_method_error(exc, force=force)


def _normalize_objects(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (ObjectPrediction, Mapping, str)):
        return [value]
    return list(value)


def _handle_method_error(exc: Exception, force: bool) -> List[Any]:
    if not force:
        traceback.print_exc()
        raise exc
    return []


def _object_to_json(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, ObjectPrediction):
        return obj.to_json()
    if hasattr(obj, "to_json") and callable(obj.to_json):
        return obj.to_json()
    if isinstance(obj, Mapping):
        return dict(obj)
    return {"label": str(obj), "score": 1.0}


def _load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _write_json(path: Path, value: Sequence[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        json.dump(list(value), f, indent=2)


def _run_external_evaluator(args: argparse.Namespace) -> None:
    command = args.scannet_evaluator_cmd.format(
        predictions=str(args.output),
        queries=str(args.queries),
        output=str(args.metrics_output),
    )
    print(f"running ScanNet evaluator: {command}")
    subprocess.run(command, shell=True, check=True)


if __name__ == "__main__":
    main(parse_args())
