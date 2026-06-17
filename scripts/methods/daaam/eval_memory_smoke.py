from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_memory_evaluation.common.reporting import evaluation_output_paths  # noqa: E402
from spatial_memory_evaluation.memory_package_validator import validate_package  # noqa: E402
from spatial_memory_evaluation.output_paths import timestamped_result_dir  # noqa: E402
from spatial_memory_evaluation.track1.data import DEFAULT_SCENE_ID  # noqa: E402
from spatial_memory_evaluation.track1.evaluator import evaluate_track1  # noqa: E402
from spatial_memory_evaluation.track2.evaluator import evaluate_track2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a DAAAM package and run Track 1/2 smoke evals.")
    parser.add_argument("package_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    package_dir = args.package_dir or _latest_package(args.scene_id)
    report = validate_package(package_dir)
    if not report.valid:
        print(json.dumps(report.to_json(), indent=2))
        return 1

    output_dir = args.output_dir or timestamped_result_dir("daaam", "daaam-memory-smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    track1_output = output_dir / "track1_eval_summary.json"
    track2_output = output_dir / "track2_eval_summary.json"

    benchmark_track1 = Path("benchmarks") / "track1" / "scannetpp" / args.scene_id
    benchmark_track2 = Path("benchmarks") / "track2" / "scannetpp" / args.scene_id

    track1_summary = evaluate_track1(
        package_dir=package_dir,
        benchmark_dir=benchmark_track1,
        mode="fixed_api",
        output=track1_output,
    )
    track2_summary = evaluate_track2(
        package_dir=package_dir,
        benchmark_dir=benchmark_track2,
        track1_benchmark_dir=benchmark_track1,
        mode="fixed_api",
        output=track2_output,
    )

    result = {
        "status": "ok",
        "package_dir": str(package_dir),
        "validation": report.to_json(),
        "track1": _result_paths(track1_output, track1_summary),
        "track2": _result_paths(track2_output, track2_summary),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _result_paths(summary_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    paths = evaluation_output_paths(summary_path)
    return {
        "summary": str(paths.summary),
        "details": str(paths.details),
        "report": str(paths.report),
        "status": summary.get("status"),
        "metrics": summary.get("metrics"),
        "result": summary.get("result"),
    }


def _latest_package(scene_id: str) -> Path:
    root = REPO_ROOT / "memories" / "daaam" / "scannetpp" / scene_id
    candidates = sorted([path for path in root.glob("*") if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no DAAAM packages found under {root}")
    return candidates[-1]


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
