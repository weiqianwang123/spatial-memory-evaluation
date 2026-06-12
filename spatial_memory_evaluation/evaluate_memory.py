from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x

from .llm_match import get_llm_match_score
from .output_paths import method_name_from_results_path, timestamped_result_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate get_memory_text predictions with internal LLM-Match."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="predictions produced by spatial_memory_evaluation.run_memory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/open-eqa-v0.json"),
        help="OpenEQA dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "metrics JSON. Default: "
            "results/<method>/llm-match/<timestamp>/metrics.json when method can be inferred"
        ),
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="method folder name for default output path when it cannot be inferred from predictions",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require predictions for every dataset question",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only evaluate the first 5 predictions",
    )
    parser.add_argument("--openai-key", default=None, help="OpenAI API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--openai-model", default="gpt-4-1106-preview")
    parser.add_argument("--openai-seed", type=int, default=1234)
    parser.add_argument("--openai-max-tokens", type=int, default=32)
    parser.add_argument("--openai-temperature", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    output = args.output
    if output is None:
        method_name = args.method_name or method_name_from_results_path(args.predictions) or "unknown"
        output = timestamped_result_dir(method_name, "llm-match") / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = _load_list(args.dataset)
    predictions = _load_list(args.predictions)
    question_id_to_item = {item["question_id"]: item for item in dataset}
    prediction_ids = [item["question_id"] for item in predictions]
    question_id_to_prediction = {item["question_id"]: item for item in predictions}

    if args.strict:
        dataset_ids = {item["question_id"] for item in dataset}
        assert dataset_ids == set(prediction_ids)

    scores = _load_scores(output)
    print(f"found {len(dataset):,} OpenEQA questions")
    print(f"found {len(predictions):,} memory predictions")
    print(f"found {len(scores):,} existing scores")

    ids_to_score = prediction_ids[:5] if args.dry_run else prediction_ids
    for question_id in tqdm(ids_to_score):
        if question_id in scores:
            continue
        if question_id not in question_id_to_item:
            raise KeyError(f"{question_id} not found in {args.dataset}")

        item = question_id_to_item[question_id]
        prediction = question_id_to_prediction[question_id]
        score = get_llm_match_score(
            question=item["question"],
            answer=item["answer"],
            prediction=_trim_prediction(prediction.get("answer", "")),
            extra_answers=item.get("extra_answers"),
            openai_key=args.openai_key,
            openai_model=args.openai_model,
            openai_seed=args.openai_seed,
            openai_max_tokens=args.openai_max_tokens,
            openai_temperature=args.openai_temperature,
            verbose=args.verbose,
        )
        scores[question_id] = score
        _write_json(output, scores)

    _write_json(output, scores)
    print(f"final score: {_mean_normalized_score(scores):.1f}")
    print(f"saving {len(scores):,} scores to {output}")


def _load_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _load_scores(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict in {path}")
    return data


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as f:
        json.dump(value, f, indent=2)


def _trim_prediction(answer: Optional[str]) -> str:
    if not answer:
        return ""
    end_idx = answer.rfind(".")
    if end_idx >= 0 and end_idx + 1 < len(answer):
        return answer[: end_idx + 1]
    return answer


def _mean_normalized_score(scores: Dict[str, float]) -> float:
    if not scores:
        return 0.0
    normalized = [
        100.0 * (min(max(float(score), 1.0), 5.0) - 1.0) / 4.0
        for score in scores.values()
    ]
    return sum(normalized) / len(normalized)


if __name__ == "__main__":
    main(parse_args())
