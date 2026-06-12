from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda x, **_: x

from .method_loader import load_method, parse_method_kwargs
from .rgbd import iter_episode_histories, load_open_eqa_dataset, load_rgbd_sequence, questions_for_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run get_memory_text on OpenEQA questions for RGB-D episodes."
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
        "--dataset",
        type=Path,
        default=Path("data/open-eqa-v0.json"),
        help="OpenEQA dataset JSON",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=Path("data/frames"),
        help="root containing extracted episode histories",
    )
    parser.add_argument(
        "--episode-prefix",
        default="scannet-v0",
        help="only run episodes with this prefix; use '' for all episodes",
    )
    parser.add_argument(
        "--episode-history",
        action="append",
        default=None,
        help="only run this exact episode_history. Can be passed more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spatial-memory-evaluation/results/memory-predictions.json"),
        help="output predictions JSON in OpenEQA format",
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
        help="only process the first 5 questions",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue after per-question method errors",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_open_eqa_dataset(args.dataset)
    episode_prefix = args.episode_prefix or None
    method_kwargs = parse_method_kwargs(args.method_kwargs)
    results = _load_existing_results(args.output)
    completed = {item["question_id"] for item in results}
    total_written = len(results)

    question_budget = 5 if args.dry_run else None
    processed_questions = 0

    if args.episode_history:
        wanted = set(args.episode_history)
        episodes = [
            episode
            for episode in iter_episode_histories(dataset, episode_prefix=None)
            if episode in wanted
        ]
    else:
        episodes = list(iter_episode_histories(dataset, episode_prefix=episode_prefix))
    print(f"found {len(episodes):,} episode histories")
    print(f"found {len(completed):,} existing memory predictions")

    for episode_history in tqdm(episodes):
        episode_questions = [
            item
            for item in questions_for_episode(dataset, episode_history)
            if item["question_id"] not in completed
        ]
        if question_budget is not None:
            remaining = question_budget - processed_questions
            if remaining <= 0:
                break
            episode_questions = episode_questions[:remaining]
        if not episode_questions:
            continue

        sequence = load_rgbd_sequence(
            frames_root=args.frames_root,
            episode_history=episode_history,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )
        method = load_method(args.method, sequence=sequence, method_kwargs=method_kwargs)

        for item in episode_questions:
            answer = _call_get_memory_text(method, item, force=args.force)
            results.append(
                {
                    "question_id": item["question_id"],
                    "episode_history": episode_history,
                    "category": item.get("category"),
                    "answer": answer,
                }
            )
            completed.add(item["question_id"])
            total_written += 1
            processed_questions += 1
            _write_json(args.output, results)

    _write_json(args.output, results)
    print(f"saving {total_written:,} memory predictions to {args.output}")


def _call_get_memory_text(method: Any, item: Dict[str, Any], force: bool) -> str:
    try:
        answer = method.get_memory_text(question=item["question"])
        return "" if answer is None else str(answer)
    except TypeError:
        try:
            answer = method.get_memory_text(item["question"])
            return "" if answer is None else str(answer)
        except Exception as exc:
            return _handle_method_error(exc, force=force)
    except Exception as exc:
        return _handle_method_error(exc, force=force)


def _handle_method_error(exc: Exception, force: bool) -> str:
    if not force:
        traceback.print_exc()
        raise exc
    return ""


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


if __name__ == "__main__":
    main(parse_args())
