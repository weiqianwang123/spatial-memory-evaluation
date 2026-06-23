"""Track 3 (OpenEQA general spatial QA) benchmark data.

OpenEQA covers two datasets: ScanNet (`scannet-v0`) and HM3D (`hm3d-v0`). The
canonical question file (question + answer + category + episode_history) is the
OpenEQA v0 release. On this machine it currently lives in the OpenEQA repo:

```text
/home/robin_wang/open-eqa/data/open-eqa-v0.json   # 1636 questions (1079 scannet, 557 hm3d)
```

Per-episode posed RGB-D / memory inputs available on NAS so far:

```text
openeqa_scannet_dbs/    # ClawS-style sqlite-vec memory DB per scannet episode (currently 1)
openeqa_frames/         # scannet-v0 frames
openeqa_scannet_rgbd/   # scannet-v0 posed RGB-D
# HM3D inputs: not yet on NAS (see .codex/path_registry.md).
```

This builder turns the OpenEQA question file into a benchmark directory:

```text
benchmarks/track3/openeqa/<dataset>/
  questions.jsonl   # question_id, question, category, episode_id   (NO answers)
  answers.jsonl     # question_id, answer, category                 (evaluator-only)
  metadata.json
```

GT answers stay in ``answers.jsonl`` and are read only by the evaluator / LLM
judge. They are never copied into a tool-LLM sandbox.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.jsonl import read_json, read_jsonl, write_json, write_jsonl


DEFAULT_OPENEQA_QUESTIONS = Path("/home/robin_wang/open-eqa/data/open-eqa-v0.json")
QUESTIONS_FILE = "questions.jsonl"
ANSWERS_FILE = "answers.jsonl"

# OpenEQA episode_history prefix -> our dataset split name.
DATASET_PREFIX = {
    "scannet-v0": "scannet",
    "hm3d-v0": "hm3d",
}


def track3_data_status(benchmark_dir: Path) -> dict[str, Any]:
    """Report whether the OpenEQA questions benchmark exists at ``benchmark_dir``."""

    questions_path = benchmark_dir / QUESTIONS_FILE
    if questions_path.exists():
        try:
            count = len(read_jsonl(questions_path))
        except (ValueError, OSError) as exc:
            return {
                "status": "data_unavailable",
                "reason": f"failed to read {questions_path}: {exc}",
                "benchmark_dir": str(benchmark_dir),
            }
        return {
            "status": "ok",
            "benchmark_dir": str(benchmark_dir),
            "question_count": count,
            "questions_path": str(questions_path),
            "answers_path": str(benchmark_dir / ANSWERS_FILE),
        }
    return {
        "status": "data_unavailable",
        "reason": (
            "OpenEQA questions not found. Build with build_track3_data using the "
            f"OpenEQA v0 question file (default {DEFAULT_OPENEQA_QUESTIONS}); see "
            ".codex/path_registry.md."
        ),
        "benchmark_dir": str(benchmark_dir),
        "expected_files": [QUESTIONS_FILE, ANSWERS_FILE],
        "openeqa_questions_default": str(DEFAULT_OPENEQA_QUESTIONS),
    }


def build_track3_data(
    *,
    openeqa_questions: Path,
    output_root: Path,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Split the OpenEQA v0 question file into per-dataset benchmark dirs.

    ``dataset`` filters to ``scannet`` or ``hm3d``; ``None`` builds both. Each
    output dir gets ``questions.jsonl`` (no answers) and ``answers.jsonl``
    (evaluator-only). HM3D rows are included if present in the question file even
    when HM3D posed RGB-D inputs are not yet on NAS; the QA file is the same.
    """

    rows = read_json(openeqa_questions)
    if not isinstance(rows, list):
        raise ValueError(f"OpenEQA question file must be a JSON list: {openeqa_questions}")

    per_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prefix = str(row.get("episode_history", "")).split("/", 1)[0]
        ds = DATASET_PREFIX.get(prefix)
        if ds is None:
            continue
        if dataset is not None and ds != dataset:
            continue
        per_dataset.setdefault(ds, []).append(row)

    summaries: dict[str, Any] = {}
    for ds, ds_rows in per_dataset.items():
        out_dir = output_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        questions = []
        answers = []
        for row in ds_rows:
            question_id = str(row.get("question_id"))
            questions.append(
                {
                    "question_id": question_id,
                    "question": row.get("question"),
                    "category": row.get("category"),
                    "episode_id": row.get("episode_history"),
                }
            )
            answers.append(
                {
                    "question_id": question_id,
                    "answer": row.get("answer"),
                    "category": row.get("category"),
                }
            )
        write_jsonl(out_dir / QUESTIONS_FILE, questions)
        write_jsonl(out_dir / ANSWERS_FILE, answers)
        summary = {
            "dataset": ds,
            "question_count": len(questions),
            "categories": dict(Counter(q["category"] for q in questions)),
            "source": str(openeqa_questions),
            "output_dir": str(out_dir),
        }
        write_json(out_dir / "metadata.json", summary)
        summaries[ds] = summary
    return {"output_root": str(output_root), "datasets": summaries}
