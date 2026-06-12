from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


METHOD_ALIASES = {
    "claws_spatial_rag": "claws",
    "dualmap": "dualmap",
    "hovsg": "hovsg",
    "llm_with_memory": "llm-with-memory",
}


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "unknown"


def method_name_from_spec(
    method_spec: str,
    method_kwargs: Optional[Mapping[str, Any]] = None,
) -> str:
    if method_kwargs:
        explicit = method_kwargs.get("result_method") or method_kwargs.get("method_name")
        if explicit:
            return slugify(str(explicit))

        base_method = method_kwargs.get("base_method")
        if base_method:
            return method_name_from_spec(str(base_method))

    module_name = method_spec.split(":", 1)[0]
    leaf = module_name.rsplit(".", 1)[-1]
    return METHOD_ALIASES.get(leaf, slugify(leaf))


def timestamped_result_dir(
    method: str,
    evaluation: str,
    *,
    timestamp: Optional[str] = None,
    root: Path = Path("results"),
) -> Path:
    return _timestamped_dir(root, method, evaluation, timestamp)


def timestamped_memory_dir(
    method: str,
    evaluation: str,
    *,
    timestamp: Optional[str] = None,
    root: Path = Path("memories"),
) -> Path:
    return _timestamped_dir(root, method, evaluation, timestamp)


def method_name_from_results_path(path: Path, *, root_name: str = "results") -> Optional[str]:
    parts = path.parts
    try:
        index = parts.index(root_name)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return slugify(parts[index + 1])


def _timestamped_dir(
    root: Path,
    method: str,
    evaluation: str,
    timestamp: Optional[str],
) -> Path:
    return root / slugify(method) / slugify(evaluation) / (timestamp or run_timestamp())
