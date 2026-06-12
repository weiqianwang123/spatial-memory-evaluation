from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional

from spatial_memory_evaluation import ObjectPrediction, RGBDSequence
from spatial_memory_evaluation.method_loader import load_method


DEFAULT_PROMPT = """You are an embodied question answering agent.
You will be given a question about an indoor scene and a scene memory extracted
from a posed RGB-D sequence. Use the scene memory as the primary evidence. If
the memory is incomplete, make a reasonable guess from the evidence and common
indoor-scene knowledge.

Answer with a short phrase or sentence. Do not mention the memory.

Scene memory:
{memory}

Q: {question}
A:"""


@dataclass(frozen=True)
class LLMMemoryAnswererConfig:
    base_method: str
    base_method_kwargs: Mapping[str, Any]
    answer_mode: str = "context"
    openai_model: str = "gpt-4-0613"
    openai_seed: int = 1234
    openai_temperature: float = 0.2
    openai_max_tokens: int = 64
    openai_key: Optional[str] = None
    prompt_template: str = DEFAULT_PROMPT
    memory_db: Optional[Path] = None
    memory_top_k: int = 80
    max_context_chars: int = 12000
    prefer_exported_memory: bool = True
    include_positions: bool = True


class LLMMemoryAnsweringMethod:
    """Wrap any spatial-memory adapter and answer OpenEQA with an LLM.

    The wrapped method owns memory construction/loading. This class turns that
    memory into a scene-graph-like text context and optionally calls an LLM.
    """

    def __init__(self, sequence: RGBDSequence, config: LLMMemoryAnswererConfig) -> None:
        self.sequence = sequence
        self.config = config
        self.base_method = load_method(
            config.base_method,
            sequence=sequence,
            method_kwargs=config.base_method_kwargs,
        )
        self._records: Optional[List[dict[str, Any]]] = None

    def get_memory_text(self, question: str) -> str:
        memory = self._memory_context(question)
        if self.config.answer_mode == "context":
            return self._format_debug_context(question, memory)
        if self.config.answer_mode == "openai":
            return self._answer_with_openai(question, memory)
        raise ValueError("answer_mode must be one of: context, openai")

    def get_object(self, query: str):
        return self.base_method.get_object(query)

    def _memory_context(self, question: str) -> str:
        records = self._memory_records()
        if records:
            selected = _rank_records(question, records)[: self.config.memory_top_k]
            return _format_records(
                selected,
                total_count=len(records),
                max_chars=self.config.max_context_chars,
                include_positions=self.config.include_positions,
            )

        raw = _call_base_memory_text(self.base_method, question)
        return _truncate(str(raw or "No structured scene memory available."), self.config.max_context_chars)

    def _memory_records(self) -> List[dict[str, Any]]:
        if self._records is not None:
            return self._records

        db_path = self._resolved_memory_db()
        if self.config.prefer_exported_memory and hasattr(self.base_method, "export_spatial_memory_db"):
            if db_path is None:
                db_path = Path("spatial-memory-evaluation/results/llm-memory-export.db")
            exported = self.base_method.export_spatial_memory_db(db_path)
            self._records = _load_memory_db(Path(exported))
            return self._records

        candidate = db_path or _base_method_memory_db(self.base_method)
        if candidate is not None and candidate.exists():
            self._records = _load_memory_db(candidate)
            return self._records

        # Some methods, notably ClawS, build their DB lazily on first memory query.
        _call_base_memory_text(self.base_method, "List the important objects in this scene.")
        candidate = db_path or _base_method_memory_db(self.base_method)
        if candidate is not None and candidate.exists():
            self._records = _load_memory_db(candidate)
            return self._records

        self._records = []
        return self._records

    def _resolved_memory_db(self) -> Optional[Path]:
        if self.config.memory_db is None:
            return None
        raw = str(self.config.memory_db)
        scene_id = _scene_id_from_sequence(self.sequence)
        episode = _safe_path_component(self.sequence.episode_history)
        if "{" in raw and "}" in raw:
            raw = raw.format(
                episode_history=self.sequence.episode_history,
                episode=episode,
                scene_id=scene_id,
            )
        return Path(raw).expanduser()

    def _answer_with_openai(self, question: str, memory: str) -> str:
        from openai import OpenAI

        key = self.config.openai_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required when answer_mode=openai")
        prompt = self.config.prompt_template.format(question=question, memory=memory)
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[{"role": "user", "content": prompt}],
            seed=self.config.openai_seed,
            temperature=self.config.openai_temperature,
            max_tokens=self.config.openai_max_tokens,
        )
        text = response.choices[0].message.content or ""
        return _parse_answer(text)

    @staticmethod
    def _format_debug_context(question: str, memory: str) -> str:
        return "\n".join(
            [
                "LLM answer mode is disabled; this is the memory context that would be passed to the LLM.",
                f"Question: {question}",
                "",
                memory,
            ]
        )


def create_method(sequence: RGBDSequence, **kwargs: Any) -> LLMMemoryAnsweringMethod:
    base_method = str(kwargs["base_method"])
    base_method_kwargs = _load_nested_kwargs(kwargs)
    prompt_template = _load_prompt_template(kwargs)
    memory_db = kwargs.get("memory_db")
    config = LLMMemoryAnswererConfig(
        base_method=base_method,
        base_method_kwargs=base_method_kwargs,
        answer_mode=str(kwargs.get("answer_mode", "context")),
        openai_model=str(kwargs.get("openai_model", "gpt-4-0613")),
        openai_seed=int(kwargs.get("openai_seed", 1234)),
        openai_temperature=float(kwargs.get("openai_temperature", 0.2)),
        openai_max_tokens=int(kwargs.get("openai_max_tokens", 64)),
        openai_key=kwargs.get("openai_key"),
        prompt_template=prompt_template,
        memory_db=Path(memory_db) if memory_db else None,
        memory_top_k=int(kwargs.get("memory_top_k", 80)),
        max_context_chars=int(kwargs.get("max_context_chars", 12000)),
        prefer_exported_memory=bool(kwargs.get("prefer_exported_memory", True)),
        include_positions=bool(kwargs.get("include_positions", True)),
    )
    return LLMMemoryAnsweringMethod(sequence=sequence, config=config)


def _load_nested_kwargs(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    path = kwargs.get("base_method_kwargs_path")
    if path:
        with Path(path).expanduser().open("r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")
        return data
    value = kwargs.get("base_method_kwargs", {})
    if isinstance(value, str):
        path = Path(value)
        if path.exists():
            with path.open("r") as f:
                value = json.load(f)
        else:
            value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("base_method_kwargs must be a JSON object or path")
    return dict(value)


def _load_prompt_template(kwargs: Mapping[str, Any]) -> str:
    prompt_path = kwargs.get("prompt_path")
    if prompt_path:
        return Path(prompt_path).expanduser().read_text()
    return str(kwargs.get("prompt_template", DEFAULT_PROMPT))


def _call_base_memory_text(method: Any, question: str) -> str:
    try:
        return str(method.get_memory_text(question=question) or "")
    except TypeError:
        return str(method.get_memory_text(question) or "")
    except NotImplementedError:
        return ""


def _base_method_memory_db(method: Any) -> Optional[Path]:
    config = getattr(method, "config", None)
    memory_db = getattr(config, "memory_db", None)
    return Path(memory_db) if memory_db else None


def _load_memory_db(path: Path) -> List[dict[str, Any]]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception:
            pass
        table = _choose_memory_table(conn)
        rows = _fetch_memory_rows(conn, table)
    finally:
        conn.close()
    return [_record_from_row(row) for row in rows]


def _choose_memory_table(conn: sqlite3.Connection) -> str:
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
        ).fetchall()
    ]
    if "spatial_memories" in tables:
        return "spatial_memories"
    for table in sorted(tables):
        cols = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info([{table}])").fetchall()
        }
        if {"snapshot_text", "pos_x", "pos_y", "pos_z"}.issubset(cols):
            return table
    raise ValueError(f"No spatial memory table found in sqlite DB")


def _fetch_memory_rows(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    cols = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    }
    if "snapshot_text" in cols:
        id_expr = "memory_id" if "memory_id" in cols else "rowid"
        label_expr = "object_name" if "object_name" in cols else "NULL"
        scene_expr = "scene_id" if "scene_id" in cols else "NULL"
        time_expr = "timestamp" if "timestamp" in cols else "NULL"
        conf_expr = "confidence" if "confidence" in cols else "NULL"
        sql = (
            f"SELECT {id_expr} AS memory_id, {scene_expr} AS scene_id, "
            f"{label_expr} AS object_name, snapshot_text, pos_x, pos_y, pos_z, "
            f"{time_expr} AS timestamp, {conf_expr} AS confidence "
            f"FROM [{table}]"
        )
        try:
            return conn.execute(sql).fetchall()
        except Exception as exc:
            if "vec" not in str(exc).lower() and "no such module" not in str(exc).lower():
                raise

    aux_table = f"{table}_auxiliary"
    aux_cols = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info([{aux_table}])").fetchall()
    }
    if {"value00", "value01", "value02", "value03"}.issubset(aux_cols):
        sql = (
            f"SELECT rowid AS memory_id, NULL AS scene_id, NULL AS object_name, "
            f"value00 AS snapshot_text, value01 AS pos_x, value02 AS pos_y, "
            f"value03 AS pos_z, value04 AS timestamp, NULL AS confidence "
            f"FROM [{aux_table}]"
        )
        return conn.execute(sql).fetchall()
    raise ValueError(f"Could not read memory rows from {table}")


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    snapshot = str(row["snapshot_text"] or "")
    label = str(row["object_name"] or _extract_label(snapshot))
    return {
        "memory_id": str(row["memory_id"]),
        "scene_id": row["scene_id"],
        "label": label,
        "snapshot_text": snapshot,
        "pos_x": _to_float(row["pos_x"]),
        "pos_y": _to_float(row["pos_y"]),
        "pos_z": _to_float(row["pos_z"]),
        "timestamp": _to_float(row["timestamp"]),
        "confidence": _to_float(row["confidence"]),
    }


def _rank_records(question: str, records: List[dict[str, Any]]) -> List[dict[str, Any]]:
    query_tokens = _content_tokens(question)
    ranked = []
    for idx, record in enumerate(records):
        text = f"{record.get('label', '')} {record.get('snapshot_text', '')}"
        tokens = _content_tokens(text)
        overlap = len(query_tokens & tokens)
        label = str(record.get("label") or "").lower()
        label_bonus = 3 if label and label in question.lower() else 0
        score = overlap + label_bonus
        ranked.append((-score, idx, record))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def _format_records(
    records: List[dict[str, Any]],
    *,
    total_count: int,
    max_chars: int,
    include_positions: bool,
) -> str:
    lines = [f"Scene graph / object memory entries shown: {len(records)} of {total_count}."]
    for idx, record in enumerate(records, 1):
        label = str(record.get("label") or "object")
        pieces = [f"{idx}. Object: {label}"]
        if include_positions and record.get("pos_x") is not None:
            pieces.append(
                "Position: "
                f"({record['pos_x']:.2f}, {record['pos_y']:.2f}, {record['pos_z']:.2f})"
            )
        snapshot = _single_line(record.get("snapshot_text", ""))
        if snapshot and snapshot.lower() != label.lower():
            pieces.append(f"Memory: {snapshot}")
        line = " | ".join(pieces)
        if sum(len(item) + 1 for item in lines) + len(line) > max_chars:
            lines.append("...")
            break
        lines.append(line)
    return "\n".join(lines)


def _parse_answer(text: str) -> str:
    text = str(text).strip()
    match = re.search(r"(?:^|\n)\s*A:\s*(.+)", text)
    if match:
        text = match.group(1).strip()
    return text.splitlines()[0].strip() if text else ""


def _extract_label(snapshot_text: str) -> str:
    object_line = re.search(r"^\s*object\s*:\s*(.+?)\s*$", snapshot_text, re.I | re.M)
    if object_line:
        return object_line.group(1).strip()
    bold = re.search(r"\*\*([^*]+)\*\*", snapshot_text)
    if bold:
        return bold.group(1).strip()
    first_line = snapshot_text.strip().splitlines()[0] if snapshot_text.strip() else "object"
    return first_line[:80]


def _content_tokens(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "do",
        "does",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "there",
        "to",
        "what",
        "where",
        "which",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in stop
    }


def _single_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: str, max_chars: int) -> str:
    value = str(value)
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 4)].rstrip() + "\n..."


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scene_id_from_sequence(sequence: RGBDSequence) -> str:
    episode = sequence.episode_history.split("/")[-1]
    if "scannet-" in episode:
        return episode.split("scannet-")[-1]
    return episode


def _safe_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value.strip("/"))
