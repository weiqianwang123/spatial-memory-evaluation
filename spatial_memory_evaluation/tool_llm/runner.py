from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from spatial_memory_evaluation.common.package_io import run_llm_command

from .native_tools import NativeToolExecutor


def run_tool_llm_query(
    *,
    package_dir: Path,
    manifest: Mapping[str, Any],
    query: Mapping[str, Any],
    llm_command: str,
    work_dir: Path,
    max_tool_iterations: int = 3,
    response_kind: str = "predictions",
) -> dict[str, Any]:
    """Run one query through an LLM + method-native tool loop.

    The prompt never embeds benchmark answers, adapter code, or build code. The
    sandbox may mount the original method source so declared native tools can
    run, while the LLM only sees the current query, method/package metadata,
    declared tool schemas, sandbox file summary, and prior observations.

    ``response_kind`` selects the final-answer contract:
    - ``"predictions"`` (Track 1/2): the LLM returns ``final.predictions``.
    - ``"answer"`` (Track 3 OpenEQA QA): the LLM returns ``final.answer`` plus
      optional ``final.evidence``.
    """

    executor = NativeToolExecutor(package_dir, manifest)
    tool_specs = executor.tool_specs()
    if not tool_specs:
        return {
            "status": "invalid",
            "reason_code": "no_method_native_llm_tools",
            "message": (
                "This package does not expose method-native LLM retrieval tools. "
                "Do not force agentic/tool-LLM eval for this method."
            ),
            "predictions": [],
            "trace": [],
        }

    sandbox_context = _prepare_tool_llm_sandbox(
        package_dir=package_dir,
        manifest=manifest,
        tool_specs=tool_specs,
        work_dir=work_dir,
    )
    query_id = _safe_filename(str(query.get("query_id") or "query"))
    query_dir = work_dir / query_id
    query_dir.mkdir(parents=True, exist_ok=True)

    observations: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    last_value: Any = None

    for step_index in range(max(0, max_tool_iterations) + 1):
        prompt_path = query_dir / f"step_{step_index:02d}_prompt.md"
        output_path = query_dir / f"step_{step_index:02d}_llm_output.json"
        prompt_path.write_text(
            _render_prompt(
                manifest=manifest,
                query=query,
                tool_specs=tool_specs,
                observations=observations,
                sandbox_context=sandbox_context,
                max_tool_iterations=max_tool_iterations,
                response_kind=response_kind,
            ),
            encoding="utf-8",
        )
        run_llm_command(
            llm_command=llm_command,
            prompt_path=prompt_path,
            output_path=output_path,
        )
        raw_output = output_path.read_text(encoding="utf-8")
        value = _load_json_value(raw_output, output_path)
        last_value = value
        trace.append(
            {
                "step": step_index,
                "prompt_path": str(prompt_path),
                "output_path": str(output_path),
                "llm_output": value,
            }
        )

        final = _extract_final(value, response_kind=response_kind)
        if final is not None:
            if response_kind == "answer":
                evidence = final.get("evidence")
                return {
                    "status": "ok",
                    "answer": str(final.get("answer") or ""),
                    "evidence": evidence if isinstance(evidence, list) else [],
                    "latency_seconds": time.perf_counter() - started,
                    "trace": trace,
                }
            predictions = final.get("predictions")
            if not isinstance(predictions, list):
                predictions = []
            return {
                "status": "ok",
                "predictions": [_normalize_prediction(pred) for pred in predictions],
                "latency_seconds": time.perf_counter() - started,
                "trace": trace,
            }

        tool_call = _extract_tool_call(value)
        if tool_call is None:
            return {
                "status": "error",
                "message": f"LLM output must contain either tool_call or final.{response_kind}",
                "predictions": [],
                "answer": "",
                "latency_seconds": time.perf_counter() - started,
                "trace": trace,
            }
        if step_index >= max_tool_iterations:
            return {
                "status": "error",
                "message": "LLM requested another tool after max_tool_iterations",
                "predictions": [],
                "answer": "",
                "latency_seconds": time.perf_counter() - started,
                "trace": trace,
            }

        tool_name = str(tool_call.get("name") or "")
        arguments = tool_call.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        observation = executor.execute(tool_name, arguments)
        observations.append(
            {
                "tool_call": {"name": tool_name, "arguments": dict(arguments)},
                "observation": observation,
            }
        )
        trace[-1]["tool_observation"] = observation

    return {
        "status": "error",
        "message": "tool-LLM loop ended without a final answer",
        "predictions": [],
        "answer": "",
        "latency_seconds": time.perf_counter() - started,
        "trace": trace,
        "last_llm_output": last_value,
    }


def _render_prompt(
    *,
    manifest: Mapping[str, Any],
    query: Mapping[str, Any],
    tool_specs: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    sandbox_context: Mapping[str, Any],
    max_tool_iterations: int,
    response_kind: str = "predictions",
) -> str:
    method_meta = manifest.get("method") if isinstance(manifest.get("method"), Mapping) else {}
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), Mapping) else {}
    package_summary = {
        "method": method_meta,
        "dataset": dataset,
        "explicit_memory": manifest.get("explicit_memory"),
        "memory_format": manifest.get("memory_format"),
    }
    if response_kind == "answer":
        query_payload = {
            "query_id": query.get("query_id"),
            "question": query.get("question") or query.get("query"),
            "episode_id": query.get("episode_id"),
        }
        task_line = "You are answering one open-ended spatial question about a scene."
        final_rules = (
            "- Retrieve relevant caption/memory context, then answer the question.\n"
            "- The answer must be a short, direct phrase (a few words), like OpenEQA.\n"
            "- Cite the memory you used as evidence."
        )
        final_format = """{
  "final": {
    "answer": "a short direct answer",
    "evidence": []
  }
}"""
    else:
        query_payload = {
            "query_id": query.get("query_id"),
            "query": query.get("query") or query.get("utterance"),
            "target_label": query.get("target_label") or query.get("canonical_label"),
            "canonical_label": query.get("canonical_label"),
            "utterance": query.get("utterance"),
            "top_k": query.get("top_k"),
        }
        task_line = "You are answering one spatial-memory retrieval query."
        final_rules = (
            "- Prefer the provided target_label/canonical_label when choosing retrieval terms.\n"
            "- Return up to top_k predictions.\n"
            "- A prediction should include label/raw_label, object_id if available,\n"
            "  position_3d or bbox_3d if available, score, and evidence."
        )
        final_format = """{
  "final": {
    "predictions": [
      {
        "object_id": "string-or-null",
        "label": "canonical-or-raw-label",
        "position_3d": [0.0, 0.0, 0.0],
        "bbox_3d": null,
        "score": 0.0,
        "evidence": []
      }
    ]
  }
}"""
    return f"""{task_line}

This is not a coding-agent task. You cannot run shell commands, inspect source
code to create new adapters, or access benchmark ground truth. You may only
call the method-native retrieval tools declared below. The evaluator will execute
the tool call over the sandbox-local raw/native memory and method source, then
return the observation in the next turn.

Package summary:
{json.dumps(package_summary, indent=2, sort_keys=True)}

Current query:
{json.dumps(query_payload, indent=2, sort_keys=True)}

Available method-native tools:
{json.dumps(tool_specs, indent=2, sort_keys=True)}

Sandbox files provided for this query/eval:
{json.dumps(sandbox_context, indent=2, sort_keys=True)}

Prior tool observations for this query:
{json.dumps(observations, indent=2, sort_keys=True)}

Rules:
- Answer only this single query_id.
- Use method-native tools to retrieve memory. Do not invent objects.
- The sandbox intentionally contains raw/native memory and original method
  source needed by the tool runtime. It does not contain evaluator fixed-API
  object tables, adapter code, benchmark answers, or memory build code.
- If you are running in an agent runtime that can inspect files, use only the
  listed sandbox files for understanding tool inputs/outputs. Do not create a
  new query interface or adapter.
{final_rules}
- You may make at most {max_tool_iterations} tool calls before final answer.

Return exactly one raw JSON object and no Markdown.

To call a tool:
{{
  "tool_call": {{
    "name": "tool_name",
    "arguments": {{"key": "value"}}
  }}
}}

To finish:
{final_format}
"""


def _prepare_tool_llm_sandbox(
    *,
    package_dir: Path,
    manifest: Mapping[str, Any],
    tool_specs: list[dict[str, Any]],
    work_dir: Path,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_memory_dir = work_dir / "raw_memory"
    method_source_dir = work_dir / "method_source"
    raw_memory_dir.mkdir(exist_ok=True)
    method_source_dir.mkdir(exist_ok=True)

    raw_memory_paths = []
    native_memory = package_dir / "memory" / "native"
    if native_memory.exists():
        raw_memory_paths.append(_symlink_once(native_memory, raw_memory_dir / "native"))

    method_meta = manifest.get("method") if isinstance(manifest.get("method"), Mapping) else {}
    method_name = _safe_filename(str(method_meta.get("name") or "method"))
    method_family = str(method_meta.get("family") or "").lower()
    if method_family in {"caption_memory", "caption_control"}:
        captions_path = package_dir / "memory" / "captions.jsonl"
        if captions_path.exists():
            raw_memory_paths.append(_symlink_once(captions_path, raw_memory_dir / "captions.jsonl"))

    if not raw_memory_paths:
        (raw_memory_dir / "README.md").write_text(
            "No package-local raw/native memory artifact was found for tool-LLM eval.\n",
            encoding="utf-8",
        )

    method_source_paths = []
    repo_path = method_meta.get("repo_path")
    if isinstance(repo_path, str) and repo_path:
        source = Path(repo_path)
        if source.exists():
            method_source_paths.append(_symlink_once(source, method_source_dir / method_name))

    tool_specs_path = work_dir / "tool_specs.json"
    tool_specs_path.write_text(json.dumps(tool_specs, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "raw_memory_dir": str(raw_memory_dir),
        "raw_memory_paths": [str(path) for path in raw_memory_paths],
        "method_source_dir": str(method_source_dir),
        "method_source_paths": [str(path) for path in method_source_paths],
        "tool_specs_path": str(tool_specs_path),
        "not_provided": [
            "fixed_api object_table/label index views unless they are also under raw_memory",
            "evaluation adapter code",
            "benchmark ground truth or answer files",
            "memory build code beyond original method source",
        ],
    }


def _symlink_once(source: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        return destination
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    return destination


def _extract_final(value: Any, *, response_kind: str = "predictions") -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    final = value.get("final")
    if isinstance(final, Mapping):
        return final
    # Allow a bare top-level final object (no "final" wrapper).
    if response_kind == "answer" and "answer" in value:
        return value
    if response_kind != "answer" and "predictions" in value:
        return value
    return None


def _extract_tool_call(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    tool_call = value.get("tool_call")
    return tool_call if isinstance(tool_call, Mapping) else None


def _normalize_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    label = value.get("label") or value.get("raw_label") or value.get("semantic_label")
    return {
        "object_id": value.get("object_id"),
        "label": label,
        "position_3d": value.get("position_3d"),
        "bbox_3d": value.get("bbox_3d"),
        "score": value.get("score"),
        "evidence": value.get("evidence") if isinstance(value.get("evidence"), list) else [],
    }


def _load_json_value(text: str, path: Path) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for candidate in _json_object_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    preview = stripped[:300].replace("\n", "\\n")
    raise ValueError(f"could not parse tool-LLM JSON output from {path}; preview={preview!r}")


def _json_object_candidates(text: str) -> list[str]:
    candidates = []
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def _safe_filename(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)[:120]
