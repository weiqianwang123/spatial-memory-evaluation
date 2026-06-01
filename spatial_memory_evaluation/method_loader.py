from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .interfaces import RGBDSequence, SpatialMemoryMethod


def parse_method_kwargs(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        with path.open("r") as f:
            value = json.load(f)
    else:
        value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--method-kwargs must be a JSON object or a path to one")
    return value


def load_method(
    method_spec: str,
    sequence: RGBDSequence,
    method_kwargs: Optional[Mapping[str, Any]] = None,
) -> SpatialMemoryMethod:
    """Load a method adapter from 'module:attribute' and attach the sequence."""

    if ":" not in method_spec:
        raise ValueError("method spec must use 'module:attribute', e.g. my_pkg.adapter:create")

    module_name, attr_name = method_spec.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    kwargs = dict(method_kwargs or {})

    method = _instantiate(target, sequence=sequence, kwargs=kwargs)
    _attach_sequence_if_supported(method, sequence)
    _validate_method(method, method_spec)
    return method


def _instantiate(target: Any, sequence: RGBDSequence, kwargs: Dict[str, Any]) -> Any:
    if not callable(target):
        return target

    signature = inspect.signature(target)
    parameters = signature.parameters
    if "sequence" in parameters:
        return target(sequence=sequence, **kwargs)
    if "rgbd_sequence" in parameters:
        return target(rgbd_sequence=sequence, **kwargs)

    try:
        return target(sequence, **kwargs)
    except TypeError:
        return target(**kwargs)


def _attach_sequence_if_supported(method: Any, sequence: RGBDSequence) -> None:
    for name in ("build_memory", "load_rgbd_sequence", "set_rgbd_sequence"):
        hook = getattr(method, name, None)
        if callable(hook):
            hook(sequence)
            return


def _validate_method(method: Any, method_spec: str) -> None:
    missing = [
        name
        for name in ("get_memory_text", "get_object")
        if not callable(getattr(method, name, None))
    ]
    if missing:
        raise TypeError(f"{method_spec} missing required method(s): {', '.join(missing)}")
