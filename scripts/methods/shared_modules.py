from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.shared_modules import get_shared_module_registry


def add_shared_module_args(parser: argparse.ArgumentParser, *, default_profile: str = "smoke") -> None:
    parser.add_argument(
        "--shared-module-profile",
        choices=("smoke", "formal"),
        default=default_profile,
        help="shared module profile used to configure external method overrides",
    )
    parser.add_argument(
        "--allow-shared-module-override",
        action="store_true",
        help="allow CLI values that differ from the shared_modules registry",
    )


def apply_hovsg_shared_modules(args: argparse.Namespace) -> None:
    settings = get_shared_module_registry().method_settings("hovsg", args.shared_module_profile)
    _set_path(args, "class_names", settings["class_names"], "OV prompt/evaluation label list")
    _set_value(args, "clip_model", settings["clip_model"], "OpenCLIP model")
    _set_value(args, "clip_pretrained", settings["clip_pretrained"], "OpenCLIP pretrained tag")
    _set_value(args, "sam_type", settings["sam_type"], "SAM type")
    _set_path(args, "sam_checkpoint", settings["sam_checkpoint"], "SAM checkpoint")
    args.shared_modules = settings["metadata"]


def apply_dualmap_shared_modules(args: argparse.Namespace) -> None:
    settings = get_shared_module_registry().method_settings("dualmap", args.shared_module_profile)
    _set_path(args, "class_names", settings["class_names"], "OV prompt/evaluation label list")
    _set_path(args, "yolo_checkpoint", settings["yolo_checkpoint"], "YOLO checkpoint")
    _set_path(args, "sam_checkpoint", settings["sam_checkpoint"], "SAM checkpoint")
    _set_path(args, "fastsam_checkpoint", settings["fastsam_checkpoint"], "FastSAM checkpoint")
    _set_value(args, "clip_model", settings["clip_model"], "OpenCLIP model")
    _set_value(args, "clip_pretrained", settings["clip_pretrained"], "OpenCLIP pretrained tag")
    args.shared_modules = settings["metadata"]


def apply_daaam_shared_modules(args: argparse.Namespace) -> None:
    settings = get_shared_module_registry().method_settings("daaam", args.shared_module_profile)
    _set_path(args, "class_names", settings["class_names"], "OV prompt/evaluation label list")
    _set_path(args, "sam_checkpoint", settings["sam_checkpoint"], "SAM checkpoint")
    _set_value(args, "sam_type", settings["sam_type"], "SAM type")
    _set_value(args, "clip_model", settings["clip_model"], "OpenCLIP model")
    _set_value(args, "clip_pretrained", settings["clip_pretrained"], "OpenCLIP pretrained tag")
    args.shared_modules = settings["metadata"]


def shared_modules_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return dict(getattr(args, "shared_modules", {}) or {})


def _set_path(args: argparse.Namespace, attr: str, shared_value: Path | None, label: str) -> None:
    if shared_value is None:
        return
    current = getattr(args, attr, None)
    if current is None:
        setattr(args, attr, Path(shared_value))
        return
    if _same_path(Path(current), Path(shared_value)):
        setattr(args, attr, Path(shared_value))
        return
    if getattr(args, "allow_shared_module_override", False):
        return
    raise ValueError(
        f"{label} must come from shared_modules for method adapter runs. "
        f"{attr}={current} shared={shared_value}. "
        "Pass --allow-shared-module-override only for an explicitly documented ablation."
    )


def _set_value(args: argparse.Namespace, attr: str, shared_value: str | None, label: str) -> None:
    if shared_value is None:
        return
    current = getattr(args, attr, None)
    if current is None:
        setattr(args, attr, shared_value)
        return
    if current == shared_value:
        return
    if getattr(args, "allow_shared_module_override", False):
        return
    raise ValueError(
        f"{label} must come from shared_modules for method adapter runs. "
        f"{attr}={current} shared={shared_value}. "
        "Pass --allow-shared-module-override only for an explicitly documented ablation."
    )


def _same_path(left: Path, right: Path) -> bool:
    return left.expanduser().resolve() == right.expanduser().resolve()
