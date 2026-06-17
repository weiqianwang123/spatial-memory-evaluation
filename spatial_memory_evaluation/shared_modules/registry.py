from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_memory_evaluation.common.labels import DEFAULT_DETECTOR_CLASS_LIST_PATH


DEFAULT_SHARED_MODULES_ROOT = Path(
    os.environ.get("SME_SHARED_MODULES_ROOT", "/data/mondo-training-dataset/semantic_mapping/modules")
)


@dataclass(frozen=True)
class ModuleSpec:
    key: str
    kind: str
    name: str
    role: str
    version: str
    status: str
    checkpoint: Path | None = None
    class_list: Path | None = None
    model_name: str | None = None
    pretrained: str | None = None
    notes: str = ""

    def is_available(self) -> bool:
        if self.checkpoint is not None:
            return self.checkpoint.exists()
        if self.class_list is not None:
            return self.class_list.exists()
        return self.status not in {"missing", "unverified_missing"}

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "name": self.name,
            "role": self.role,
            "version": self.version,
            "status": self.status,
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "class_list": str(self.class_list) if self.class_list is not None else None,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "available": self.is_available(),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class MethodModule:
    spec: ModuleSpec
    required: bool

    def to_json(self) -> dict[str, Any]:
        value = self.spec.to_json()
        value["required"] = self.required
        return value


class SharedModuleRegistry:
    """Registry for shared perception modules used by method adapters.

    This registry records canonical paths and model names, but does not import
    heavy detector/segmenter libraries. Method scripts translate these specs
    into each external repo's native config overrides.
    """

    def __init__(self, root: Path = DEFAULT_SHARED_MODULES_ROOT):
        self.root = Path(root)
        self._modules = self._build_modules()

    def get(self, key: str) -> ModuleSpec:
        try:
            return self._modules[key]
        except KeyError as exc:
            raise KeyError(f"unknown shared module key: {key}") from exc

    def method_modules(self, method: str, profile: str = "smoke") -> list[MethodModule]:
        method = method.lower()
        profile = profile.lower()
        refs = _METHOD_PROFILES.get(profile, {}).get(method)
        if refs is None:
            raise KeyError(f"unknown shared module profile for method={method!r} profile={profile!r}")
        return [MethodModule(spec=self.get(key), required=required) for key, required in refs]

    def method_metadata(self, method: str, profile: str = "smoke") -> dict[str, Any]:
        modules = self.method_modules(method, profile)
        return {
            "policy": "shared_modules",
            "profile": profile,
            "method": method,
            "root": str(self.root),
            "vocabulary_mode": "closed",
            "modules": [module.to_json() for module in modules],
        }

    def check_method(self, method: str, profile: str = "smoke") -> list[str]:
        missing: list[str] = []
        for module in self.method_modules(method, profile):
            if module.required and not module.spec.is_available():
                missing.append(module.spec.key)
        return missing

    def method_settings(self, method: str, profile: str = "smoke") -> dict[str, Any]:
        modules = self.method_modules(method, profile)
        by_kind = {module.spec.kind: module.spec for module in modules}
        settings: dict[str, Any] = {
            "metadata": self.method_metadata(method, profile),
        }
        class_list = by_kind.get("class_list")
        if class_list is not None:
            settings["class_names"] = class_list.class_list
        sam = by_kind.get("sam")
        if sam is not None:
            settings["sam_type"] = sam.version
            settings["sam_checkpoint"] = sam.checkpoint
        yolo_world = by_kind.get("yolo_world")
        if yolo_world is not None:
            settings["yolo_checkpoint"] = yolo_world.checkpoint
        fastsam = by_kind.get("fastsam")
        if fastsam is not None:
            settings["fastsam_checkpoint"] = fastsam.checkpoint
        openclip = by_kind.get("openclip")
        if openclip is not None:
            settings["clip_model"] = openclip.model_name
            settings["clip_pretrained"] = openclip.pretrained
        return settings

    def _build_modules(self) -> dict[str, ModuleSpec]:
        return {
            "detector_class_list.canonical": ModuleSpec(
                key="detector_class_list.canonical",
                kind="class_list",
                name="canonical detector-coverable class list",
                role="closed-vocabulary object labels for formal Track 1/2",
                version="detector_coverable",
                status="present",
                class_list=DEFAULT_DETECTOR_CLASS_LIST_PATH,
                notes="Generated from DEFAULT_DETECTOR_COVERABLE_LABELS.",
            ),
            "sam.vit_h": ModuleSpec(
                key="sam.vit_h",
                kind="sam",
                name="Segment Anything",
                role="formal shared mask proposal target",
                version="vit_h",
                status="missing",
                checkpoint=self.root / "sam" / "vit_h" / "sam_vit_h_4b8939.pth",
                notes="Formal target; centralize before formal runs.",
            ),
            "sam.vit_b": ModuleSpec(
                key="sam.vit_b",
                kind="sam",
                name="Segment Anything",
                role="smoke shared mask proposal fallback",
                version="vit_b",
                status="present_local",
                checkpoint=Path("/home/robin_wang/DualMap/sam_b.pt"),
                notes="Smoke fallback until SAM ViT-H is centralized.",
            ),
            "yolo_world.v8s": ModuleSpec(
                key="yolo_world.v8s",
                kind="yolo_world",
                name="YOLO-World",
                role="closed-vocabulary detector backend constrained by canonical class list",
                version="yolov8s-world",
                status="present_local",
                checkpoint=Path("/home/robin_wang/DualMap/yolov8s-world.pt"),
                notes="Local smoke checkpoint; centralize or replace for formal runs.",
            ),
            "fastsam.s": ModuleSpec(
                key="fastsam.s",
                kind="fastsam",
                name="FastSAM",
                role="optional fast segmentation supplement",
                version="FastSAM-s",
                status="unverified",
                checkpoint=Path("/home/robin_wang/DualMap/model/FastSAM-s.pt"),
                notes="Optional only; not required for current Track 1/2 main runs.",
            ),
            "openclip.vit_b_32": ModuleSpec(
                key="openclip.vit_b_32",
                kind="openclip",
                name="OpenCLIP",
                role="smoke shared visual-language feature encoder",
                version="ViT-B-32/laion2b_s34b_b79k",
                status="package_resolved",
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                notes="Resolved through open_clip; no local checkpoint path is recorded.",
            ),
            "openclip.vit_h_14": ModuleSpec(
                key="openclip.vit_h_14",
                kind="openclip",
                name="OpenCLIP",
                role="formal shared visual-language feature encoder target",
                version="ViT-H-14/laion2b_s32b_b79k",
                status="package_resolved",
                model_name="ViT-H-14",
                pretrained="laion2b_s32b_b79k",
                notes="Resolved through open_clip; confirm all methods can run before formal use.",
            ),
        }


_METHOD_PROFILES: dict[str, dict[str, list[tuple[str, bool]]]] = {
    "smoke": {
        "hovsg": [
            ("detector_class_list.canonical", True),
            ("sam.vit_b", True),
            ("openclip.vit_b_32", True),
        ],
        "dualmap": [
            ("detector_class_list.canonical", True),
            ("yolo_world.v8s", True),
            ("sam.vit_b", True),
            ("openclip.vit_b_32", True),
            ("fastsam.s", False),
        ],
    },
    "formal": {
        "hovsg": [
            ("detector_class_list.canonical", True),
            ("sam.vit_h", True),
            ("openclip.vit_h_14", True),
        ],
        "dualmap": [
            ("detector_class_list.canonical", True),
            ("yolo_world.v8s", True),
            ("sam.vit_h", True),
            ("openclip.vit_h_14", True),
            ("fastsam.s", False),
        ],
    },
}


def get_shared_module_registry(root: Path | None = None) -> SharedModuleRegistry:
    return SharedModuleRegistry(root=root or DEFAULT_SHARED_MODULES_ROOT)
