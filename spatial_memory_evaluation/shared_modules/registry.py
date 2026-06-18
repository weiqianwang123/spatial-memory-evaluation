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
            "vocabulary_mode": "open_vocabulary",
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
        dam = by_kind.get("dam")
        if dam is not None:
            settings["dam_model_path"] = dam.checkpoint or dam.model_name
        sentence_embedding = by_kind.get("sentence_embedding")
        if sentence_embedding is not None:
            settings["sentence_embedding_model"] = sentence_embedding.checkpoint or sentence_embedding.model_name
        return settings

    def _build_modules(self) -> dict[str, ModuleSpec]:
        return {
            "detector_class_list.canonical": ModuleSpec(
                key="detector_class_list.canonical",
                kind="class_list",
                name="shared OV detector prompt/evaluation label list",
                role="shared prompt list and detector-coverable evaluation label set for Track 1/2",
                version="detector_coverable",
                status="present",
                class_list=DEFAULT_DETECTOR_CLASS_LIST_PATH,
                notes=(
                    "Generated from DEFAULT_DETECTOR_COVERABLE_LABELS. It is used as the "
                    "shared OV detector prompt/evaluation list, not as a closed-detector policy."
                ),
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
                role="smoke open-vocabulary detector fallback",
                version="yolov8s-world",
                status="present_local",
                checkpoint=Path("/home/robin_wang/DualMap/yolov8s-world.pt"),
                notes=(
                    "Only YOLO-World checkpoint currently found locally. Use for smoke runs until "
                    "the stronger shared formal checkpoint is centralized."
                ),
            ),
            "yolo_world.v8l": ModuleSpec(
                key="yolo_world.v8l",
                kind="yolo_world",
                name="YOLO-World",
                role="formal strongest shared open-vocabulary detector target",
                version="yolov8l-world",
                status="missing_local",
                checkpoint=self.root / "yolo" / "yolo_world" / "yolov8l-world.pt",
                notes=(
                    "Formal target: DualMap's native config and ConceptGraphs streamlined path both "
                    "reference yolov8l-world.pt. It was not found locally during the 2026-06-17 audit; "
                    "download or symlink it before formal runs."
                ),
            ),
            "fastsam.s_pt": ModuleSpec(
                key="fastsam.s_pt",
                kind="fastsam",
                name="FastSAM",
                role="smoke optional fast segmentation checkpoint",
                version="FastSAM-s",
                status="nas_snapshot",
                checkpoint=self.root / "fastsam" / "s" / "FastSAM-s.pt",
                notes=(
                    "Optional only; not required for current Track 1/2 main runs. "
                    "Keep the .pt under shared_modules before exporting TensorRT engines."
                ),
            ),
            "fastsam.x_trt_640x480": ModuleSpec(
                key="fastsam.x_trt_640x480",
                kind="fastsam",
                name="FastSAM TensorRT",
                role="DAAAM-native realtime segmentation engine",
                version="FastSAM-x-640x480.engine",
                status="nas_snapshot",
                checkpoint=self.root / "fastsam" / "x" / "FastSAM-x-640x480.engine",
                notes=(
                    "DAAAM native-fast route target. Export from FastSAM-x.pt on the target "
                    "GPU/TensorRT stack and store under shared_modules; the adapter passes the "
                    "absolute engine path to DAAAM without editing the DAAAM repo."
                ),
            ),
            "fastsam.s_trt_640x480": ModuleSpec(
                key="fastsam.s_trt_640x480",
                kind="fastsam",
                name="FastSAM TensorRT",
                role="DAAAM-native realtime segmentation smoke fallback",
                version="FastSAM-s-640x480.engine",
                status="nas_snapshot",
                checkpoint=self.root / "fastsam" / "s" / "FastSAM-s-640x480.engine",
                notes=(
                    "Smaller optional TensorRT engine for DAAAM smoke/debug runs. Record it as "
                    "a native-fast ablation if used instead of the formal FastSAM-x engine."
                ),
            ),
            "openclip.vit_b_32": ModuleSpec(
                key="openclip.vit_b_32",
                kind="openclip",
                name="OpenCLIP",
                role="smoke shared visual-language feature encoder",
                version="ViT-B-32/laion2b_s34b_b79k",
                status="nas_snapshot",
                checkpoint=(
                    self.root
                    / "openclip"
                    / "ViT-B-32"
                    / "laion2b_s34b_b79k"
                    / "hf_cache"
                    / "models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K"
                    / "snapshots"
                    / "1a25a446712ba5ee05982a381eed697ef9b435cf"
                    / "open_clip_model.safetensors"
                ),
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                notes=(
                    "Resolved through open_clip using the NAS HuggingFace cache under "
                    "shared_modules/openclip/ViT-B-32/laion2b_s34b_b79k/hf_cache."
                ),
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
            "daaam.dam_3b": ModuleSpec(
                key="daaam.dam_3b",
                kind="dam",
                name="Describe Anything Model",
                role="DAAAM-native object grounding / free-text description model",
                version="nvidia/DAM-3B",
                status="nas_snapshot",
                checkpoint=self.root / "dam" / "nvidia_DAM-3B",
                model_name="nvidia/DAM-3B",
                notes="Loaded by DAAAM grounding workers through the DAM package; use NAS snapshot for reproducible runs.",
            ),
            "daaam.sentence_t5_large": ModuleSpec(
                key="daaam.sentence_t5_large",
                kind="sentence_embedding",
                name="SentenceTransformers",
                role="DAAAM-native text embedding for scene-understanding tools",
                version="sentence-transformers/sentence-t5-large",
                status="nas_snapshot",
                checkpoint=self.root / "embeddings" / "sentence-transformers_sentence-t5-large",
                model_name="sentence-transformers/sentence-t5-large",
                notes="Used for DAAAM deterministic scene-graph semantic tools; use NAS snapshot for reproducible runs.",
            ),
            "daaam.hydra_spark_dsg": ModuleSpec(
                key="daaam.hydra_spark_dsg",
                kind="scene_graph",
                name="Hydra / Spark-DSG",
                role="DAAAM-native dynamic scene graph construction and storage",
                version="method_native",
                status="repo_present",
                notes="Uses the DAAAM/Hydra Python bindings available in the DAAAM runtime env.",
            ),
            "daaam.botsort_reid": ModuleSpec(
                key="daaam.botsort_reid",
                kind="tracker",
                name="BotSort / ReID",
                role="DAAAM-native object track association",
                version="method_native",
                status="package_resolved",
                notes="DAAAM uses boxmot/BotSort plus ReID weights; exact runtime path is recorded per build.",
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
            ("fastsam.s_pt", False),
        ],
        "conceptgraphs": [
            ("detector_class_list.canonical", True),
            ("yolo_world.v8s", True),
            ("sam.vit_b", True),
            ("openclip.vit_b_32", True),
        ],
        "daaam": [
            ("detector_class_list.canonical", True),
            ("sam.vit_b", True),
            ("fastsam.x_trt_640x480", False),
            ("openclip.vit_b_32", True),
            ("daaam.dam_3b", True),
            ("daaam.sentence_t5_large", True),
            ("daaam.hydra_spark_dsg", True),
            ("daaam.botsort_reid", True),
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
            ("yolo_world.v8l", True),
            ("sam.vit_h", True),
            ("openclip.vit_h_14", True),
            ("fastsam.s_pt", False),
        ],
        "conceptgraphs": [
            ("detector_class_list.canonical", True),
            ("yolo_world.v8l", True),
            ("sam.vit_h", True),
            ("openclip.vit_h_14", True),
        ],
        "daaam": [
            ("detector_class_list.canonical", True),
            ("sam.vit_h", True),
            ("fastsam.x_trt_640x480", False),
            ("openclip.vit_h_14", True),
            ("daaam.dam_3b", True),
            ("daaam.sentence_t5_large", True),
            ("daaam.hydra_spark_dsg", True),
            ("daaam.botsort_reid", True),
        ],
    },
}


def get_shared_module_registry(root: Path | None = None) -> SharedModuleRegistry:
    return SharedModuleRegistry(root=root or DEFAULT_SHARED_MODULES_ROOT)
