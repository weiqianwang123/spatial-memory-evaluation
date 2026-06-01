from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable


@dataclass(frozen=True)
class RGBDFrame:
    """One RGB-D frame exported from an OpenEQA episode history."""

    index: int
    rgb_path: Path
    depth_path: Optional[Path] = None
    pose_path: Optional[Path] = None


@dataclass(frozen=True)
class RGBDSequence:
    """A per-episode RGB-D sequence passed to the evaluated method."""

    episode_history: str
    root: Path
    frames: Sequence[RGBDFrame]
    intrinsic_color_path: Optional[Path] = None
    intrinsic_depth_path: Optional[Path] = None
    extrinsic_color_path: Optional[Path] = None
    extrinsic_depth_path: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_depth(self) -> None:
        missing = [frame.rgb_path.name for frame in self.frames if frame.depth_path is None]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"{self.episode_history} has RGB-only frames; missing depth for {preview}"
            )

    def require_poses(self) -> None:
        missing = [frame.rgb_path.name for frame in self.frames if frame.pose_path is None]
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(
                f"{self.episode_history} has no camera poses for {preview}"
            )


@dataclass
class ObjectPrediction:
    """Standard object output shape for downstream ScanNet-style evaluators."""

    label: str
    score: float = 1.0
    object_id: Optional[str] = None
    bbox_3d: Optional[List[float]] = None
    bbox_2d: Optional[List[float]] = None
    position_3d: Optional[List[float]] = None
    mask_path: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, {}, [])}


@runtime_checkable
class SpatialMemoryMethod(Protocol):
    """Interface each spatial-memory method adapter must expose."""

    def get_memory_text(self, question: str) -> str:
        """Answer an OpenEQA question from the method's built memory."""

    def get_object(self, query: str) -> Sequence[Union[ObjectPrediction, Mapping[str, Any]]]:
        """Return object predictions for a ScanNet object query."""
