from __future__ import annotations

import math
import runpy
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def main() -> None:
    _install_matplotlib_backend_guard()
    _install_safe_crop_patch()
    script_path = Path("application/semantic_segmentation.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            "HOV-SG semantic_segmentation.py was not found. Run this launcher "
            "with cwd set to the HOV-SG repo root."
        )
    runpy.run_path(str(script_path), run_name="__main__")


def _install_matplotlib_backend_guard() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    original_switch_backend = plt.switch_backend

    def switch_backend(name: str) -> None:
        if str(name).lower() == "tkagg":
            print(
                "[hovsg wrapper] ignoring TkAgg backend request; using Agg for headless run",
                file=sys.stderr,
            )
            return original_switch_backend("Agg")
        return original_switch_backend(name)

    plt.switch_backend = switch_backend


def _install_safe_crop_patch() -> None:
    import hovsg.utils.sam_utils as sam_utils

    sam_utils.crop_all_bounding_boxs = _safe_crop_all_bounding_boxs

    # Import after patching sam_utils so the extractor's module-global symbol is
    # also the guarded function.
    import hovsg.models.sam_clip_feats_extractor as extractor

    extractor.crop_all_bounding_boxs = _safe_crop_all_bounding_boxs


def _safe_crop_all_bounding_boxs(
    image: np.ndarray,
    masks: list[dict[str, Any]],
    block_background: bool = False,
    bbox_margin: int = 0,
) -> list[np.ndarray]:
    images = []
    for mask_index, mask in enumerate(masks):
        crop = _safe_crop(image, mask, block_background=block_background, bbox_margin=bbox_margin)
        if crop is None:
            _log_empty_crop(mask_index, mask, image.shape, block_background)
            crop = _blank_crop_like(image)
        else:
            crop = cv2.resize(crop, (512, 512))
        images.append(crop)
    return images


def _safe_crop(
    image: np.ndarray,
    mask: dict[str, Any],
    *,
    block_background: bool,
    bbox_margin: int,
) -> Optional[np.ndarray]:
    bbox = _clamped_bbox(mask.get("bbox"), image.shape, bbox_margin)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    source = image
    if block_background:
        segmentation = np.asarray(mask.get("segmentation"))
        if segmentation.shape[:2] == image.shape[:2]:
            source = image * np.expand_dims(segmentation.astype(image.dtype), -1)
    crop = source[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return None
    return crop


def _clamped_bbox(
    bbox: Any,
    image_shape: tuple[int, ...],
    margin: int,
) -> Optional[tuple[int, int, int, int]]:
    if bbox is None or len(bbox) != 4:
        return None
    height, width = image_shape[:2]
    x, y, w, h = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    x0 = max(0, int(math.floor(x - margin)))
    y0 = max(0, int(math.floor(y - margin)))
    x1 = min(width, int(math.ceil(x + w + margin)))
    y1 = min(height, int(math.ceil(y + h + margin)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _blank_crop_like(image: np.ndarray) -> np.ndarray:
    channels = image.shape[2] if image.ndim == 3 else 3
    return np.zeros((512, 512, channels), dtype=image.dtype)


def _log_empty_crop(
    mask_index: int,
    mask: dict[str, Any],
    image_shape: tuple[int, ...],
    block_background: bool,
) -> None:
    count = getattr(_log_empty_crop, "count", 0) + 1
    setattr(_log_empty_crop, "count", count)
    if count <= 20 or count % 100 == 0:
        print(
            "[hovsg safe crop] empty crop replaced with blank crop "
            f"count={count} mask_index={mask_index} bbox={mask.get('bbox')} "
            f"image_shape={image_shape} block_background={block_background}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
