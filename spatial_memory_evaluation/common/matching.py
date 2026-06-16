from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .labels import normalize_label


@dataclass(frozen=True)
class Match:
    gt_id: str
    prediction_index: int
    distance_m: float


def match_objects(
    gt_objects: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    aliases: Mapping[str, str] | None = None,
) -> tuple[list[Match], list[int], list[str], int]:
    candidates: list[tuple[float, int, str]] = []
    duplicate_candidates = 0

    for pred_index, pred in enumerate(predictions):
        pred_center = prediction_center(pred)
        if pred_center is None:
            continue
        pred_label = normalize_label(pred.get("label"), aliases)
        for gt in gt_objects:
            gt_label = normalize_label(gt.get("canonical_label") or gt.get("label"), aliases)
            if pred_label != gt_label:
                continue
            gt_center = _as_vector(gt.get("center_3d"))
            if gt_center is None:
                continue
            distance = euclidean_distance(pred_center, gt_center)
            if distance <= match_threshold(gt):
                candidates.append((distance, pred_index, str(gt["gt_id"])))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_gt: set[str] = set()
    matches: list[Match] = []
    seen_candidate_gt: set[str] = set()

    for distance, pred_index, gt_id in candidates:
        if gt_id in seen_candidate_gt:
            duplicate_candidates += 1
        else:
            seen_candidate_gt.add(gt_id)
        if pred_index in used_predictions or gt_id in used_gt:
            continue
        used_predictions.add(pred_index)
        used_gt.add(gt_id)
        matches.append(Match(gt_id=gt_id, prediction_index=pred_index, distance_m=distance))

    unmatched_predictions = [i for i in range(len(predictions)) if i not in used_predictions]
    unmatched_gt = [str(gt["gt_id"]) for gt in gt_objects if str(gt["gt_id"]) not in used_gt]
    return matches, unmatched_predictions, unmatched_gt, duplicate_candidates


def inventory_metrics(
    gt_objects: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    valid_predictions = [pred for pred in predictions if prediction_center(pred) is not None]
    matches, unmatched_predictions, unmatched_gt, duplicate_count = match_objects(gt_objects, valid_predictions, aliases)
    tp = len(matches)
    fp = len(unmatched_predictions)
    fn = len(unmatched_gt)
    distances = [match.distance_m for match in matches]
    return {
        "gt_count": len(gt_objects),
        "prediction_count": len(predictions),
        "valid_prediction_count": len(valid_predictions),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "duplicate_count": duplicate_count,
        "false_memory_ratio": safe_div(fp, len(valid_predictions)),
        "mean_center_error_m": mean(distances),
        "median_center_error_m": median(distances),
        "matches": [match.__dict__ for match in matches],
        "unmatched_gt_ids": unmatched_gt,
    }


def prediction_center(prediction: Mapping[str, Any]) -> list[float] | None:
    center = _as_vector(prediction.get("position_3d"))
    if center is not None:
        return center
    bbox = _as_vector(prediction.get("bbox_3d"), expected_len=6)
    if bbox is None:
        return None
    return [
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0,
    ]


def match_threshold(gt_object: Mapping[str, Any]) -> float:
    diagonal = gt_object.get("bbox_diag_m")
    try:
        diagonal_value = float(diagonal)
    except (TypeError, ValueError):
        diagonal_value = 0.0
    return min(2.0, max(0.5, 0.5 * diagonal_value))


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(av, bv)))


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _as_vector(value: Any, expected_len: int = 3) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    return vector
