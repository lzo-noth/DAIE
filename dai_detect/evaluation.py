"""Event-level evaluation shared by training, blind tests and unit tests."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _bounds(item: dict) -> tuple[float, float]:
    start = item.get("start_s", item.get("start"))
    end = item.get("end_s", item.get("end"))
    if start is None or end is None:
        raise ValueError(f"区间缺少起止时间: {item}")
    return float(start), float(end)


def interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return overlap / union if union > 0 else 0.0


def match_events(
    predicted: Iterable[dict],
    ground_truth: Iterable[dict],
    iou_threshold: float = 0.3,
) -> dict:
    predictions = [(_bounds(item), item) for item in predicted]
    truths = [(_bounds(item), item) for item in ground_truth]
    pairs: list[tuple[float, int, int]] = []
    for pred_index, (pred_bounds, _) in enumerate(predictions):
        for gt_index, (gt_bounds, _) in enumerate(truths):
            score = interval_iou(pred_bounds, gt_bounds)
            if score >= iou_threshold:
                pairs.append((score, pred_index, gt_index))
    pairs.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict] = []
    for score, pred_index, gt_index in pairs:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        pred_bounds, pred_item = predictions[pred_index]
        gt_bounds, _ = truths[gt_index]
        matches.append(
            {
                "pred_index": pred_index,
                "gt_index": gt_index,
                "iou": round(score, 4),
                "start_error": round(abs(pred_bounds[0] - gt_bounds[0]), 3),
                "end_error": round(abs(pred_bounds[1] - gt_bounds[1]), 3),
                "predicted": pred_item,
                "ground_truth": gt_bounds,
            }
        )
    return {
        "matches": matches,
        "unmatched_predictions": [index for index in range(len(predictions)) if index not in used_pred],
        "unmatched_ground_truth": [index for index in range(len(truths)) if index not in used_gt],
        "predictions": predictions,
        "ground_truth": truths,
    }


def evaluate_events(
    predicted: Iterable[dict],
    ground_truth: Iterable[dict],
    duration_s: float | None = None,
    iou_threshold: float = 0.3,
) -> dict:
    predictions = list(predicted)
    truths = list(ground_truth)
    result = match_events(predictions, truths, iou_threshold=iou_threshold)
    matched = len(result["matches"])
    precision = matched / len(predictions) if predictions else (1.0 if not truths else 0.0)
    recall = matched / len(truths) if truths else (1.0 if not predictions else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    errors = [error for match in result["matches"] for error in (match["start_error"], match["end_error"])]
    boundary_p95 = float(np.percentile(errors, 95)) if errors else None
    boundary_mean = float(np.mean(errors)) if errors else None

    false_positives = len(result["unmatched_predictions"])
    false_negatives = len(result["unmatched_ground_truth"])
    fp_per_hour = None
    if duration_s and duration_s > 0:
        fp_per_hour = round(false_positives / (duration_s / 3600.0), 3)

    return {
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(f1, 4),
        "matched": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "boundary_p95_s": round(boundary_p95, 3) if boundary_p95 is not None else None,
        "boundary_mean_s": round(boundary_mean, 3) if boundary_mean is not None else None,
        "false_positives_per_hour": fp_per_hour,
        "iou_threshold": iou_threshold,
        "matches": result["matches"],
        "unmatched_predictions": [
            predictions[index] for index in result["unmatched_predictions"]
        ],
        "unmatched_ground_truth": [
            {"start_s": truths[index][0][0], "end_s": truths[index][0][1]}
            for index in result["unmatched_ground_truth"]
        ],
    }
