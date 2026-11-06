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


def _max_weight_matching(
    predictions: list[tuple[tuple[float, float], dict]],
    truths: list[tuple[tuple[float, float], dict]],
    iou_threshold: float,
) -> list[dict]:
    """Bipartite min-cost max-flow: maximum cardinality, then maximum total IoU.

    Greedy IoU sorting can leave a matchable event unmatched when two
    predictions compete for overlapping ground-truth spans. Successive
    shortest augmenting paths avoid that.
    """
    edges: list[tuple[int, int, float]] = []
    for pred_index, (pred_bounds, _) in enumerate(predictions):
        for gt_index, (gt_bounds, _) in enumerate(truths):
            score = interval_iou(pred_bounds, gt_bounds)
            if score >= iou_threshold:
                edges.append((pred_index, gt_index, score))
    if not edges:
        return []

    source = 0
    pred_base = 1
    gt_base = 1 + len(predictions)
    sink = gt_base + len(truths)
    graph: list[list[list]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, cost: float) -> list:
        forward = [end, capacity, cost, len(graph[end])]
        backward = [start, 0, -cost, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    for pred_index in range(len(predictions)):
        add_edge(source, pred_base + pred_index, 1, 0.0)
    for gt_index in range(len(truths)):
        add_edge(gt_base + gt_index, sink, 1, 0.0)
    original_edges = []
    for pred_index, gt_index, score in edges:
        forward = add_edge(pred_base + pred_index, gt_base + gt_index, 1, -score)
        original_edges.append((pred_index, gt_index, score, forward))

    for _ in range(min(len(predictions), len(truths))):
        distance = [float("inf")] * len(graph)
        previous_node = [-1] * len(graph)
        previous_edges: list[list | None] = [None] * len(graph)
        distance[source] = 0.0
        queue = [source]
        in_queue = [False] * len(graph)
        in_queue[source] = True
        while queue:
            node = queue.pop(0)
            in_queue[node] = False
            if distance[node] == float("inf"):
                continue
            for edge in graph[node]:
                target, capacity, cost = edge[0], edge[1], edge[2]
                if capacity <= 0:
                    continue
                candidate = distance[node] + cost
                if candidate < distance[target] - 1e-12:
                    distance[target] = candidate
                    previous_node[target] = node
                    previous_edges[target] = edge
                    if not in_queue[target]:
                        queue.append(target)
                        in_queue[target] = True
        if previous_node[sink] == -1:
            break
        node = sink
        while node != source:
            edge = previous_edges[node]
            if edge is None:
                break
            edge[1] -= 1
            graph[edge[0]][edge[3]][1] += 1
            node = previous_node[node]

    matches = []
    for pred_index, gt_index, score, forward in original_edges:
        if forward[1] != 0:
            continue
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
    matches.sort(key=lambda item: item["pred_index"])
    return matches


def match_events(
    predicted: Iterable[dict],
    ground_truth: Iterable[dict],
    iou_threshold: float = 0.3,
) -> dict:
    predictions = [(_bounds(item), item) for item in predicted]
    truths = [(_bounds(item), item) for item in ground_truth]
    matches = _max_weight_matching(predictions, truths, iou_threshold)
    used_pred = {int(match["pred_index"]) for match in matches}
    used_gt = {int(match["gt_index"]) for match in matches}
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
    precision = matched / len(predictions) if predictions else None
    recall = matched / len(truths) if truths else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (None if precision is None or recall is None else 0.0)
    )

    errors = [error for match in result["matches"] for error in (match["start_error"], match["end_error"])]
    boundary_p95 = float(np.percentile(errors, 95)) if errors else None
    boundary_mean = float(np.mean(errors)) if errors else None
    start_errors = [float(match["start_error"]) for match in result["matches"]]
    end_errors = [float(match["end_error"]) for match in result["matches"]]
    max_errors = [
        max(float(match["start_error"]), float(match["end_error"]))
        for match in result["matches"]
    ]
    boundary_start_p95 = float(np.percentile(start_errors, 95)) if start_errors else None
    boundary_end_p95 = float(np.percentile(end_errors, 95)) if end_errors else None
    boundary_max_p95 = float(np.percentile(max_errors, 95)) if max_errors else None

    false_positives = len(result["unmatched_predictions"])
    false_negatives = len(result["unmatched_ground_truth"])
    fp_per_hour = None
    if duration_s and duration_s > 0:
        fp_per_hour = round(false_positives / (duration_s / 3600.0), 3)

    return {
        "event_precision": round(precision, 4) if precision is not None else None,
        "event_recall": round(recall, 4) if recall is not None else None,
        "event_f1": round(f1, 4) if f1 is not None else None,
        "ground_truth_events": len(truths),
        "predicted_events": len(predictions),
        "matched": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "boundary_p95_s": round(boundary_p95, 3) if boundary_p95 is not None else None,
        "boundary_mean_s": round(boundary_mean, 3) if boundary_mean is not None else None,
        "boundary_start_p95_s": (
            round(boundary_start_p95, 3) if boundary_start_p95 is not None else None
        ),
        "boundary_end_p95_s": (
            round(boundary_end_p95, 3) if boundary_end_p95 is not None else None
        ),
        "boundary_max_p95_s": (
            round(boundary_max_p95, 3) if boundary_max_p95 is not None else None
        ),
        "false_positives_per_hour": fp_per_hour,
        "iou_threshold": iou_threshold,
        "matches": result["matches"],
        "unmatched_predictions": [
            predictions[index] for index in result["unmatched_predictions"]
        ],
        "unmatched_ground_truth": [
            {"start_s": _bounds(truths[index])[0], "end_s": _bounds(truths[index])[1]}
            for index in result["unmatched_ground_truth"]
        ],
    }
