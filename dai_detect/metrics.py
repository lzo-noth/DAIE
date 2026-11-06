"""Ranking metrics with correct tie handling.

The previous in-script AUC mixed original sample indices with sorted rank
positions, so equal prediction scores produced wrong ranks. ROC-AUC is the
Mann–Whitney U statistic, which requires average ranks inside each tie group.
"""

from __future__ import annotations

import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores):
        raise ValueError("labels 与 scores 长度不一致")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    # Positions are 1-based; a tie group covering positions start+1 .. start+count
    # gets the average rank start + (count + 1) / 2.
    average_ranks = starts + (counts + 1) / 2.0
    ranks_sorted = np.repeat(average_ranks, counts)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = ranks_sorted
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    """Pick the F1-maximising threshold; ties prefer the value closest to 0.5."""
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    best = (0.5, 0.0, 0.0, 0.0)
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[3] + 1e-9 or (
            abs(f1 - best[3]) <= 1e-9 and abs(threshold - 0.5) < abs(best[0] - 0.5)
        ):
            best = (float(threshold), precision, recall, f1)
    return best[0], {
        "precision": round(best[1], 4),
        "recall": round(best[2], 4),
        "f1": round(best[3], 4),
    }
