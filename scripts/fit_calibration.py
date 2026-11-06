#!/usr/bin/env python3
"""Fit a candidate ad scorer from persisted LLM labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.calibration import FEATURES, label_features


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _load_rows(paths: list[Path]) -> tuple[list[list[float]], np.ndarray, np.ndarray, list[str], list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    baselines: list[float] = []
    groups: list[str] = []
    episodes: list[str] = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        group = (report.get("feed") or {}).get("title") or path.parent.name
        entries = report.get("llm_labels") or []
        for entry in entries:
            features = entry.get("features") or {}
            rows.append(label_features(features, str(entry.get("tier") or "")))
            labels.append(1 if entry.get("is_ad") else 0)
            baselines.append(
                float(
                    entry.get("heuristic_score")
                    if entry.get("heuristic_score") is not None
                    else entry.get("score_before_llm") or 0.0
                )
            )
            groups.append(str(group))
            episodes.append(f"{group}|{entry.get('episode') or ''}")
    return rows, np.asarray(labels), np.asarray(baselines), groups, episodes


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (x - mean) / scale, mean, scale


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 0.05, lr: float = 0.2, steps: int = 3000) -> tuple[np.ndarray, float]:
    xb = np.c_[np.ones(len(x)), x]
    weights = np.zeros(xb.shape[1])
    positives = max(int(y.sum()), 1)
    negatives = len(y) - positives
    sample_weight = np.where(y == 1, negatives / positives, 1.0)
    for _ in range(steps):
        probability = _sigmoid(xb @ weights)
        gradient = xb.T @ ((probability - y) * sample_weight) / len(y) + l2 * weights
        weights -= lr * gradient
    return weights[1:], float(weights[0])


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    positive = scores[y == 1]
    negative = scores[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return 0.0
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))


def _choose_threshold(scores: np.ndarray, y: np.ndarray, min_recall: float) -> tuple[float, dict]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        predicted = scores >= threshold
        true_positive = int((predicted & (y == 1)).sum())
        recall = true_positive / max(1, int((y == 1).sum()))
        if recall < min_recall:
            continue
        precision = true_positive / max(1, int(predicted.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        candidate = (threshold, precision, recall, f1)
        if best is None or precision > best[1]:
            best = candidate
    if best is None:
        threshold = 0.05
        predicted = scores >= threshold
        true_positive = int((predicted & (y == 1)).sum())
        recall = true_positive / max(1, int((y == 1).sum()))
        precision = true_positive / max(1, int(predicted.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        best = (threshold, precision, recall, f1)
    threshold, precision, recall, f1 = best
    return float(threshold), {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = []
    for path in args.reports:
        paths.extend(path.rglob("report.json") if path.is_dir() else [path])
    rows, labels, baselines, groups, episodes = _load_rows(sorted(set(paths)))
    if len(rows) == 0 or labels.sum() == 0 or (len(labels) - labels.sum()) == 0:
        print(f"not enough labels: samples={len(rows)} ads={int(labels.sum())}")
        return 1
    x = np.asarray(rows, dtype=np.float64)
    print(f"samples={len(rows)} ads={int(labels.sum())} non_ads={len(rows)-int(labels.sum())} groups={len(set(groups))}")

    unique_groups = sorted(set(groups))
    learned_cv = np.zeros(len(labels))
    if len(unique_groups) >= 2:
        for group in unique_groups:
            test = np.asarray([g == group for g in groups])
            train = ~test
            if labels[train].sum() == 0 or labels[test].sum() == 0:
                learned_cv[test] = 0.0
                continue
            standardized, mean, scale = _standardize(x[train])
            weights, intercept = _fit_logistic(standardized, labels[train])
            test_x = (x[test] - mean) / scale
            learned_cv[test] = _sigmoid(test_x @ weights + intercept)
    else:
        learned_cv = baselines.copy()

    learned_threshold, learned_metrics = _choose_threshold(learned_cv, labels, args.min_recall)
    baseline_threshold, baseline_metrics = _choose_threshold(baselines, labels, args.min_recall)
    learned_auc = _auc(learned_cv, labels)
    baseline_auc = _auc(baselines, labels)
    print(f"learned  auc={learned_auc:.3f} threshold={learned_threshold:.2f} {learned_metrics}")
    print(f"baseline auc={baseline_auc:.3f} threshold={baseline_threshold:.2f} {baseline_metrics}")

    from collections import defaultdict

    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for score, label, episode in zip(learned_cv, labels, episodes):
        buckets[episode].append((float(score), int(label)))
    total_ads = int(labels.sum())
    recall_at: dict[int, float] = {}
    top_k = 0
    for k in (5, 8, 10, 12, 15, 20, 24, 30, 40):
        hits = 0
        for entries in buckets.values():
            entries.sort(key=lambda item: item[0], reverse=True)
            hits += sum(label for _, label in entries[:k])
        recall_at[k] = hits / max(1, total_ads)
        if top_k == 0 and recall_at[k] >= args.min_recall and k >= 8:
            top_k = k
    print("recall@k:", {k: round(v, 2) for k, v in recall_at.items()})
    if top_k == 0:
        top_k = 40

    improved = learned_auc > baseline_auc + 0.01 and recall_at.get(top_k, 0.0) >= args.min_recall
    standardized, mean, scale = _standardize(x)
    weights, intercept = _fit_logistic(standardized, labels)
    payload = {
        "enabled": bool(improved),
        "mode": "rank",
        "top_k": int(top_k),
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "intercept": intercept,
        "threshold": learned_threshold,
        "min_recall": args.min_recall,
        "samples": len(rows),
        "ads": int(labels.sum()),
        "groups": unique_groups,
        "metrics": {
            "learned": {**learned_metrics, "auc": round(learned_auc, 3), "recall_at_k": round(recall_at[top_k], 3)},
            "baseline": {**baseline_metrics, "auc": round(baseline_auc, 3)},
            "recall_at": {str(k): round(v, 3) for k, v in recall_at.items()},
        },
    }
    print("enabled:", payload["enabled"], "top_k:", top_k)
    if not args.dry_run:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
