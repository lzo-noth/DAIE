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
from dai_detect.keywords import load_terms, match_terms


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _row_key(feed: str, episode: str, start: float, end: float, tier: str) -> str:
    return f"{feed}|{episode}|{start:.1f}|{end:.1f}|{tier}"


def _load_rows(
    paths: list[Path],
) -> tuple[
    list[list[float]],
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    list[float],
    list[str],
    list[str],
    list[str],
]:
    entries: dict[str, tuple[list[float], int, float, str, str, float, str, str, str]] = {}
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                feed = str(row.get("feed") or "")
                episode = str(row.get("episode") or "")
                tier = str(row.get("tier") or "")
                start = float(row.get("start_s") or 0.0)
                end = float(row.get("end_s") or 0.0)
                duration = float(row.get("episode_duration_s") or 0.0)
                key = _row_key(feed, episode, start, end, tier)
                entries[key] = (
                    label_features(row.get("features") or {}, tier),
                    1 if row.get("is_ad") else 0,
                    float(row.get("heuristic_score") if row.get("heuristic_score") is not None else row.get("score_before_llm") or 0.0),
                    feed,
                    f"{feed}|{episode}",
                    duration,
                    str(row.get("asr_text") or ""),
                    str(row.get("run_id") or ""),
                    str(row.get("collected_at") or ""),
                )
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        group = (report.get("feed") or {}).get("title") or path.parent.name
        episodes = {
            episode.get("title"): float(episode.get("duration_s") or 0.0)
            for episode in report.get("episodes") or []
        }
        for entry in report.get("llm_labels") or []:
            features = entry.get("features") or {}
            episode = str(entry.get("episode") or "")
            tier = str(entry.get("tier") or "")
            start = float(entry.get("start_s") or 0.0)
            end = float(entry.get("end_s") or 0.0)
            duration = episodes.get(episode, 0.0)
            key = _row_key(group, episode, start, end, tier)
            entries[key] = (
                label_features(features, tier),
                1 if entry.get("is_ad") else 0,
                float(
                    entry.get("heuristic_score")
                    if entry.get("heuristic_score") is not None
                    else entry.get("score_before_llm") or 0.0
                ),
                group,
                f"{group}|{episode}",
                duration,
                str(entry.get("asr_text") or features.get("asr_text") or ""),
                "",
                "",
            )
    rows = [item[0] for item in entries.values()]
    labels = np.asarray([item[1] for item in entries.values()], dtype=np.float64)
    baselines = np.asarray([item[2] for item in entries.values()], dtype=np.float64)
    groups = [item[3] for item in entries.values()]
    episodes = [item[4] for item in entries.values()]
    durations = [item[5] for item in entries.values()]
    texts = [item[6] for item in entries.values()]
    run_ids = [item[7] for item in entries.values()]
    timestamps = [item[8] for item in entries.values()]
    return rows, labels, baselines, groups, episodes, durations, texts, run_ids, timestamps


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


def _choose_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    min_recall: float,
    target_precision: float | None = None,
) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.02, 0.98, 97):
        predicted = scores >= threshold
        true_positive = int((predicted & (y == 1)).sum())
        recall = true_positive / max(1, int((y == 1).sum()))
        if recall < min_recall:
            continue
        precision = true_positive / max(1, int(predicted.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        candidates.append((float(threshold), precision, recall, f1))

    chosen = None
    if target_precision is not None:
        feasible = [item for item in candidates if item[1] >= target_precision]
        if feasible:
            chosen = max(feasible, key=lambda item: (item[2], item[1], item[3]))
    if chosen is None and candidates:
        chosen = (
            max(candidates, key=lambda item: item[1])
            if target_precision is None
            else max(candidates, key=lambda item: item[3])
        )
    if chosen is None:
        threshold = 0.05
        predicted = scores >= threshold
        true_positive = int((predicted & (y == 1)).sum())
        recall = true_positive / max(1, int((y == 1).sum()))
        precision = true_positive / max(1, int(predicted.sum()))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        chosen = (threshold, precision, recall, f1)
    threshold, precision, recall, f1 = chosen
    return float(threshold), {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    episodes: list[str],
    ks: tuple[int, ...],
) -> tuple[dict[int, float], dict[int, float], dict[int, int], dict[int, int]]:
    from collections import defaultdict

    buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for score, label, episode in zip(scores, labels, episodes):
        buckets[episode].append((float(score), int(label)))
    total_ads = int(labels.sum())
    recall_at: dict[int, float] = {}
    precision_at: dict[int, float] = {}
    false_positive_at: dict[int, int] = {}
    selected_at: dict[int, int] = {}
    for k in ks:
        hits = 0
        selected = 0
        false_positives = 0
        for entries in buckets.values():
            top = sorted(entries, key=lambda item: item[0], reverse=True)[:k]
            selected += len(top)
            hits += sum(label for _, label in top)
            false_positives += sum(1 for _, label in top if label == 0)
        recall_at[k] = hits / max(1, total_ads)
        precision_at[k] = hits / max(1, selected)
        false_positive_at[k] = false_positives
        selected_at[k] = selected
    return recall_at, precision_at, false_positive_at, selected_at


def _episode_hours(episodes: list[str], durations: list[float]) -> float:
    unique: dict[str, float] = {}
    for episode, duration in zip(episodes, durations):
        if duration > 0:
            unique[episode] = max(unique.get(episode, 0.0), duration)
    return sum(unique.values()) / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("calibration.json"))
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--mode", choices=("filter", "rank"), default="rank")
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--target-precision", type=float)
    parser.add_argument("--target-recall", type=float)
    parser.add_argument("--target-recall-at-k", type=int, default=40)
    parser.add_argument("--target-recall-at", type=float, default=0.95)
    parser.add_argument("--target-auc", type=float)
    parser.add_argument("--target-fp-per-hour", type=float)
    parser.add_argument("--target-candidates-per-hour", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = []
    for path in args.reports:
        if path.is_dir():
            paths.extend(path.rglob("report.json"))
            paths.extend(path.rglob("*.jsonl"))
        else:
            paths.append(path)
    (
        rows,
        labels,
        baselines,
        groups,
        episodes,
        durations,
        texts,
        run_ids,
        timestamps,
    ) = _load_rows(sorted(set(paths)))
    if len(rows) == 0 or labels.sum() == 0 or (len(labels) - labels.sum()) == 0:
        print(f"not enough labels: samples={len(rows)} ads={int(labels.sum())}")
        metrics = {
            "samples": len(rows),
            "ads": int(labels.sum()) if len(labels) else 0,
            "error": "not enough labels",
            "target_met": False,
        }
        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
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

    learned_threshold, learned_metrics = _choose_threshold(
        learned_cv, labels, args.min_recall, args.target_precision
    )
    baseline_threshold, baseline_metrics = _choose_threshold(
        baselines, labels, args.min_recall, args.target_precision
    )
    learned_auc = _auc(learned_cv, labels)
    baseline_auc = _auc(baselines, labels)
    print(f"learned  auc={learned_auc:.3f} threshold={learned_threshold:.2f} {learned_metrics}")
    print(f"baseline auc={baseline_auc:.3f} threshold={baseline_threshold:.2f} {baseline_metrics}")

    ks = (1, 2, 3, 5, 8, 10, 12, 15, 20, 24, 30, 40)
    recall_at, precision_at, false_positive_at, selected_at = _at_k(learned_cv, labels, episodes, ks)
    print("recall@k:", {k: round(v, 2) for k, v in recall_at.items()})
    print("precision@k:", {k: round(v, 2) for k, v in precision_at.items()})
    if args.mode == "rank":
        top_k = 0
        for k in (8, 10, 12, 15, 20, 24, 30, 40):
            if recall_at.get(k, 0.0) >= args.min_recall:
                top_k = k
                break
        if top_k == 0:
            top_k = 40
        precision = precision_at[top_k]
        recall = recall_at[top_k]
        false_positives = false_positive_at[top_k]
        selected = selected_at[top_k]
    else:
        top_k = 0
        predicted = learned_cv >= learned_threshold
        precision = learned_metrics["precision"]
        recall = learned_metrics["recall"]
        false_positives = int((predicted & (labels == 0)).sum())
        selected = int(predicted.sum())

    hours = _episode_hours(episodes, durations)
    fp_per_hour = round(false_positives / hours, 2) if hours > 0 else None
    candidates_per_hour = round(selected / hours, 2) if hours > 0 else None

    # The gate must measure the round that is actually reviewed. Older rounds
    # accumulate different DAI creatives for the same episode, which inflates
    # the per-episode candidate count and makes recall@K meaningless.
    latest_run = ""
    latest_timestamp = ""
    for run_id, timestamp in zip(run_ids, timestamps):
        if timestamp and timestamp > latest_timestamp:
            latest_timestamp = timestamp
            latest_run = run_id
    latest_indices = (
        [index for index, run_id in enumerate(run_ids) if run_id == latest_run]
        if latest_run
        else list(range(len(rows)))
    )
    if not latest_indices:
        latest_indices = list(range(len(rows)))
    latest_scores = learned_cv[latest_indices]
    latest_labels = labels[latest_indices]
    latest_episodes = [episodes[index] for index in latest_indices]
    latest_durations = [durations[index] for index in latest_indices]
    latest_recall_at, latest_precision_at, latest_fp_at, latest_selected_at = _at_k(
        latest_scores, latest_labels, latest_episodes, ks
    )
    latest_hours = _episode_hours(latest_episodes, latest_durations)
    if args.mode == "rank":
        latest_selected = latest_selected_at[top_k]
        latest_false_positives = latest_fp_at[top_k]
        latest_precision = latest_precision_at[top_k]
        latest_recall = latest_recall_at[top_k]
    else:
        latest_predicted = latest_scores >= learned_threshold
        latest_selected = int(latest_predicted.sum())
        latest_false_positives = int((latest_predicted & (latest_labels == 0)).sum())
        latest_true_positive = int((latest_predicted & (latest_labels == 1)).sum())
        latest_precision = latest_true_positive / max(1, latest_selected)
        latest_recall = latest_true_positive / max(1, int(latest_labels.sum()))
    latest_ads = int(latest_labels.sum())
    if latest_ads == 0:
        latest_recall = 1.0
        latest_recall_at = {k: 1.0 for k in ks}
    latest_fp_per_hour = (
        round(latest_false_positives / latest_hours, 2) if latest_hours > 0 else None
    )
    latest_candidates_per_hour = (
        round(latest_selected / latest_hours, 2) if latest_hours > 0 else None
    )
    latest = {
        "run_id": latest_run,
        "samples": len(latest_indices),
        "ads": latest_ads,
        "episode_hours": round(latest_hours, 3),
        "learned": {
            "precision": round(latest_precision, 3),
            "recall": round(latest_recall, 3),
            "auc": round(_auc(latest_scores, latest_labels), 3),
        },
        "recall_at": {str(k): round(v, 3) for k, v in latest_recall_at.items()},
        "precision_at": {str(k): round(v, 3) for k, v in latest_precision_at.items()},
        "fp_per_hour": latest_fp_per_hour,
        "candidates_per_hour": latest_candidates_per_hour,
    }
    print(
        f"latest run={latest_run} samples={latest['samples']} ads={latest_ads} "
        f"recall@{args.target_recall_at_k}={latest_recall_at.get(args.target_recall_at_k)} "
        f"candidates/hour={latest_candidates_per_hour}"
    )

    terms = load_terms(Path("feedback/keywords.json"))
    hits = [bool(match_terms(text, terms)) for text in texts]
    total_ads = int(labels.sum())
    keyword_tp = int(sum(1 for hit, label in zip(hits, labels) if hit and label == 1))
    keyword_fp = int(sum(1 for hit, label in zip(hits, labels) if hit and label == 0))
    keyword_metrics = {
        "terms": len(terms),
        "texts": int(sum(1 for text in texts if text)),
        "hits": keyword_tp + keyword_fp,
        "precision": round(keyword_tp / max(1, keyword_tp + keyword_fp), 3),
        "recall": round(keyword_tp / max(1, total_ads), 3),
    }

    targets = {
        "precision": args.target_precision,
        "recall": args.target_recall,
        "recall_at_k": args.target_recall_at_k,
        "recall_at": args.target_recall_at,
        "auc": args.target_auc,
        "fp_per_hour": args.target_fp_per_hour,
        "candidates_per_hour": args.target_candidates_per_hour,
    }
    checks = []
    if args.target_auc is not None:
        checks.append(learned_auc >= args.target_auc)
    if latest_ads > 0:
        if args.target_recall_at_k and args.target_recall_at is not None:
            checks.append(
                latest_recall_at.get(args.target_recall_at_k, 0.0) >= args.target_recall_at
            )
        if args.target_recall is not None:
            checks.append(latest_recall >= args.target_recall)
        if args.target_precision is not None and args.target_precision > 0:
            checks.append(latest_precision >= args.target_precision)
    if args.target_fp_per_hour is not None and latest_fp_per_hour is not None:
        checks.append(latest_fp_per_hour <= args.target_fp_per_hour)
    if args.target_candidates_per_hour is not None and latest_candidates_per_hour is not None:
        checks.append(latest_candidates_per_hour <= args.target_candidates_per_hour)
    target_met = bool(checks) and all(checks)

    improved = learned_auc > baseline_auc + 0.01
    metrics = {
        "mode": args.mode,
        "samples": len(rows),
        "ads": int(labels.sum()),
        "non_ads": int(len(rows) - labels.sum()),
        "groups": unique_groups,
        "episode_hours": round(hours, 3),
        "threshold": round(float(learned_threshold), 3),
        "top_k": int(top_k),
        "learned": {**learned_metrics, "auc": round(learned_auc, 3)},
        "baseline": {**baseline_metrics, "auc": round(baseline_auc, 3)},
        "recall_at": {str(k): round(v, 3) for k, v in recall_at.items()},
        "precision_at": {str(k): round(v, 3) for k, v in precision_at.items()},
        "latest": latest,
        "keyword": keyword_metrics,
        "fp_per_hour": fp_per_hour,
        "candidates_per_hour": candidates_per_hour,
        "targets": targets,
        "target_met": target_met,
    }
    print(
        f"fp/hour={fp_per_hour} candidates/hour={candidates_per_hour} "
        f"latest_fp/hour={latest_fp_per_hour} latest_candidates/hour={latest_candidates_per_hour} "
        f"target_met={target_met} targets={targets}"
    )
    print(f"keyword={keyword_metrics}")

    standardized, mean, scale = _standardize(x)
    weights, intercept = _fit_logistic(standardized, labels)
    payload = {
        "enabled": bool(improved),
        "mode": args.mode,
        "top_k": int(top_k),
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "intercept": float(intercept),
        "threshold": round(float(learned_threshold), 3),
        "min_recall": args.min_recall,
        "samples": len(rows),
        "ads": int(labels.sum()),
        "groups": unique_groups,
        "metrics": metrics,
    }
    print("enabled:", payload["enabled"], "mode:", args.mode, "top_k:", top_k)
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("wrote metrics", args.metrics_out)
    if not args.dry_run:
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
