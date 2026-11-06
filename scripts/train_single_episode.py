#!/usr/bin/env python3
"""Train the single-recording ad classifier and boundary model.

Runtime uses only one audio file; training may use many audio files with
labels. Two supervision modes are supported:

1. Full supervision — ``dataset/annotations.jsonl`` lines:
   ``{"audio": "...", "feed": "...", "episode": "...", "ads": [{"start_s": 1.0, "end_s": 31.0}]}``
2. Weak supervision — ``feedback/labels.jsonl`` plus an audio manifest:
   ``{"feed": "...", "episode": "...", "audio": "..."}``.
   Candidate spans are used as noisy positives/negatives and no event-level
   metric is reported.

Training-time augmentation is supported by listing augmented variants as
additional annotation lines with the same ``feed`` value, so they stay in the
same validation group.

All math is plain NumPy (Newton logistic regression); nothing here runs at
detection time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.audio import file_sha256
from dai_detect.boundary_model import (
    BOUNDARY_FEATURES,
    boundary_change_matrix,
    heuristic_boundary_scores,
)
from dai_detect.decode_config import resolve_thresholds
from dai_detect.evaluation import evaluate_events
from dai_detect.metrics import best_f1_threshold, roc_auc, threshold_metrics
from dai_detect.refinement import refine_events
from dai_detect.single_episode import extract_features
from dai_detect.single_features import WINDOW_FEATURES, build_windows, window_labels
from dai_detect.temporal_decoder import decode_windows


@dataclass
class EpisodeSample:
    feed: str
    episode: str
    audio: str
    windows: dict
    indices: np.ndarray
    labels: np.ndarray
    ads: Optional[list[dict]]
    duration_s: float
    change_scores: np.ndarray
    change_grid_s: float
    boundary_features: np.ndarray


def _load_annotations(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not payload.get("audio"):
            continue
        rows.append(payload)
    return rows


def _resolve_audio(value: str, base: Path) -> str:
    if value.startswith(("http://", "https://")):
        return value
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    local = base / candidate
    if local.exists():
        return str(local)
    return str(candidate)


def _positive_frames(times: np.ndarray, edges: list[float], half: float = 0.30) -> list[int]:
    indices: list[int] = []
    for edge in edges:
        low = int(np.searchsorted(times, edge - half, side="left"))
        high = int(np.searchsorted(times, edge + half, side="right"))
        indices.extend(range(low, high))
    return sorted({index for index in indices if 0 <= index < len(times)})


def _sample_boundary_heads(
    change: dict,
    ads: list[dict],
    seed: int,
    negative_ratio: int = 5,
) -> dict:
    """Return separate onset/offset training matrices and labels."""
    features = np.asarray(change.get("features", np.zeros((0, len(BOUNDARY_FEATURES)))))
    times = np.asarray(change.get("times", []))
    empty = {
        "onset_X": np.zeros((0, len(BOUNDARY_FEATURES)), dtype=np.float32),
        "onset_y": np.zeros(0),
        "offset_X": np.zeros((0, len(BOUNDARY_FEATURES)), dtype=np.float32),
        "offset_y": np.zeros(0),
    }
    if len(times) == 0 or len(features) == 0 or not ads:
        return empty
    starts = [float(ad["start_s"]) for ad in ads]
    ends = [float(ad["end_s"]) for ad in ads]
    edges = starts + ends
    onset_positive = _positive_frames(times, starts)
    offset_positive = _positive_frames(times, ends)

    rng = np.random.default_rng(seed)
    wanted = min(
        len(times),
        max(1, max(len(onset_positive), len(offset_positive)) * negative_ratio),
    )
    negatives: list[int] = []
    attempts = 0
    while len(negatives) < wanted and attempts < wanted * 30:
        attempts += 1
        index = int(rng.integers(0, len(times)))
        nearest = min((abs(float(times[index]) - edge) for edge in edges), default=1e9)
        if nearest >= 5.0:
            negatives.append(index)
    negatives = sorted(set(negatives))

    def _build(positives: list[int]) -> tuple[np.ndarray, np.ndarray]:
        if not positives:
            return empty["onset_X"], empty["onset_y"]
        X = np.vstack([features[positives], features[negatives]]).astype(np.float32)
        y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        return X, y

    onset_X, onset_y = _build(onset_positive)
    offset_X, offset_y = _build(offset_positive)
    return {
        "onset_X": onset_X,
        "onset_y": onset_y,
        "offset_X": offset_X,
        "offset_y": offset_y,
    }


def _weak_labels(windows: dict, labels: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Window labels from candidate-level LLM rows (positives win ties)."""
    span = float(windows["span_s"])
    times = np.asarray(windows["times"])
    chosen: dict[int, int] = {}
    for row in labels:
        start = float(row.get("start_s") or 0.0)
        end = float(row.get("end_s") or 0.0)
        label = 1 if row.get("is_ad") else 0
        for index, center in enumerate(times):
            if start <= center <= end:
                if index not in chosen or label == 1:
                    chosen[index] = label
    indices = np.asarray(sorted(chosen), dtype=np.int64)
    labels_y = np.asarray([chosen[index] for index in indices], dtype=np.float64)
    return indices, labels_y


def _load_weak(path: Path, manifest: Path, max_episodes: int) -> tuple[list[tuple[dict, list[dict]]], int]:
    rows_by_key: dict[tuple[str, str, str], list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        audio_id = str(row.get("audio_id") or "")
        if not audio_id:
            # Legacy labels without an audio identity cannot be tied to a
            # training snapshot; using them would silently train on a different
            # DAI variant.
            continue
        key = (str(row.get("feed") or ""), str(row.get("episode") or ""), audio_id)
        rows_by_key.setdefault(key, []).append(row)
    manifest_rows = []
    skipped = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        audio_id = str(payload.get("audio_id") or "")
        if not audio_id:
            skipped += 1
            continue
        key = (str(payload.get("feed") or ""), str(payload.get("episode") or ""), audio_id)
        labels = rows_by_key.get(key, [])
        if labels and payload.get("audio"):
            manifest_rows.append((payload, labels))
        if max_episodes and len(manifest_rows) >= max_episodes:
            break
    return manifest_rows, skipped


def _identity_problems(entry: dict, audio: str, mono: dict, allow_unverified: bool) -> list[str]:
    problems: list[str] = []
    source_hash = str(entry.get("source_sha256") or "")
    audio_id = str(entry.get("audio_id") or "")
    remote = audio.startswith(("http://", "https://"))
    if source_hash and not remote:
        actual = file_sha256(audio)
        if actual != source_hash:
            problems.append(
                f"source_sha256 不匹配（标注 {source_hash[:12]}…, 文件 {actual[:12] or 'unreadable'}…）"
            )
    if audio_id:
        actual_id = str(mono.get("audio_id") or "")
        if actual_id != audio_id:
            problems.append(f"audio_id 不匹配（标注 {audio_id[:12]}…, 解码 {actual_id[:12] or 'empty'}…）")
        kind = str(entry.get("audio_id_kind") or "")
        actual_kind = str(mono.get("audio_id_kind") or "")
        if kind and actual_kind and kind != actual_kind:
            problems.append(f"audio_id_kind 不一致（标注 {kind}, 解码 {actual_kind}）")
    if remote and source_hash and not audio_id:
        problems.append("远程音频只有 source_sha256，无法校验字节；请先用 scripts/snapshot_audio.py 归档")
    if not source_hash and not audio_id and not allow_unverified:
        problems.append("缺少 source_sha256/audio_id，无法证明标注与训练音频是同一份")
    return problems


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 20) -> dict:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    Z = (X - mean) / scale
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    weight_positive = len(y) / (2.0 * positives)
    weight_negative = len(y) / (2.0 * negatives)
    sample_weights = np.where(y == 1, weight_positive, weight_negative)
    weights = np.zeros(Z.shape[1])
    weights[-1] = math.log(positives / negatives)
    regularization = np.ones(Z.shape[1])
    regularization[-1] = 0.0
    for _ in range(max_iter):
        logits = np.clip(Z @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        curvature = sample_weights * probabilities * (1.0 - probabilities) + 1e-6
        hessian = Z.T @ (Z * curvature[:, None]) + l2 * np.diag(regularization) + 1e-6 * np.eye(Z.shape[1])
        gradient = Z.T @ (sample_weights * (probabilities - y)) + l2 * (weights * regularization)
        delta = np.linalg.solve(hessian, gradient)
        weights -= delta
        if np.linalg.norm(delta) < 1e-6:
            break
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights[:-1].tolist(),
        "intercept": float(weights[-1]),
    }


def _predict(model: dict, X: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    weights = np.asarray(model["weights"])
    logits = np.clip(((X - mean) / scale) @ weights + model["intercept"], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _balance(X: np.ndarray, y: np.ndarray, ratio: int = 4, seed: int = 13) -> tuple[np.ndarray, np.ndarray]:
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if len(negatives) == 0 or len(positives) == 0:
        return X, y
    wanted = min(len(negatives), len(positives) * ratio)
    rng = np.random.default_rng(seed)
    negatives = rng.choice(negatives, size=wanted, replace=False)
    indices = np.concatenate([positives, negatives])
    return X[indices], y[indices]


def _group_folds(groups: list[str]) -> list[list[int]]:
    """Leave-one-feed-out folds: each fold returns the held-out indices.

    The caller marks exactly these indices as test rows. Returning the
    complement here would train on the held-out feed and test on everything
    else, which is the reverse of grouped cross-validation.
    """
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError(
            "至少需要两个独立节目（feed）才能做分组交叉验证；"
            "同一节目内部高度相似，单节目评测会过于乐观"
        )
    return [
        [index for index, group in enumerate(groups) if group == held]
        for held in unique
    ]


def _window_rows(indices: list[int], samples: list[EpisodeSample], offsets: dict[int, tuple[int, int]], total: int) -> np.ndarray:
    mask = np.zeros(total, dtype=bool)
    for index in indices:
        begin, end = offsets[id(samples[index])]
        mask[begin:end] = True
    return mask


def _inner_folds(train_indices: list[int], samples: list[EpisodeSample]) -> list[list[int]]:
    """Folds used only inside an outer training set for threshold selection."""
    feeds = sorted({samples[index].feed for index in train_indices})
    if len(feeds) >= 2:
        return [[index for index in train_indices if samples[index].feed == feed] for feed in feeds]
    ordered = sorted(train_indices)
    if len(ordered) >= 2:
        middle = len(ordered) // 2
        return [ordered[:middle], ordered[middle:]]
    return [ordered]


def _choose_classifier_threshold(
    train_indices: list[int],
    samples: list[EpisodeSample],
    X_all: np.ndarray,
    y_all: np.ndarray,
    offsets: dict[int, tuple[int, int]],
    seed: int,
) -> tuple[float, dict]:
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for fold_index, held in enumerate(_inner_folds(train_indices, samples)):
        inner_train = [index for index in train_indices if index not in set(held)]
        train_rows = _window_rows(inner_train, samples, offsets, len(y_all))
        test_rows = _window_rows(held, samples, offsets, len(y_all))
        if len(np.unique(y_all[train_rows])) < 2 or len(np.unique(y_all[test_rows])) < 2:
            continue
        model = _fit_logistic(*_balance(X_all[train_rows], y_all[train_rows], ratio=4, seed=seed + fold_index))
        probabilities.append(_predict(model, X_all[test_rows]))
        labels.append(y_all[test_rows])
    if not probabilities:
        return 0.5, {"source": "fallback", "reason": "no_valid_inner_fold"}
    probabilities_all = np.concatenate(probabilities)
    labels_all = np.concatenate(labels)
    if len(np.unique(labels_all)) < 2:
        return 0.5, {"source": "fallback", "reason": "single_class_inner_oof"}
    threshold, metrics = best_f1_threshold(labels_all, probabilities_all)
    metrics["auc"] = round(roc_auc(labels_all, probabilities_all), 4)
    metrics["source"] = "inner_grouped_cv"
    metrics["folds"] = len(probabilities)
    return threshold, metrics


def _choose_boundary_thresholds(
    train_indices: list[int],
    samples: list[EpisodeSample],
    onset_X: list[np.ndarray],
    onset_y: list[np.ndarray],
    offset_X: list[np.ndarray],
    offset_y: list[np.ndarray],
    episode_indices: list[int],
    seed: int,
) -> tuple[float, float, dict, dict]:
    onset_probabilities: list[np.ndarray] = []
    onset_labels: list[np.ndarray] = []
    offset_probabilities: list[np.ndarray] = []
    offset_labels: list[np.ndarray] = []
    for fold_index, held in enumerate(_inner_folds(train_indices, samples)):
        held_set = set(held)
        train_set = {index for index in train_indices if index not in held_set}
        onset_train_X, onset_train_y = _gather_boundary(onset_X, onset_y, episode_indices, train_set)
        onset_test_X, onset_test_y = _gather_boundary(onset_X, onset_y, episode_indices, held_set)
        offset_train_X, offset_train_y = _gather_boundary(offset_X, offset_y, episode_indices, train_set)
        offset_test_X, offset_test_y = _gather_boundary(offset_X, offset_y, episode_indices, held_set)
        if len(np.unique(onset_train_y)) >= 2 and len(np.unique(onset_test_y)) >= 2:
            head = _fit_logistic(*_balance(onset_train_X, onset_train_y, ratio=6, seed=seed + fold_index))
            onset_probabilities.append(_predict(head, onset_test_X))
            onset_labels.append(onset_test_y)
        if len(np.unique(offset_train_y)) >= 2 and len(np.unique(offset_test_y)) >= 2:
            head = _fit_logistic(*_balance(offset_train_X, offset_train_y, ratio=6, seed=seed + 100 + fold_index))
            offset_probabilities.append(_predict(head, offset_test_X))
            offset_labels.append(offset_test_y)

    onset_threshold, onset_metrics = 0.5, {"source": "fallback"}
    offset_threshold, offset_metrics = 0.5, {"source": "fallback"}
    if onset_probabilities:
        on_prob = np.concatenate(onset_probabilities)
        on_labels = np.concatenate(onset_labels)
        if len(np.unique(on_labels)) >= 2:
            onset_threshold, onset_metrics = best_f1_threshold(on_labels, on_prob)
            onset_metrics["auc"] = round(roc_auc(on_labels, on_prob), 4)
            onset_metrics["source"] = "inner_grouped_cv"
    if offset_probabilities:
        off_prob = np.concatenate(offset_probabilities)
        off_labels = np.concatenate(offset_labels)
        if len(np.unique(off_labels)) >= 2:
            offset_threshold, offset_metrics = best_f1_threshold(off_labels, off_prob)
            offset_metrics["auc"] = round(roc_auc(off_labels, off_prob), 4)
            offset_metrics["source"] = "inner_grouped_cv"
    return onset_threshold, offset_threshold, onset_metrics, offset_metrics


def _gather_boundary(
    matrices: list[np.ndarray],
    labels: list[np.ndarray],
    episode_indices: list[int],
    allowed: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not matrices:
        return np.zeros((0, len(BOUNDARY_FEATURES)), dtype=np.float32), np.zeros(0)
    chosen_X = [
        matrix
        for matrix, episode in zip(matrices, episode_indices)
        if episode in allowed and len(matrix)
    ]
    chosen_y = [
        values
        for values, episode in zip(labels, episode_indices)
        if episode in allowed and len(values)
    ]
    if not chosen_X:
        return np.zeros((0, len(BOUNDARY_FEATURES)), dtype=np.float32), np.zeros(0)
    return np.vstack(chosen_X).astype(np.float32), np.concatenate(chosen_y)


def _event_metrics(
    samples: list[EpisodeSample],
    indices: list[int],
    probabilities: dict[int, np.ndarray],
    thresholds: dict[int, float] | float,
    refine: bool = True,
    boundary_heads: Optional[dict[int, tuple[dict, dict, float, float]]] = None,
) -> dict:
    predictions: list[dict] = []
    ground_truth: list[dict] = []
    total_duration = 0.0
    labeled = [index for index in indices if samples[index].ads is not None]
    if not labeled:
        return {}
    for index in labeled:
        sample = samples[index]
        probability = np.asarray(probabilities[index])
        expected = len(sample.windows["times"])
        if len(probability) != expected:
            raise ValueError(
                f"事件评估概率长度 {len(probability)} 与完整窗口数 {expected} 不一致；"
                "必须用该折模型对全部窗口重新预测"
            )
        fold_threshold = (
            float(thresholds.get(index, 0.5))
            if isinstance(thresholds, dict)
            else float(thresholds)
        )
        enter, exit_value = resolve_thresholds(fold_threshold)
        events = decode_windows(
            sample.windows["times"],
            probability,
            hop_s=float(sample.windows["hop_s"]),
            span_s=float(sample.windows["span_s"]),
            min_ad_s=6.0,
            enter_threshold=enter,
            exit_threshold=exit_value,
        )
        if refine:
            if boundary_heads and index in boundary_heads:
                onset_head, offset_head, onset_threshold, offset_threshold = boundary_heads[index]
                features = sample.boundary_features.astype(np.float32)
                onset_scores = _predict(onset_head, features)
                offset_scores = _predict(offset_head, features)
                change = {
                    "times": np.arange(len(features)) * sample.change_grid_s,
                    "onset_scores": onset_scores,
                    "offset_scores": offset_scores,
                    "scores": np.maximum(onset_scores, offset_scores),
                    "kind": "learned_onset_offset",
                    "onset_threshold": onset_threshold,
                    "offset_threshold": offset_threshold,
                }
            else:
                change = {
                    "times": np.arange(len(sample.change_scores)) * sample.change_grid_s,
                    "scores": sample.change_scores,
                    "kind": "heuristic",
                }
            refined = refine_events(events, change, search_s=2.5, min_duration=6.0)
        else:
            refined = [
                {
                    "start_s": event.start,
                    "end_s": event.end,
                }
                for event in events
            ]
        predictions.extend(refined)
        ground_truth.extend(
            [{"start_s": float(ad["start_s"]), "end_s": float(ad["end_s"])} for ad in sample.ads]
        )
        total_duration += sample.duration_s
    metrics = evaluate_events(predictions, ground_truth, duration_s=total_duration)
    return {
        key: metrics[key]
        for key in (
            "event_precision",
            "event_recall",
            "event_f1",
            "boundary_p95_s",
            "false_positives",
            "false_negatives",
            "false_positives_per_hour",
            "ground_truth_events",
            "predicted_events",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, help="完全监督标注 JSONL")
    parser.add_argument("--labels", type=Path, help="LLM 候选标注（弱监督）")
    parser.add_argument("--audio-manifest", type=Path, help="弱监督音频清单 JSONL")
    parser.add_argument("--model-out", type=Path, default=Path("models/ad_segment_model.json"))
    parser.add_argument("--boundary-out", type=Path, default=Path("models/boundary_model.json"))
    parser.add_argument("--proxy", help="HTTP(S) 代理")
    parser.add_argument("--max-episodes", type=int, default=0, help="最多处理多少集，0 表示全部")
    parser.add_argument("--skip-boundary", action="store_true", help="不训练边界模型")
    parser.add_argument(
        "--allow-unverified-audio",
        action="store_true",
        help="允许没有 source_sha256/audio_id 的标注（默认拒绝，避免训练到不同 DAI 变体）",
    )
    parser.add_argument(
        "--allow-audio-mismatch",
        action="store_true",
        help="音频身份不匹配时仅警告并继续（默认中止）",
    )
    args = parser.parse_args()

    annotations_path = args.annotations
    weak_mode = False
    if annotations_path is None:
        if not (args.labels and args.audio_manifest):
            parser.error("需要 --annotations，或同时提供 --labels 与 --audio-manifest")
        weak_mode = True
    base = (annotations_path.parent if annotations_path else args.audio_manifest.parent).resolve()
    if annotations_path:
        entries = _load_annotations(annotations_path)
        weak_pairs: list[tuple[dict, list[dict]]] = []
    else:
        weak_pairs, weak_skipped = _load_weak(args.labels, args.audio_manifest, args.max_episodes)
        print(f"[identity] 弱监督清单缺少 audio_id 而跳过 {weak_skipped} 条")
        entries = [
            {**payload, "ads": None}
            for payload, _ in weak_pairs
        ]
    weak_by_key = (
        {
            (
                str(payload.get("feed") or ""),
                str(payload.get("episode") or ""),
                str(payload.get("audio_id") or ""),
            ): labels
            for payload, labels in weak_pairs
        }
        if weak_mode
        else {}
    )
    if args.max_episodes and len(entries) > args.max_episodes:
        entries = entries[: args.max_episodes]

    samples: list[EpisodeSample] = []
    boundary_onset_X: list[np.ndarray] = []
    boundary_onset_y: list[np.ndarray] = []
    boundary_offset_X: list[np.ndarray] = []
    boundary_offset_y: list[np.ndarray] = []
    boundary_groups: list[str] = []
    boundary_episode_indices: list[int] = []
    audio_index = 0
    for entry in entries:
        audio = _resolve_audio(str(entry["audio"]), base)
        feed = str(entry.get("feed") or "unknown")
        name = str(entry.get("episode") or Path(audio).name)
        print(f"[extract] {feed} / {name}", flush=True)
        mono, forensic = extract_features(audio, proxy=args.proxy)
        identity = _identity_problems(entry, audio, mono, args.allow_unverified_audio)
        if identity:
            for problem in identity:
                print(f"[identity] {name}: {problem}", flush=True)
            if not args.allow_audio_mismatch:
                print("[identity] 中止训练；如确认音频一致可用 --allow-audio-mismatch 覆盖")
                return 1
        windows = build_windows(mono, forensic)
        duration = float(windows["duration_s"])
        if not weak_mode:
            ads = [
                {"start_s": float(item["start_s"]), "end_s": float(item["end_s"])}
                for item in (entry.get("ads") or [])
            ]
            indices, labels = window_labels(windows, ads)
        else:
            key = (
                str(entry.get("feed") or ""),
                str(entry.get("episode") or ""),
                str(entry.get("audio_id") or ""),
            )
            rows = weak_by_key.get(key)
            ads = None
            indices, labels = _weak_labels(windows, rows or [])
        if len(indices) == 0:
            print(f"[skip] {name}: 无可用标注窗口")
            continue
        change = boundary_change_matrix(mono, forensic)
        scores = heuristic_boundary_scores(change["features"])
        if not weak_mode and not args.skip_boundary and ads:
            heads = _sample_boundary_heads(change, ads, seed=audio_index + 7)
            if len(heads["onset_y"]) or len(heads["offset_y"]):
                episode_index = len(samples)
                boundary_onset_X.append(heads["onset_X"])
                boundary_onset_y.append(heads["onset_y"])
                boundary_offset_X.append(heads["offset_X"])
                boundary_offset_y.append(heads["offset_y"])
                boundary_groups.append(feed)
                boundary_episode_indices.append(episode_index)
        samples.append(
            EpisodeSample(
                feed=feed,
                episode=name,
                audio=audio,
                windows=windows,
                indices=indices,
                labels=labels,
                ads=ads,
                duration_s=duration,
                change_scores=scores,
                change_grid_s=float(change["times"][1] - change["times"][0]) if len(change["times"]) > 1 else 0.01,
                boundary_features=change["features"].astype(np.float16),
            )
        )
        audio_index += 1

    if not samples:
        print("没有可用样本，未训练模型")
        return 1

    X_parts = [sample.windows["features"][sample.indices] for sample in samples]
    y_parts = [sample.labels for sample in samples]
    X_all = np.vstack(X_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.float64)
    groups = [sample.feed for sample in samples]
    if len(np.unique(y_all)) < 2:
        print("样本只有单一类别，无法训练")
        return 1
    print(f"[data] {len(samples)} 集, {int(y_all.sum())} 正窗口, {int(len(y_all) - y_all.sum())} 负窗口")

    # Out-of-fold probabilities for threshold selection and honest metrics.
    try:
        folds = _group_folds(groups)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1
    offset = 0
    per_episode_offset: dict[int, tuple[int, int]] = {}
    for sample in samples:
        per_episode_offset[id(sample)] = (offset, offset + len(sample.labels))
        offset += len(sample.labels)
    oof = np.zeros(len(y_all), dtype=np.float64)
    full_oof: dict[int, np.ndarray] = {}
    fold_boundary_heads: dict[int, tuple[dict, dict, float, float]] = {}
    fold_thresholds: dict[int, float] = {}
    boundary_onset_oof: list[np.ndarray] = []
    boundary_onset_labels: list[np.ndarray] = []
    boundary_offset_oof: list[np.ndarray] = []
    boundary_offset_labels: list[np.ndarray] = []
    fold_reports: list[dict] = []
    all_indices = list(range(len(samples)))

    for fold_index, held in enumerate(folds):
        held_set = set(held)
        outer_train = [index for index in all_indices if index not in held_set]
        train_rows = _window_rows(outer_train, samples, per_episode_offset, len(y_all))
        held_rows = _window_rows(held, samples, per_episode_offset, len(y_all))
        train_y = y_all[train_rows]
        if len(np.unique(train_y)) < 2:
            print(
                f"[error] 第 {fold_index + 1} 折训练集只有一个类别，无法进行有效评估"
            )
            return 1

        fold_threshold, threshold_metrics_info = _choose_classifier_threshold(
            outer_train, samples, X_all, y_all, per_episode_offset, seed=fold_index + 1
        )
        for index in held:
            fold_thresholds[index] = fold_threshold

        model = _fit_logistic(*_balance(X_all[train_rows], train_y, ratio=4, seed=fold_index + 1))
        oof[held_rows] = _predict(model, X_all[held_rows])
        for index in held:
            full_oof[index] = _predict(model, samples[index].windows["features"])

        allowed = set(outer_train)
        onset_X, onset_y = _gather_boundary(
            boundary_onset_X, boundary_onset_y, boundary_episode_indices, allowed
        )
        offset_X, offset_y = _gather_boundary(
            boundary_offset_X, boundary_offset_y, boundary_episode_indices, allowed
        )
        onset_head = offset_head = None
        onset_threshold = offset_threshold = 0.5
        if boundary_episode_indices:
            if len(onset_y) == 0 or len(offset_y) == 0:
                print(f"[error] 第 {fold_index + 1} 折训练集没有可用的边界标注")
                return 1
            if len(np.unique(onset_y)) < 2 or len(np.unique(offset_y)) < 2:
                print(f"[error] 第 {fold_index + 1} 折边界训练集只有一个类别")
                return 1
            onset_threshold, offset_threshold, _, _ = _choose_boundary_thresholds(
                outer_train,
                samples,
                boundary_onset_X,
                boundary_onset_y,
                boundary_offset_X,
                boundary_offset_y,
                boundary_episode_indices,
                seed=fold_index + 31,
            )
            onset_head = _fit_logistic(*_balance(onset_X, onset_y, ratio=6, seed=fold_index + 11))
            offset_head = _fit_logistic(*_balance(offset_X, offset_y, ratio=6, seed=fold_index + 21))

        if onset_head is not None and offset_head is not None:
            for index in held:
                if index in boundary_episode_indices:
                    fold_boundary_heads[index] = (
                        onset_head,
                        offset_head,
                        onset_threshold,
                        offset_threshold,
                    )
            for entry, episode in enumerate(boundary_episode_indices):
                if episode not in held_set:
                    continue
                if len(boundary_onset_y[entry]):
                    boundary_onset_oof.append(_predict(onset_head, boundary_onset_X[entry]))
                    boundary_onset_labels.append(boundary_onset_y[entry])
                if len(boundary_offset_y[entry]):
                    boundary_offset_oof.append(_predict(offset_head, boundary_offset_X[entry]))
                    boundary_offset_labels.append(boundary_offset_y[entry])

        events = _event_metrics(
            samples,
            held,
            full_oof,
            thresholds=fold_thresholds,
            refine=True,
            boundary_heads=fold_boundary_heads,
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "held": held,
                "threshold": round(fold_threshold, 4),
                "threshold_metrics": threshold_metrics_info,
                "events": events,
            }
        )
        print(
            f"[fold {fold_index + 1}/{len(folds)}] 留出 {len(held)} 集, "
            f"threshold={fold_threshold:.3f}, events={events}",
            flush=True,
        )

    # Threshold used for deployment is also selected without the evaluated
    # episode's labels: inner grouped CV over the whole training set.
    deployment_threshold, deployment_threshold_metrics = _choose_classifier_threshold(
        all_indices, samples, X_all, y_all, per_episode_offset, seed=101
    )
    window_metrics = threshold_metrics(y_all, oof, deployment_threshold)
    window_metrics["auc"] = round(roc_auc(y_all, oof), 4)
    window_metrics["threshold"] = round(deployment_threshold, 4)
    window_metrics["threshold_selection"] = deployment_threshold_metrics
    final_model = _fit_logistic(*_balance(X_all, y_all, ratio=4, seed=101))
    event_metrics = _event_metrics(
        samples,
        all_indices,
        full_oof,
        thresholds=fold_thresholds,
        refine=True,
        boundary_heads=fold_boundary_heads,
    )
    threshold = deployment_threshold

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    identity_rows = [
        {
            "feed": entry.get("feed"),
            "episode": entry.get("episode"),
            "audio": entry.get("audio"),
            "audio_id": entry.get("audio_id"),
            "source_sha256": entry.get("source_sha256"),
            "ads": entry.get("ads"),
        }
        for entry in entries
    ]
    data_version = hashlib.sha256(
        json.dumps(identity_rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    model_version = f"ad-{trained_at.replace('-', '').replace(':', '').replace('+', '')}-{data_version[:8]}"

    payload = {
        "version": 1,
        "kind": "logistic",
        "features": list(WINDOW_FEATURES),
        **final_model,
        "threshold": round(threshold, 4),
        "model_version": model_version,
        "data_version": data_version,
        "git_commit": os.environ.get("GITHUB_SHA", ""),
        "trained_at": trained_at,
        "metrics": {
            "episodes": len(samples),
            "feeds": len(set(groups)),
            "windows": int(len(y_all)),
            "positives": int(y_all.sum()),
            "negatives": int(len(y_all) - y_all.sum()),
            "window": window_metrics,
            "event": event_metrics,
            "folds": fold_reports,
            "supervision": "weak_llm_candidates" if weak_mode else "full_annotations",
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[model] {args.model_out}")
    print(json.dumps({key: payload["metrics"][key] for key in ("window", "event")}, ensure_ascii=False))

    if not weak_mode and not args.skip_boundary and boundary_episode_indices:
        allowed_all = set(range(len(samples)))
        onset_X, onset_y = _gather_boundary(
            boundary_onset_X, boundary_onset_y, boundary_episode_indices, allowed_all
        )
        offset_X, offset_y = _gather_boundary(
            boundary_offset_X, boundary_offset_y, boundary_episode_indices, allowed_all
        )
        if (
            len(onset_y)
            and len(offset_y)
            and len(np.unique(onset_y)) >= 2
            and len(np.unique(offset_y)) >= 2
        ):
            onset_threshold, offset_threshold, onset_selection, offset_selection = (
                _choose_boundary_thresholds(
                    all_indices,
                    samples,
                    boundary_onset_X,
                    boundary_onset_y,
                    boundary_offset_X,
                    boundary_offset_y,
                    boundary_episode_indices,
                    seed=909,
                )
            )
            onset_metrics: dict = {"threshold_selection": onset_selection}
            offset_metrics: dict = {"threshold_selection": offset_selection}
            if boundary_onset_oof:
                on_prob = np.concatenate(boundary_onset_oof)
                on_labels = np.concatenate(boundary_onset_labels)
                onset_metrics.update(threshold_metrics(on_labels, on_prob, onset_threshold))
                onset_metrics["auc"] = round(roc_auc(on_labels, on_prob), 4)
            if boundary_offset_oof:
                off_prob = np.concatenate(boundary_offset_oof)
                off_labels = np.concatenate(boundary_offset_labels)
                offset_metrics.update(threshold_metrics(off_labels, off_prob, offset_threshold))
                offset_metrics["auc"] = round(roc_auc(off_labels, off_prob), 4)

            final_onset = _fit_logistic(*_balance(onset_X, onset_y, ratio=6, seed=707))
            final_offset = _fit_logistic(*_balance(offset_X, offset_y, ratio=6, seed=808))
            boundary_payload = {
                "version": 1,
                "kind": "onset_offset",
                "features": list(BOUNDARY_FEATURES),
                "onset": final_onset,
                "offset": final_offset,
                "threshold": round(max(onset_threshold, offset_threshold), 4),
                "onset_threshold": round(onset_threshold, 4),
                "offset_threshold": round(offset_threshold, 4),
                "model_version": f"boundary-{trained_at.replace('-', '').replace(':', '').replace('+', '')}-{data_version[:8]}",
                "data_version": data_version,
                "git_commit": payload["git_commit"],
                "trained_at": trained_at,
                "metrics": {
                    "samples": int(len(onset_y) + len(offset_y)),
                    "onset": onset_metrics,
                    "offset": offset_metrics,
                },
            }
            args.boundary_out.parent.mkdir(parents=True, exist_ok=True)
            args.boundary_out.write_text(
                json.dumps(boundary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[boundary] {args.boundary_out}")
            print(json.dumps({"onset": onset_metrics, "offset": offset_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
