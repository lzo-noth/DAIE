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
from dai_detect.metrics import (
    best_f1_threshold,
    metrics_from_counts,
    roc_auc,
    threshold_counts,
    threshold_metrics,
)
from dai_detect.refinement import refine_events
from dai_detect.single_episode import extract_features
from dai_detect.feed_identity import canonical_feed_key
from dai_detect.single_features import WINDOW_FEATURES, build_windows, window_labels
from dai_detect.temporal_decoder import decode_windows


MIN_BOUNDARY_EVENT_RECALL = 0.80
MAX_BOUNDARY_FALSE_POSITIVES_PER_HOUR = 15.0
MAX_EVENT_FALSE_POSITIVES_PER_HOUR = 15.0
MIN_AD_S = 10.0
MAX_CALIBRATION_ROUNDS = 5
THRESHOLD_TOLERANCE = 0.01


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
    if len(times) == 0 or len(features) == 0:
        return empty
    rng = np.random.default_rng(seed)
    if not ads:
        # Confirmed no-ad episodes contribute natural production changes as
        # negative boundary examples. This teaches the heads that ordinary
        # cuts, transitions and music changes are not ad boundaries.
        strength = np.mean(np.abs(features), axis=1)
        cutoff = float(np.percentile(strength, 90)) if len(strength) else 0.0
        candidates = np.where(strength >= cutoff)[0]
        budget = max(1, min(len(features), 128))
        if len(candidates) > budget:
            candidates = rng.choice(candidates, size=budget, replace=False)
        chosen = np.unique(candidates)
        if len(chosen) == 0:
            return empty
        X = features[chosen].astype(np.float32)
        y = np.zeros(len(chosen))
        return {
            "onset_X": X,
            "onset_y": y,
            "offset_X": X.copy(),
            "offset_y": y.copy(),
        }

    starts = [float(ad["start_s"]) for ad in ads]
    ends = [float(ad["end_s"]) for ad in ads]
    edges = starts + ends
    onset_positive = _positive_frames(times, starts)
    offset_positive = _positive_frames(times, ends)

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
        if end - start < MIN_AD_S:
            # 与部署端同一广告定义：短于 MIN_AD_S 的候选不作为广告标签。
            continue
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


def _identity_problems(
    entry: dict,
    audio: str,
    mono: dict,
    allow_unverified: bool,
    strict_audio_id: bool = False,
) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    warnings: list[str] = []
    source_hash = str(entry.get("source_sha256") or "")
    audio_id = str(entry.get("audio_id") or "")
    remote = audio.startswith(("http://", "https://"))
    source_verified = False
    if source_hash and not remote:
        actual = file_sha256(audio)
        if actual != source_hash:
            problems.append(
                f"source_sha256 不匹配（标注 {source_hash[:12]}…, 文件 {actual[:12] or 'unreadable'}…）"
            )
        else:
            source_verified = True

    def record(message: str) -> None:
        # PCM 指纹依赖解码器实现；字节一致时把它降级为警告。
        if source_verified and not strict_audio_id:
            warnings.append(f"{message}（source_sha256 已确认字节一致，按解码器差异处理）")
        else:
            problems.append(message)

    kind = str(entry.get("audio_id_kind") or "")
    if audio_id:
        actual_id = str(mono.get("audio_id") or "")
        if actual_id != audio_id:
            record(f"audio_id 不匹配（标注 {audio_id[:12]}…, 解码 {actual_id[:12] or 'empty'}…）")
        actual_kind = str(mono.get("audio_id_kind") or "")
        if not kind:
            problems.append("标注缺少 audio_id_kind（必须标明 PCM 指纹的生成方式）")
        elif not actual_kind:
            problems.append("解码结果缺少 audio_id_kind，无法校验音频身份")
        elif kind != actual_kind:
            record(f"audio_id_kind 不一致（标注 {kind}, 解码 {actual_kind}）")
    elif not allow_unverified:
        problems.append("标注缺少 audio_id（解码 PCM 内容指纹）")
    if not remote and not source_hash and not allow_unverified:
        problems.append("本地音频标注缺少 source_sha256，无法证明标注用的原始字节")
    if remote and source_hash and not audio_id:
        problems.append("远程音频只有 source_sha256，无法校验字节；请先用 scripts/snapshot_audio.py 归档")
    return problems, warnings


def _duplicate_identity_problems(entries: list[dict]) -> list[str]:
    """Reject the same audio (or source bytes) appearing under multiple feeds."""
    by_audio_id: dict[str, set[str]] = {}
    by_source: dict[str, set[str]] = {}
    episodes: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for entry in entries:
        feed = str(entry.get("feed") or "")
        episode = str(entry.get("episode") or "")
        audio_id = str(entry.get("audio_id") or "")
        source_hash = str(entry.get("source_sha256") or "")
        if audio_id:
            by_audio_id.setdefault(audio_id, set()).add(feed)
        if source_hash:
            by_source.setdefault(source_hash, set()).add(feed)
        key = (feed, episode)
        audio = str(entry.get("audio") or "")
        if episode and key in episodes and episodes[key] != audio:
            problems.append(
                f"重复的 feed/episode 标识 {feed!r}/{episode!r} 对应不同音频"
            )
        episodes[key] = audio
    for audio_id, feeds in by_audio_id.items():
        if len(feeds) > 1:
            problems.append(
                f"audio_id {audio_id[:12]}… 同时出现在多个 feed: {sorted(feeds)}"
            )
    for source_hash, feeds in by_source.items():
        if len(feeds) > 1:
            problems.append(
                f"source_sha256 {source_hash[:12]}… 同时出现在多个 feed: {sorted(feeds)}"
            )
    return problems


def _annotation_problems(
    ads: list[dict], duration_s: float, require_ad_type: bool = False
) -> list[str]:
    """Structural checks for full-supervision ad intervals."""
    problems: list[str] = []
    ordered: list[dict] = []
    for index, ad in enumerate(ads):
        try:
            start = float(ad["start_s"])
            end = float(ad["end_s"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"广告区间 #{index + 1} 缺少有效的 start_s/end_s")
            continue
        ad_type = str(ad.get("type") or ad.get("ad_type") or "")
        # 新标签写 third_party_ad，并额外记录内容/制作/插入机制三个维度；
        # 历史数据里的 third_party_inserted 继续接受，但不再作为唯一取值。
        if require_ad_type and ad_type not in (
            "third_party_ad",
            "third_party_inserted",
        ):
            problems.append(
                f"广告区间 #{index + 1} 的 type 必须是 third_party_ad"
            )
        if start < 0:
            problems.append(f"广告区间 #{index + 1} 的 start_s 小于 0")
        if end <= start:
            problems.append(f"广告区间 #{index + 1} 的 end_s 不大于 start_s")
        if duration_s > 0 and end > duration_s + 0.5:
            problems.append(
                f"广告区间 #{index + 1} 的 end_s 超出音频时长 {duration_s:.1f}s"
            )
        ordered.append({"start_s": start, "end_s": end, "index": index + 1})
    ordered.sort(key=lambda item: item["start_s"])
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_s"] < previous["end_s"] - 1e-6:
            problems.append(
                f"广告区间 #{previous['index']} 与 #{current['index']} 重叠"
            )
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


def _training_artifacts_complete(
    boundary_written: bool, weak_mode: bool, skip_boundary: bool
) -> bool:
    return bool(weak_mode or skip_boundary or boundary_written)


def _cv_quality(
    groups: list[str], fold_reports: list[dict], boundary_expected: bool
) -> dict:
    """Classify nested CV as strict only when every inner pipeline fold ran.

    A fold counts as complete only when the classifier and, if boundary labels
    exist, both learned boundary heads trained and produced held-out
    predictions. Attempting a fold is not the same as completing it.
    """
    strict = len(set(groups)) >= 3
    details: list[dict] = []
    for report in fold_reports:
        event = report.get("threshold_metrics") or {}
        attempted = int(event.get("folds_attempted") or 0)
        event_ok = bool(
            event.get("source")
            and str(event.get("source")).startswith("inner_grouped_cv")
            and attempted > 0
            and int(event.get("pipeline_folds_completed") or 0) == attempted
        )
        boundary_ok = True
        if boundary_expected:
            selection = report.get("boundary_threshold_selection") or {}
            for key in ("onset", "offset"):
                metrics = selection.get(key) or {}
                coverage = metrics.get("coverage") or {}
                boundary_attempted = int(coverage.get("folds_attempted") or 0)
                boundary_ok = boundary_ok and bool(
                    boundary_attempted > 0
                    and int(coverage.get("pipeline_folds_completed") or 0) == boundary_attempted
                    and metrics.get("boundary_calibration_valid") is True
                )
        details.append(
            {
                "fold": report.get("fold"),
                "event_source": event.get("source"),
                "event_folds_attempted": attempted,
                "event_pipeline_folds_completed": int(
                    event.get("pipeline_folds_completed") or 0
                ),
                "event_pipeline_complete": event_ok,
                "boundary_pipeline_complete": boundary_ok if boundary_expected else None,
            }
        )
        strict = strict and event_ok and boundary_ok
    return {
        "level": "strict_nested_grouped_cv" if strict else "limited_nested_cv",
        "feeds": len(set(groups)),
        "boundary_expected": boundary_expected,
        "folds": details,
    }


def _calibration_validity(
    event_metrics: dict,
    final_metrics: dict,
    onset_selection: dict,
    offset_selection: dict,
    boundary_expected: bool,
    converged: bool,
    cv_quality_level: str,
    max_false_positives_per_hour: Optional[float] = MAX_EVENT_FALSE_POSITIVES_PER_HOUR,
) -> dict:
    """Final acceptance flags for the deployed threshold combination.

    Selection validity and final-combination validity are separate:
    ``event_metrics`` describes the search result, while ``final_metrics`` is a
    fresh decode/refine/evaluate pass with the exact returned thresholds. Only
    the latter can make a model deployment-eligible, and only with strict
    nested grouped CV.
    """
    attempted = int(event_metrics.get("folds_attempted") or 0)
    event_ok = bool(
        event_metrics.get("objective") == "event_f1"
        and attempted > 0
        and int(event_metrics.get("pipeline_folds_completed") or 0) == attempted
    )
    final_attempted = int(final_metrics.get("folds_attempted") or 0)
    final_coverage_ok = bool(
        final_attempted > 0
        and int(final_metrics.get("pipeline_folds_completed") or 0) == final_attempted
    )
    fp_rate = final_metrics.get("false_positives_per_hour")
    fp_ok = (
        None
        if max_false_positives_per_hour is None or max_false_positives_per_hour <= 0
        else bool(fp_rate is not None and fp_rate <= max_false_positives_per_hour)
    )
    if boundary_expected:
        onset_ok = bool((onset_selection or {}).get("boundary_calibration_valid"))
        offset_ok = bool((offset_selection or {}).get("boundary_calibration_valid"))
    else:
        onset_ok = offset_ok = None
    process_valid = bool(
        event_ok
        and (onset_ok and offset_ok if boundary_expected else True)
        and converged
    )
    deployment_eligible = bool(
        process_valid
        and final_coverage_ok
        and (fp_ok if fp_ok is not None else True)
        and cv_quality_level == "strict_nested_grouped_cv"
    )
    return {
        "valid": deployment_eligible,
        "calibration_process_valid": process_valid,
        "deployment_eligible": deployment_eligible,
        "cv_quality": cv_quality_level,
        "converged": bool(converged),
        "event_threshold_valid": event_ok,
        "event_false_positive_rate_valid": fp_ok,
        "final_coverage_valid": final_coverage_ok,
        "onset_threshold_valid": onset_ok,
        "offset_threshold_valid": offset_ok,
        "final_pipeline_valid": deployment_eligible,
    }


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


def _inner_folds(train_indices: list[int], samples: list[EpisodeSample]) -> tuple[list[list[int]], str]:
    """Folds used only inside an outer training set for threshold selection."""
    feeds = sorted({samples[index].feed for index in train_indices})
    if len(feeds) >= 2:
        return (
            [[index for index in train_indices if samples[index].feed == feed] for feed in feeds],
            "inner_grouped_cv",
        )
    ordered = sorted(train_indices)
    if len(ordered) >= 2:
        middle = len(ordered) // 2
        return [ordered[:middle], ordered[middle:]], "within_feed_fallback"
    return [ordered], "within_feed_fallback"


def _choose_event_threshold(
    train_indices: list[int],
    samples: list[EpisodeSample],
    X_all: np.ndarray,
    y_all: np.ndarray,
    offsets: dict[int, tuple[int, int]],
    seed: int,
    boundary_onset_X: list[np.ndarray] | None = None,
    boundary_onset_y: list[np.ndarray] | None = None,
    boundary_offset_X: list[np.ndarray] | None = None,
    boundary_offset_y: list[np.ndarray] | None = None,
    boundary_episode_indices: list[int] | None = None,
    boundary_thresholds: tuple[float, float] = (0.5, 0.5),
    max_false_positives_per_hour: Optional[float] = MAX_EVENT_FALSE_POSITIVES_PER_HOUR,
) -> tuple[float, dict]:
    """Select the threshold by decoded event F1 on inner held-out episodes.

    Window F1 is only a fallback when the inner holdouts contain no labelled
    ad events (for example weak supervision).
    """
    folds, fold_kind = _inner_folds(train_indices, samples)
    inner_full: dict[int, np.ndarray] = {}
    inner_heads: dict[int, tuple[dict, dict, float, float]] = {}
    has_events = False
    pipeline_required = bool(boundary_episode_indices)
    folds_attempted = len(folds)
    classifier_completed = 0
    pipeline_completed = 0
    for fold_index, held in enumerate(folds):
        inner_train = [index for index in train_indices if index not in set(held)]
        train_rows = _window_rows(inner_train, samples, offsets, len(y_all))
        if len(np.unique(y_all[train_rows])) < 2:
            continue
        model = _fit_logistic(
            *_balance(X_all[train_rows], y_all[train_rows], ratio=4, seed=seed + fold_index)
        )
        classifier_completed += 1

        onset_head = offset_head = None
        if pipeline_required:
            inner_set = set(inner_train)
            onset_X, onset_y = _gather_boundary(
                boundary_onset_X or [], boundary_onset_y or [], boundary_episode_indices, inner_set
            )
            offset_X, offset_y = _gather_boundary(
                boundary_offset_X or [], boundary_offset_y or [], boundary_episode_indices, inner_set
            )
            if len(onset_y) and len(np.unique(onset_y)) >= 2:
                onset_head = _fit_logistic(
                    *_balance(onset_X, onset_y, ratio=6, seed=seed + 11 + fold_index)
                )
            if len(offset_y) and len(np.unique(offset_y)) >= 2:
                offset_head = _fit_logistic(
                    *_balance(offset_X, offset_y, ratio=6, seed=seed + 21 + fold_index)
                )
            if onset_head is None or offset_head is None:
                # Calibration must mirror deployment. If the fold cannot train
                # learned boundary heads, skip it instead of mixing heuristic
                # refinements into an Event F1 objective.
                continue
        pipeline_completed += 1

        for index in held:
            inner_full[index] = _predict(model, samples[index].windows["features"])
            if samples[index].ads is not None:
                has_events = True
            if onset_head is not None and offset_head is not None:
                inner_heads[index] = (
                    onset_head,
                    offset_head,
                    boundary_thresholds[0],
                    boundary_thresholds[1],
                )

    folds_completed = pipeline_completed if pipeline_required else classifier_completed

    if inner_full and has_events:
        limit = max_false_positives_per_hour
        candidates: list[tuple[float, dict, float]] = []
        # 0.70 以上常是满足 FP/hour 的唯一区域；上界太低会让硬约束永远不可达。
        for candidate in np.linspace(0.05, 0.95, 37):
            metrics = _event_metrics(
                samples,
                sorted(inner_full),
                inner_full,
                thresholds=float(candidate),
                refine=True,
                boundary_heads=inner_heads,
            )
            if not metrics:
                continue
            fp_rate = float(metrics.get("false_positives_per_hour") or 0.0)
            candidates.append((float(candidate), metrics, fp_rate))
        if candidates:
            if limit is not None and limit > 0:
                eligible = [item for item in candidates if item[2] <= limit]
                fp_valid = bool(eligible)
            else:
                eligible = candidates
                fp_valid = None
            if eligible:
                best = max(
                    eligible,
                    key=lambda item: (
                        float(item[1].get("event_f1") or 0.0),
                        -item[2],
                        -abs(item[0] - 0.5),
                    ),
                )
            else:
                # 硬约束不可达时退化为最保守的候选（FP 最低），而不是为了 F1
                # 放低阈值堆误报。
                best = min(
                    candidates,
                    key=lambda item: (
                        item[2],
                        -float(item[1].get("event_f1") or 0.0),
                        -item[0],
                    ),
                )
            return best[0], {
                "source": f"{fold_kind}_event_f1",
                "objective": "event_f1",
                "event": best[1],
                "max_false_positives_per_hour": limit,
                "false_positive_rate_valid": fp_valid,
                "folds_attempted": folds_attempted,
                "folds_completed": folds_completed,
                "folds_skipped": folds_attempted - folds_completed,
                "classifier_folds_completed": classifier_completed,
                "pipeline_folds_completed": pipeline_completed,
                "pipeline_required": pipeline_required,
                "held_out_episodes": len(inner_full),
            }

    threshold, metrics = _choose_classifier_threshold(
        train_indices, samples, X_all, y_all, offsets, seed
    )
    metrics["objective"] = "window_f1_fallback"
    metrics["max_false_positives_per_hour"] = max_false_positives_per_hour
    metrics["false_positive_rate_valid"] = None
    metrics["folds_attempted"] = folds_attempted
    metrics["folds_completed"] = folds_completed
    metrics["folds_skipped"] = folds_attempted - folds_completed
    metrics["classifier_folds_completed"] = classifier_completed
    metrics["pipeline_folds_completed"] = pipeline_completed
    metrics["pipeline_required"] = pipeline_required
    metrics["held_out_episodes"] = len(inner_full)
    return threshold, metrics


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
    folds, fold_kind = _inner_folds(train_indices, samples)
    for fold_index, held in enumerate(folds):
        inner_train = [index for index in train_indices if index not in set(held)]
        train_rows = _window_rows(inner_train, samples, offsets, len(y_all))
        test_rows = _window_rows(held, samples, offsets, len(y_all))
        # Only the training fold needs both classes. A single-class test fold
        # (all content or all ads) is still informative for threshold choice.
        if len(np.unique(y_all[train_rows])) < 2:
            continue
        model = _fit_logistic(*_balance(X_all[train_rows], y_all[train_rows], ratio=4, seed=seed + fold_index))
        probabilities.append(_predict(model, X_all[test_rows]))
        labels.append(y_all[test_rows])
    if not probabilities:
        return 0.5, {"source": "fallback", "reason": "no_valid_inner_fold", "fold_kind": fold_kind}
    probabilities_all = np.concatenate(probabilities)
    labels_all = np.concatenate(labels)
    if len(np.unique(labels_all)) < 2:
        return 0.5, {"source": "fallback", "reason": "single_class_inner_oof", "fold_kind": fold_kind}
    threshold, metrics = best_f1_threshold(labels_all, probabilities_all)
    metrics["auc"] = round(roc_auc(labels_all, probabilities_all), 4)
    metrics["source"] = fold_kind
    metrics["folds"] = len(probabilities)
    metrics["test_windows"] = int(len(labels_all))
    return threshold, metrics


def _boundary_candidates(
    center: float = 0.5,
    coarse: tuple[float, float, float] = (0.2, 0.95, 0.1),
    radius: float = 0.1,
    step: float = 0.01,
) -> tuple[list[float], list[float]]:
    """Return a coarse grid plus a fine grid around ``center``."""
    low, high, stride = coarse
    coarse_values = [round(float(value), 4) for value in np.arange(low, high + 1e-9, stride)]
    fine_low = max(0.01, center - radius)
    fine_high = min(0.99, center + radius)
    fine_values = [
        round(float(value), 4)
        for value in np.arange(fine_low, fine_high + 1e-9, step)
        if min(abs(float(value) - value0) for value0 in coarse_values) > 1e-9
    ]
    return coarse_values, fine_values


def _select_boundary_candidate(
    results: list[dict],
    min_recall_ratio: float = 0.8,
    min_event_recall: float = MIN_BOUNDARY_EVENT_RECALL,
    max_false_positives_per_hour: Optional[float] = None,
) -> dict:
    """Pick a threshold by P95 without trading away event coverage or FPs.

    P95 only sees matched events, so a candidate with one or two easy matches
    can look perfect while missing most ads. Candidates must first retain
    ``min_recall_ratio`` of the best achievable match count and reach the
    absolute ``min_event_recall`` target. When enabled, candidates above
    ``max_false_positives_per_hour`` are excluded before P95 is compared.
    Invalid dimensions are reported separately so a low P95 is never presented
    as a valid calibration by itself.
    """
    if not results:
        return {
            "threshold": 0.5,
            "matched": 0,
            "p95": None,
            "recall_valid": False,
            "false_positive_rate_valid": False,
        }

    def fp_rate(result: dict) -> Optional[float]:
        value = result.get("false_positives_per_hour")
        return float(value) if value is not None else None

    limit = max_false_positives_per_hour
    if limit is not None and limit > 0:
        fp_ok_results = [
            result
            for result in results
            if fp_rate(result) is not None and fp_rate(result) <= limit
        ]
    else:
        fp_ok_results = list(results)

    if not fp_ok_results:
        fallback = min(
            results,
            key=lambda result: (
                fp_rate(result) if fp_rate(result) is not None else float("inf"),
                result.get("p95") if result.get("p95") is not None else float("inf"),
                abs(float(result["threshold"]) - 0.5),
            ),
        )
        return {
            **fallback,
            "recall_valid": False,
            "false_positive_rate_valid": False,
        }

    best_matched = max(int(result.get("matched") or 0) for result in fp_ok_results)
    ground_truth = max(
        int(result.get("ground_truth_events") or 0) for result in fp_ok_results
    )
    if best_matched <= 0:
        chosen = min(
            fp_ok_results, key=lambda result: abs(float(result["threshold"]) - 0.5)
        )
        return {
            **chosen,
            "recall_valid": False,
            "false_positive_rate_valid": True,
        }

    required = max(1, int(math.ceil(min_recall_ratio * best_matched)))
    required_absolute = (
        max(1, int(math.ceil(min_event_recall * ground_truth))) if ground_truth else None
    )
    within_range = [
        result
        for result in fp_ok_results
        if int(result.get("matched") or 0) >= required
    ]
    eligible = [
        result
        for result in within_range
        if required_absolute is not None
        and int(result.get("matched") or 0) >= required_absolute
    ]

    def p95_value(result: dict) -> float:
        p95 = result.get("p95")
        return float(p95) if p95 is not None else float("inf")

    def key(result: dict) -> tuple[float, int, int, float]:
        return (
            p95_value(result),
            -int(result.get("matched") or 0),
            int(result.get("false_positives") or 0),
            abs(float(result["threshold"]) - 0.5),
        )

    if eligible:
        return {
            **min(eligible, key=key),
            "recall_valid": True,
            "false_positive_rate_valid": True,
        }

    fallback = min(
        within_range or fp_ok_results,
        key=lambda result: (
            -int(result.get("matched") or 0),
            p95_value(result),
            int(result.get("false_positives") or 0),
            abs(float(result["threshold"]) - 0.5),
        ),
    )
    return {
        **fallback,
        "recall_valid": False,
        "false_positive_rate_valid": True,
    }


def _choose_boundary_thresholds(
    train_indices: list[int],
    samples: list[EpisodeSample],
    onset_X: list[np.ndarray],
    onset_y: list[np.ndarray],
    offset_X: list[np.ndarray],
    offset_y: list[np.ndarray],
    episode_indices: list[int],
    X_all: np.ndarray,
    y_all: np.ndarray,
    offsets: dict[int, tuple[int, int]],
    seed: int,
    provisional_threshold: float = 0.5,
    min_recall_ratio: float = 0.8,
    min_event_recall: float = MIN_BOUNDARY_EVENT_RECALL,
    max_false_positives_per_hour: Optional[float] = MAX_BOUNDARY_FALSE_POSITIVES_PER_HOUR,
) -> tuple[float, float, dict, dict]:
    """Calibrate onset/offset thresholds against final refined boundaries.

    Candidate spans are decoded with ``provisional_threshold`` so callers can
    alternate classifier and boundary calibration. Both thresholds use a
    coarse-to-fine search, and P95 is optimized only among candidates that
    retain ``min_recall_ratio`` of the best achievable event matches. Episodes
    where decoding finds no events stay in the denominator instead of being
    silently dropped.
    """
    folds, fold_kind = _inner_folds(train_indices, samples)
    stored: list[dict] = []
    # Count every held-out ad event, including folds whose boundary heads could
    # not be trained. Otherwise skipped folds vanish from recall and a
    # calibration that decoded nothing would look perfect.
    ground_truth_events = sum(
        len(samples[index].ads or [])
        for held in folds
        for index in held
        if samples[index].ads
    )
    folds_attempted = len(folds)
    classifier_completed = 0
    onset_completed = 0
    offset_completed = 0
    pipeline_completed = 0
    positive_episodes = 0
    negative_control_episodes = 0
    for fold_index, held in enumerate(folds):
        inner_train = [index for index in train_indices if index not in set(held)]
        train_rows = _window_rows(inner_train, samples, offsets, len(y_all))
        if len(np.unique(y_all[train_rows])) < 2:
            continue
        classifier_completed += 1
        classifier = _fit_logistic(
            *_balance(X_all[train_rows], y_all[train_rows], ratio=4, seed=seed + fold_index)
        )
        inner_set = set(inner_train)
        onset_train_X, onset_train_y = _gather_boundary(onset_X, onset_y, episode_indices, inner_set)
        offset_train_X, offset_train_y = _gather_boundary(offset_X, offset_y, episode_indices, inner_set)
        if len(onset_train_y) and len(np.unique(onset_train_y)) >= 2:
            onset_completed += 1
        if len(offset_train_y) and len(np.unique(offset_train_y)) >= 2:
            offset_completed += 1
        if (
            len(onset_train_y) == 0
            or len(offset_train_y) == 0
            or len(np.unique(onset_train_y)) < 2
            or len(np.unique(offset_train_y)) < 2
        ):
            continue
        onset_head = _fit_logistic(
            *_balance(onset_train_X, onset_train_y, ratio=6, seed=seed + 11 + fold_index)
        )
        offset_head = _fit_logistic(
            *_balance(offset_train_X, offset_train_y, ratio=6, seed=seed + 21 + fold_index)
        )
        contributed = False
        for index in held:
            sample = samples[index]
            if sample.ads is None:
                continue
            if sample.ads:
                positive_episodes += 1
            else:
                negative_control_episodes += 1
            probabilities = _predict(classifier, sample.windows["features"])
            enter, exit_value = resolve_thresholds(provisional_threshold)
            events = decode_windows(
                sample.windows["times"],
                probabilities,
                hop_s=float(sample.windows["hop_s"]),
                span_s=float(sample.windows["span_s"]),
                min_ad_s=10.0,
                enter_threshold=enter,
                exit_threshold=exit_value,
            )
            features = sample.boundary_features.astype(np.float32)
            ground_truth = [
                {"start_s": float(ad["start_s"]), "end_s": float(ad["end_s"])}
                for ad in sample.ads
            ]
            stored.append(
                {
                    "events": events,
                    "onset_scores": _predict(onset_head, features) if events else None,
                    "offset_scores": _predict(offset_head, features) if events else None,
                    "times": np.arange(len(features)) * sample.change_grid_s,
                    "gt": ground_truth,
                    "duration_s": float(sample.duration_s),
                }
            )
            contributed = True
        if contributed:
            pipeline_completed += 1

    coverage = {
        "fold_kind": fold_kind,
        "folds_attempted": folds_attempted,
        "folds_completed": pipeline_completed,
        "folds_skipped": folds_attempted - pipeline_completed,
        "classifier_folds_completed": classifier_completed,
        "onset_folds_completed": onset_completed,
        "offset_folds_completed": offset_completed,
        "pipeline_folds_completed": pipeline_completed,
        "episodes": len(stored),
        "positive_boundary_episodes": positive_episodes,
        "negative_control_episodes": negative_control_episodes,
        "ground_truth_events": ground_truth_events,
        "provisional_classifier_threshold": round(float(provisional_threshold), 4),
    }
    total_duration = sum(item["duration_s"] for item in stored)

    def evaluate(
        onset_threshold: float, offset_threshold: float
    ) -> tuple[list[float], list[float], int, int]:
        start_errors: list[float] = []
        end_errors: list[float] = []
        matched = 0
        false_positives = 0
        for item in stored:
            if item["events"] and item["onset_scores"] is not None:
                change = {
                    "times": item["times"],
                    "onset_scores": item["onset_scores"],
                    "offset_scores": item["offset_scores"],
                    "scores": np.maximum(item["onset_scores"], item["offset_scores"]),
                    "kind": "learned_onset_offset",
                    "onset_threshold": onset_threshold,
                    "offset_threshold": offset_threshold,
                }
                # 与部署端一致：精修不能把事件缩到比最短广告时长还短。
                refined = refine_events(
                    item["events"], change, search_s=2.5, min_duration=MIN_AD_S
                )
            else:
                refined = []
            metrics = evaluate_events(refined, item["gt"])
            matched += len(metrics["matches"])
            false_positives += int(metrics["false_positives"])
            for match in metrics["matches"]:
                start_errors.append(float(match["start_error"]))
                end_errors.append(float(match["end_error"]))
        return start_errors, end_errors, matched, false_positives

    def measure(kind: str, value: float, other: float) -> dict:
        if kind == "onset":
            start_errors, _, matched, false_positives = evaluate(value, other)
            errors = start_errors
        else:
            _, end_errors, matched, false_positives = evaluate(other, value)
            errors = end_errors
        fp_rate = (
            round(false_positives / (total_duration / 3600.0), 3)
            if total_duration > 0
            else None
        )
        return {
            "threshold": round(float(value), 4),
            "matched": matched,
            "ground_truth_events": ground_truth_events,
            "false_positives": false_positives,
            "false_positives_per_hour": fp_rate,
            "p95": float(np.percentile(errors, 95)) if errors else None,
        }

    def search(kind: str, other: float) -> tuple[float, dict]:
        coarse, _ = _boundary_candidates(center=0.5)
        results = [measure(kind, value, other) for value in coarse]
        best = _select_boundary_candidate(
            results,
            min_recall_ratio,
            min_event_recall,
            max_false_positives_per_hour,
        )
        _, fine = _boundary_candidates(center=float(best["threshold"]))
        results.extend(measure(kind, value, other) for value in fine)
        best = _select_boundary_candidate(
            results,
            min_recall_ratio,
            min_event_recall,
            max_false_positives_per_hour,
        )
        return float(best["threshold"]), best

    # Coordinate descent: onset and offset both move the decoded boundary, so
    # alternate them twice instead of optimizing either against a stale value.
    onset_threshold = offset_threshold = 0.5
    onset_result: dict = {
        "threshold": 0.5,
        "matched": 0,
        "p95": None,
        "recall_valid": False,
    }
    offset_result: dict = dict(onset_result)
    for _ in range(2):
        onset_threshold, onset_result = search("onset", offset_threshold)
        offset_threshold, offset_result = search("offset", onset_threshold)

    # Coordinate descent evaluates onset with the previous offset and offset
    # with the new onset. Re-run the final pair once so the reported P95,
    # recall and FP/hour all describe the thresholds that are actually
    # returned, not two independently measured halves.
    final_start_errors, final_end_errors, final_matched, final_false_positives = evaluate(
        onset_threshold, offset_threshold
    )
    final_fp_rate = (
        round(final_false_positives / (total_duration / 3600.0), 3)
        if total_duration > 0
        else None
    )
    final_recall = (
        float(final_matched) / ground_truth_events if ground_truth_events else None
    )
    final_recall_valid = bool(
        ground_truth_events
        and final_recall is not None
        and final_recall >= min_event_recall
    )
    final_fp_valid = bool(
        max_false_positives_per_hour is None
        or max_false_positives_per_hour <= 0
        or (final_fp_rate is not None and final_fp_rate <= max_false_positives_per_hour)
    )
    final_valid = final_recall_valid and final_fp_valid
    final_joint = {
        "thresholds": {
            "onset": round(float(onset_threshold), 4),
            "offset": round(float(offset_threshold), 4),
        },
        "matched_events": int(final_matched),
        "false_positives": int(final_false_positives),
        "false_positives_per_hour": final_fp_rate,
        "event_recall": round(final_recall, 4) if final_recall is not None else None,
        "recall_valid": final_recall_valid,
        "false_positive_rate_valid": final_fp_valid,
        "boundary_calibration_valid": final_valid,
        "boundary_start_p95_s": (
            round(float(np.percentile(final_start_errors, 95)), 3)
            if final_start_errors
            else None
        ),
        "boundary_end_p95_s": (
            round(float(np.percentile(final_end_errors, 95)), 3)
            if final_end_errors
            else None
        ),
    }
    common = {
        "source": f"{fold_kind}_boundary_error",
        "objective": "boundary_p95",
        "final_joint": True,
        "final_joint_metrics": final_joint,
        "boundary_calibration_valid": final_valid,
        "recall_valid": final_recall_valid,
        "false_positive_rate_valid": final_fp_valid,
        "matched_events": int(final_matched),
        "false_positives": int(final_false_positives),
        "false_positives_per_hour": final_fp_rate,
        "max_false_positives_per_hour": max_false_positives_per_hour,
        "ground_truth_events": ground_truth_events,
        "event_recall": round(final_recall, 4) if final_recall is not None else None,
        "min_recall_ratio": min_recall_ratio,
        "min_event_recall": min_event_recall,
        "provisional_classifier_threshold": round(float(provisional_threshold), 4),
        "coverage": coverage,
    }
    onset_metrics = {
        **common,
        "boundary_start_p95_s": final_joint["boundary_start_p95_s"],
    }
    offset_metrics = {
        **common,
        "boundary_end_p95_s": final_joint["boundary_end_p95_s"],
    }
    return onset_threshold, offset_threshold, onset_metrics, offset_metrics


def _calibrate_thresholds(
    train_indices: list[int],
    samples: list[EpisodeSample],
    X_all: np.ndarray,
    y_all: np.ndarray,
    offsets: dict[int, tuple[int, int]],
    boundary_onset_X: list[np.ndarray],
    boundary_onset_y: list[np.ndarray],
    boundary_offset_X: list[np.ndarray],
    boundary_offset_y: list[np.ndarray],
    boundary_episode_indices: list[int],
    event_seed: int,
    boundary_seed: int,
    min_recall_ratio: float = 0.8,
    min_event_recall: float = MIN_BOUNDARY_EVENT_RECALL,
    max_false_positives_per_hour: Optional[float] = MAX_BOUNDARY_FALSE_POSITIVES_PER_HOUR,
    max_event_false_positives_per_hour: Optional[float] = MAX_EVENT_FALSE_POSITIVES_PER_HOUR,
) -> tuple[float, float, float, dict, dict, dict]:
    """Coordinate descent over classifier, onset and offset thresholds.

    Each round selects the classifier threshold using the current boundary
    thresholds, then recalibrates onset/offset on spans decoded by that
    classifier threshold. The loop stops when every threshold moves by at most
    ``THRESHOLD_TOLERANCE`` or after ``MAX_CALIBRATION_ROUNDS``. Without
    boundary labels there is nothing to alternate and one round is recorded.
    """
    onset_threshold = offset_threshold = 0.5
    event_threshold = 0.5
    event_metrics: dict = {}
    onset_metrics: dict = {}
    offset_metrics: dict = {}
    initial_event_threshold: float | None = None
    history: list[dict] = []
    converged = False
    rounds = 0
    for round_index in range(MAX_CALIBRATION_ROUNDS):
        previous_event = event_threshold
        previous_onset = onset_threshold
        previous_offset = offset_threshold
        event_threshold, event_metrics = _choose_event_threshold(
            train_indices,
            samples,
            X_all,
            y_all,
            offsets,
            seed=event_seed,
            boundary_onset_X=boundary_onset_X,
            boundary_onset_y=boundary_onset_y,
            boundary_offset_X=boundary_offset_X,
            boundary_offset_y=boundary_offset_y,
            boundary_episode_indices=boundary_episode_indices,
            boundary_thresholds=(onset_threshold, offset_threshold),
            max_false_positives_per_hour=max_event_false_positives_per_hour,
        )
        if initial_event_threshold is None:
            initial_event_threshold = event_threshold
        rounds = round_index + 1
        if not boundary_episode_indices:
            history.append(
                {
                    "round": rounds,
                    "classifier_threshold": round(float(event_threshold), 4),
                    "onset_threshold": round(float(onset_threshold), 4),
                    "offset_threshold": round(float(offset_threshold), 4),
                }
            )
            converged = True
            break

        onset_threshold, offset_threshold, onset_metrics, offset_metrics = (
            _choose_boundary_thresholds(
                train_indices,
                samples,
                boundary_onset_X,
                boundary_onset_y,
                boundary_offset_X,
                boundary_offset_y,
                boundary_episode_indices,
                X_all,
                y_all,
                offsets,
                seed=boundary_seed,
                provisional_threshold=event_threshold,
                min_recall_ratio=min_recall_ratio,
                min_event_recall=min_event_recall,
                max_false_positives_per_hour=max_false_positives_per_hour,
            )
        )
        classifier_delta = abs(event_threshold - previous_event)
        onset_delta = abs(onset_threshold - previous_onset)
        offset_delta = abs(offset_threshold - previous_offset)
        history.append(
            {
                "round": rounds,
                "classifier_threshold": round(float(event_threshold), 4),
                "onset_threshold": round(float(onset_threshold), 4),
                "offset_threshold": round(float(offset_threshold), 4),
                "classifier_delta": round(float(classifier_delta), 4),
                "onset_delta": round(float(onset_delta), 4),
                "offset_delta": round(float(offset_delta), 4),
                "boundary_calibration_valid": bool(
                    onset_metrics.get("boundary_calibration_valid")
                    and offset_metrics.get("boundary_calibration_valid")
                ),
            }
        )
        if max(classifier_delta, onset_delta, offset_delta) <= THRESHOLD_TOLERANCE:
            converged = True
            break

    event_metrics["calibration"] = {
        "rounds": rounds,
        "converged": converged,
        "final_recalibrated": False,
        "initial_classifier_threshold": round(float(initial_event_threshold or 0.5), 4),
        "final_classifier_threshold": round(float(event_threshold), 4),
        "max_rounds": MAX_CALIBRATION_ROUNDS,
        "tolerance": THRESHOLD_TOLERANCE,
        "history": history,
    }

    if boundary_episode_indices and not converged:
        # The last loop round selected the classifier threshold before the
        # final boundary update. Re-select it against the returned boundary
        # thresholds so deployment does not receive a mixed parameter set.
        event_threshold, event_metrics = _choose_event_threshold(
            train_indices,
            samples,
            X_all,
            y_all,
            offsets,
            seed=event_seed,
            boundary_onset_X=boundary_onset_X,
            boundary_onset_y=boundary_onset_y,
            boundary_offset_X=boundary_offset_X,
            boundary_offset_y=boundary_offset_y,
            boundary_episode_indices=boundary_episode_indices,
            boundary_thresholds=(onset_threshold, offset_threshold),
            max_false_positives_per_hour=max_event_false_positives_per_hour,
        )
        event_metrics["calibration"] = {
            "rounds": rounds,
            "converged": False,
            "final_recalibrated": True,
            "initial_classifier_threshold": round(
                float(initial_event_threshold or 0.5), 4
            ),
            "final_classifier_threshold": round(float(event_threshold), 4),
            "max_rounds": MAX_CALIBRATION_ROUNDS,
            "tolerance": THRESHOLD_TOLERANCE,
            "history": history,
        }

    calibration = event_metrics["calibration"]
    event_metrics["final_calibration_metrics"] = _final_joint_metrics(
        train_indices,
        samples,
        X_all,
        y_all,
        offsets,
        boundary_onset_X,
        boundary_onset_y,
        boundary_offset_X,
        boundary_offset_y,
        boundary_episode_indices,
        seed=event_seed,
        classifier_threshold=event_threshold,
        onset_threshold=onset_threshold,
        offset_threshold=offset_threshold,
        max_false_positives_per_hour=max_event_false_positives_per_hour,
    )
    calibration["final_thresholds"] = event_metrics["final_calibration_metrics"].get(
        "final_thresholds"
    )
    return event_threshold, onset_threshold, offset_threshold, event_metrics, onset_metrics, offset_metrics


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


def _aggregate_fold_window_metrics(
    y_all: np.ndarray,
    oof: np.ndarray,
    folds: list[list[int]],
    fold_thresholds: dict[int, float],
    samples: list[EpisodeSample],
    offsets: dict[int, tuple[int, int]],
) -> dict:
    tp = fp = fn = 0
    fold_aucs: list[float | None] = []
    for held in folds:
        if not held:
            continue
        rows = _window_rows(held, samples, offsets, len(y_all))
        threshold = fold_thresholds.get(held[0], 0.5)
        a, b, c = threshold_counts(y_all[rows], oof[rows], threshold)
        tp += a
        fp += b
        fn += c
        if len(np.unique(y_all[rows])) >= 2:
            fold_aucs.append(roc_auc(y_all[rows], oof[rows]))
        else:
            fold_aucs.append(None)
    metrics = metrics_from_counts(tp, fp, fn)
    metrics["threshold_source"] = "per_outer_fold"
    metrics["folds"] = len(folds)
    metrics["auc_fold"] = [round(value, 4) if value is not None else None for value in fold_aucs]
    valid = [value for value in fold_aucs if value is not None]
    metrics["auc_macro"] = round(float(np.mean(valid)), 4) if valid else None
    return metrics


def _aggregate_head_groups(
    groups: list[tuple[np.ndarray, np.ndarray, float, int]]
) -> dict:
    if not groups:
        return {}
    tp = fp = fn = 0
    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    by_fold: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for probabilities, labels, threshold, fold_index in groups:
        a, b, c = threshold_counts(labels, probabilities, threshold)
        tp += a
        fp += b
        fn += c
        probability_parts.append(probabilities)
        label_parts.append(labels)
        by_fold.setdefault(fold_index, []).append((probabilities, labels))
    metrics = metrics_from_counts(tp, fp, fn)
    metrics["threshold_source"] = "per_outer_fold"
    metrics["auc_pooled"] = round(
        roc_auc(np.concatenate(label_parts), np.concatenate(probability_parts)), 4
    )
    fold_aucs: list[float] = []
    for items in by_fold.values():
        labels_all = np.concatenate([labels for _, labels in items])
        probabilities_all = np.concatenate([probabilities for probabilities, _ in items])
        if len(np.unique(labels_all)) >= 2:
            fold_aucs.append(roc_auc(labels_all, probabilities_all))
    metrics["auc_fold"] = [round(value, 4) for value in fold_aucs]
    metrics["auc_macro"] = round(float(np.mean(fold_aucs)), 4) if fold_aucs else None
    metrics["auc"] = metrics["auc_macro"]
    metrics["auc_pooled_is_diagnostic"] = True
    return metrics


def _event_metrics(
    samples: list[EpisodeSample],
    indices: list[int],
    probabilities: dict[int, np.ndarray],
    thresholds: dict[int, float] | float,
    refine: bool = True,
    boundary_heads: Optional[dict[int, tuple[dict, dict, float, float]]] = None,
    require_boundary: bool = False,
) -> dict:
    labeled = [index for index in indices if samples[index].ads is not None]
    if not labeled:
        return {}
    total_duration = 0.0
    matched = false_positives = false_negatives = 0
    predicted_total = ground_truth_total = 0
    boundary_errors: list[float] = []
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
            min_ad_s=10.0,
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
            elif require_boundary and events:
                # Refinement runs even when ground truth is empty, so an
                # unlabelled episode that produces a false positive must use
                # the same learned boundary path as deployment.
                raise ValueError(
                    f"第 {index} 集存在待精修事件但缺少学习型边界头；"
                    "需要边界模型时不允许回退启发式精修"
                )
            else:
                change = {
                    "times": np.arange(len(sample.change_scores)) * sample.change_grid_s,
                    "scores": sample.change_scores,
                    "kind": "heuristic",
                }
            refined = refine_events(
                events, change, search_s=2.5, min_duration=MIN_AD_S
            )
        else:
            refined = [
                {
                    "start_s": event.start,
                    "end_s": event.end,
                }
                for event in events
            ]
        # Timestamps restart at zero for every episode, so matching must be
        # scoped to one episode. Pooling spans across episodes can pair a false
        # positive in one show with a miss in another and report a perfect
        # match.
        truth = [
            {"start_s": float(ad["start_s"]), "end_s": float(ad["end_s"])}
            for ad in sample.ads
        ]
        episode = evaluate_events(refined, truth, duration_s=sample.duration_s)
        matched += int(episode["matched"])
        false_positives += int(episode["false_positives"])
        false_negatives += int(episode["false_negatives"])
        predicted_total += int(episode["predicted_events"])
        ground_truth_total += int(episode["ground_truth_events"])
        for match in episode["matches"]:
            boundary_errors.extend(
                (float(match["start_error"]), float(match["end_error"]))
            )
        total_duration += sample.duration_s

    precision = matched / predicted_total if predicted_total else None
    recall = matched / ground_truth_total if ground_truth_total else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "event_precision": round(precision, 4) if precision is not None else None,
        "event_recall": round(recall, 4) if recall is not None else None,
        "event_f1": round(f1, 4) if f1 is not None else None,
        "boundary_p95_s": (
            round(float(np.percentile(boundary_errors, 95)), 3) if boundary_errors else None
        ),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_positives_per_hour": (
            round(false_positives / (total_duration / 3600.0), 3)
            if total_duration > 0
            else None
        ),
        "ground_truth_events": ground_truth_total,
        "predicted_events": predicted_total,
    }


def _final_joint_metrics(
    train_indices: list[int],
    samples: list[EpisodeSample],
    X_all: np.ndarray,
    y_all: np.ndarray,
    offsets: dict[int, tuple[int, int]],
    boundary_onset_X: list[np.ndarray],
    boundary_onset_y: list[np.ndarray],
    boundary_offset_X: list[np.ndarray],
    boundary_offset_y: list[np.ndarray],
    boundary_episode_indices: list[int],
    seed: int,
    classifier_threshold: float,
    onset_threshold: float,
    offset_threshold: float,
    max_false_positives_per_hour: Optional[float] = MAX_EVENT_FALSE_POSITIVES_PER_HOUR,
) -> dict:
    """Re-run the full inner OOF pipeline with the final threshold triplet.

    The thresholds are fixed before decoding; this function does not search.
    It exists so the reported precision/recall/P95/FP-hour and coverage all
    describe the exact combination that will be deployed.
    """
    folds, fold_kind = _inner_folds(train_indices, samples)
    probabilities: dict[int, np.ndarray] = {}
    heads: dict[int, tuple[dict, dict, float, float]] = {}
    folds_attempted = len(folds)
    classifier_completed = 0
    pipeline_completed = 0
    pipeline_required = bool(boundary_episode_indices)
    for fold_index, held in enumerate(folds):
        inner_train = [index for index in train_indices if index not in set(held)]
        train_rows = _window_rows(inner_train, samples, offsets, len(y_all))
        if len(np.unique(y_all[train_rows])) < 2:
            continue
        model = _fit_logistic(
            *_balance(X_all[train_rows], y_all[train_rows], ratio=4, seed=seed + fold_index)
        )
        classifier_completed += 1
        if pipeline_required:
            inner_set = set(inner_train)
            onset_X, onset_y = _gather_boundary(
                boundary_onset_X, boundary_onset_y, boundary_episode_indices, inner_set
            )
            offset_X, offset_y = _gather_boundary(
                boundary_offset_X, boundary_offset_y, boundary_episode_indices, inner_set
            )
            onset_head = offset_head = None
            if len(onset_y) and len(np.unique(onset_y)) >= 2:
                onset_head = _fit_logistic(
                    *_balance(onset_X, onset_y, ratio=6, seed=seed + 11 + fold_index)
                )
            if len(offset_y) and len(np.unique(offset_y)) >= 2:
                offset_head = _fit_logistic(
                    *_balance(offset_X, offset_y, ratio=6, seed=seed + 21 + fold_index)
                )
            if onset_head is None or offset_head is None:
                continue
            for index in held:
                heads[index] = (
                    onset_head,
                    offset_head,
                    onset_threshold,
                    offset_threshold,
                )
        pipeline_completed += 1
        for index in held:
            probabilities[index] = _predict(model, samples[index].windows["features"])

    metrics = _event_metrics(
        samples,
        sorted(probabilities),
        probabilities,
        thresholds=classifier_threshold,
        refine=True,
        boundary_heads=heads,
        require_boundary=pipeline_required,
    )
    fp_rate = metrics.get("false_positives_per_hour")
    fp_valid = (
        None
        if max_false_positives_per_hour is None or max_false_positives_per_hour <= 0
        else bool(fp_rate is not None and fp_rate <= max_false_positives_per_hour)
    )
    metrics.update(
        {
            "source": f"{fold_kind}_final_joint",
            "objective": "final_joint",
            "final_thresholds": {
                "classifier": round(float(classifier_threshold), 4),
                "onset": round(float(onset_threshold), 4),
                "offset": round(float(offset_threshold), 4),
            },
            "folds_attempted": folds_attempted,
            "classifier_folds_completed": classifier_completed,
            "pipeline_folds_completed": pipeline_completed,
            "pipeline_required": pipeline_required,
            "max_false_positives_per_hour": max_false_positives_per_hour,
            "false_positive_rate_valid": fp_valid,
        }
    )
    return metrics


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
    parser.add_argument(
        "--strict-audio-id",
        action="store_true",
        help="即使 source_sha256 一致，也要求 PCM audio_id 完全一致（跨解码器会失败）",
    )
    parser.add_argument(
        "--min-boundary-event-recall",
        type=float,
        default=MIN_BOUNDARY_EVENT_RECALL,
        help="边界阈值校准要求的最低事件召回率，未达到时标记校准无效",
    )
    parser.add_argument(
        "--max-boundary-fp-per-hour",
        type=float,
        default=MAX_BOUNDARY_FALSE_POSITIVES_PER_HOUR,
        help="边界阈值候选允许的最大每小时误报数；<=0 表示不约束",
    )
    parser.add_argument(
        "--max-event-fp-per-hour",
        type=float,
        default=MAX_EVENT_FALSE_POSITIVES_PER_HOUR,
        help="分类阈值和最终流水线允许的最大每小时误报数；<=0 表示不约束",
    )
    parser.add_argument(
        "--allow-cross-feed-duplicates",
        action="store_true",
        help="允许同一 audio_id/source_sha256 出现在多个 feed（默认拒绝，防止数据泄漏）",
    )
    parser.add_argument(
        "--allow-invalid-annotations",
        action="store_true",
        help="广告区间结构校验失败时仍继续（默认中止）",
    )
    parser.add_argument(
        "--require-ad-type",
        action="store_true",
        help="要求每个广告区间显式标注 type=third_party_inserted",
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

    duplicates = _duplicate_identity_problems(entries)
    if duplicates:
        for problem in duplicates:
            print(f"[data] {problem}", flush=True)
        if not args.allow_cross_feed_duplicates:
            print(
                "[data] 中止训练：同一音频/字节出现在多个 feed，或 feed/episode 标识重复；"
                "确认无误可用 --allow-cross-feed-duplicates",
                flush=True,
            )
            return 1

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
        if not weak_mode and (
            not str(entry.get("feed") or "").strip()
            or not str(entry.get("episode") or "").strip()
        ):
            print(f"[data] {name}: feed 或 episode 标识为空", flush=True)
            if not args.allow_invalid_annotations:
                return 1
        print(f"[extract] {feed} / {name}", flush=True)
        mono, forensic = extract_features(audio, proxy=args.proxy)
        identity, identity_warnings = _identity_problems(
            entry,
            audio,
            mono,
            args.allow_unverified_audio,
            args.strict_audio_id,
        )
        for warning in identity_warnings:
            print(f"[identity-warn] {name}: {warning}", flush=True)
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
                {
                    **item,
                    "start_s": float(item["start_s"]),
                    "end_s": float(item["end_s"]),
                }
                for item in (entry.get("ads") or [])
            ]
            annotation = _annotation_problems(
                ads, duration, require_ad_type=args.require_ad_type
            )
            if annotation:
                for problem in annotation:
                    print(f"[data] {name}: {problem}", flush=True)
                if not args.allow_invalid_annotations:
                    print("[data] 中止训练；确认标注后可修正，或用 --allow-invalid-annotations 覆盖")
                    return 1
            # 检测器无法输出短于 MIN_AD_S 的事件；训练标签必须用同一定义，
            # 否则短广告会把广告窗口教成负样本、同时又算进评测分母。
            short = [item for item in ads if item["end_s"] - item["start_s"] < MIN_AD_S]
            if short:
                durations = ", ".join(
                    f"{item['end_s'] - item['start_s']:.2f}s" for item in short[:5]
                )
                print(
                    f"[data] {name}: 排除 {len(short)} 段短于 {MIN_AD_S:g}s 的广告（{durations}）",
                    flush=True,
                )
            ads = [item for item in ads if item["end_s"] - item["start_s"] >= MIN_AD_S]
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
        if not weak_mode and not args.skip_boundary and ads is not None:
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
    boundary_onset_eval: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    boundary_offset_eval: list[tuple[np.ndarray, np.ndarray, float, int]] = []
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

        allowed = set(outer_train)
        if boundary_episode_indices:
            onset_X, onset_y = _gather_boundary(
                boundary_onset_X, boundary_onset_y, boundary_episode_indices, allowed
            )
            offset_X, offset_y = _gather_boundary(
                boundary_offset_X, boundary_offset_y, boundary_episode_indices, allowed
            )
            if len(onset_y) == 0 or len(offset_y) == 0:
                print(f"[error] 第 {fold_index + 1} 折训练集没有可用的边界标注")
                return 1
            if len(np.unique(onset_y)) < 2 or len(np.unique(offset_y)) < 2:
                print(f"[error] 第 {fold_index + 1} 折边界训练集只有一个类别")
                return 1

        (
            fold_threshold,
            onset_threshold,
            offset_threshold,
            threshold_metrics_info,
            boundary_onset_selection,
            boundary_offset_selection,
        ) = _calibrate_thresholds(
            outer_train,
            samples,
            X_all,
            y_all,
            per_episode_offset,
            boundary_onset_X,
            boundary_onset_y,
            boundary_offset_X,
            boundary_offset_y,
            boundary_episode_indices,
            event_seed=fold_index + 1,
            boundary_seed=fold_index + 31,
            min_event_recall=args.min_boundary_event_recall,
            max_false_positives_per_hour=args.max_boundary_fp_per_hour,
            max_event_false_positives_per_hour=args.max_event_fp_per_hour,
        )
        if not boundary_episode_indices:
            onset_X = onset_y = offset_X = offset_y = None
        for index in held:
            fold_thresholds[index] = fold_threshold

        model = _fit_logistic(*_balance(X_all[train_rows], train_y, ratio=4, seed=fold_index + 1))
        oof[held_rows] = _predict(model, X_all[held_rows])
        for index in held:
            full_oof[index] = _predict(model, samples[index].windows["features"])

        onset_head = offset_head = None
        if boundary_episode_indices:
            onset_head = _fit_logistic(*_balance(onset_X, onset_y, ratio=6, seed=fold_index + 11))
            offset_head = _fit_logistic(*_balance(offset_X, offset_y, ratio=6, seed=fold_index + 21))

        if onset_head is not None and offset_head is not None:
            # Bind learned heads to every held episode, including confirmed
            # no-ad episodes that may still produce false positives. Deployment
            # does not know the labels, so evaluation must not take a
            # different refinement path for unlabelled audio.
            for index in held:
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
                    boundary_onset_eval.append(
                        (
                            _predict(onset_head, boundary_onset_X[entry]),
                            boundary_onset_y[entry],
                            onset_threshold,
                            fold_index,
                        )
                    )
                if len(boundary_offset_y[entry]):
                    boundary_offset_eval.append(
                        (
                            _predict(offset_head, boundary_offset_X[entry]),
                            boundary_offset_y[entry],
                            offset_threshold,
                            fold_index,
                        )
                    )

        events = _event_metrics(
            samples,
            held,
            full_oof,
            thresholds=fold_thresholds,
            refine=True,
            boundary_heads=fold_boundary_heads,
            require_boundary=bool(boundary_episode_indices),
        )
        fold_reports.append(
            {
                "fold": fold_index,
                "held": held,
                "threshold": round(fold_threshold, 4),
                "threshold_metrics": threshold_metrics_info,
                "boundary_threshold_selection": {
                    "onset": boundary_onset_selection,
                    "offset": boundary_offset_selection,
                },
                "window_auc": (
                    round(roc_auc(y_all[held_rows], oof[held_rows]), 4)
                    if len(np.unique(y_all[held_rows])) >= 2
                    else None
                ),
                "events": events,
            }
        )
        print(
            f"[fold {fold_index + 1}/{len(folds)}] 留出 {len(held)} 集, "
            f"threshold={fold_threshold:.3f}, events={events}",
            flush=True,
        )

    # Cross-validation metrics use each outer fold's own threshold. The
    # deployment threshold is calibrated separately on the full training set
    # and must not be used to recompute CV metrics.
    window_metrics = _aggregate_fold_window_metrics(
        y_all, oof, folds, fold_thresholds, samples, per_episode_offset
    )
    window_metrics["auc_pooled"] = round(roc_auc(y_all, oof), 4)
    window_metrics["auc"] = window_metrics["auc_macro"]
    window_metrics["auc_pooled_is_diagnostic"] = True

    cv_quality = _cv_quality(
        groups,
        fold_reports,
        boundary_expected=bool(boundary_episode_indices),
    )
    (
        deployment_threshold,
        deployment_onset_threshold,
        deployment_offset_threshold,
        deployment_threshold_metrics,
        deployment_onset_selection,
        deployment_offset_selection,
    ) = _calibrate_thresholds(
        all_indices,
        samples,
        X_all,
        y_all,
        per_episode_offset,
        boundary_onset_X,
        boundary_onset_y,
        boundary_offset_X,
        boundary_offset_y,
        boundary_episode_indices,
        event_seed=101,
        boundary_seed=909,
        min_event_recall=args.min_boundary_event_recall,
        max_false_positives_per_hour=args.max_boundary_fp_per_hour,
        max_event_false_positives_per_hour=args.max_event_fp_per_hour,
    )
    deployment_calibration = {
        "threshold": round(deployment_threshold, 4),
        "onset_threshold": round(deployment_onset_threshold, 4),
        "offset_threshold": round(deployment_offset_threshold, 4),
        "window": threshold_metrics(y_all, oof, deployment_threshold),
        "threshold_selection": deployment_threshold_metrics,
    }
    deployment_calibration_meta = deployment_threshold_metrics.get("calibration") or {}
    deployment_calibration.update(
        _calibration_validity(
            deployment_threshold_metrics,
            deployment_threshold_metrics.get("final_calibration_metrics") or {},
            deployment_onset_selection,
            deployment_offset_selection,
            boundary_expected=bool(boundary_episode_indices),
            converged=bool(deployment_calibration_meta.get("converged")),
            cv_quality_level=cv_quality["level"],
            max_false_positives_per_hour=args.max_event_fp_per_hour,
        )
    )
    deployment_calibration["final_recalibrated"] = bool(
        deployment_calibration_meta.get("final_recalibrated")
    )
    # Calibration acceptance is a process check, not a product-level
    # performance certification. Precision/recall/P95 must come from an
    # independent blind test, never from thresholds selected on inner folds.
    deployment_calibration["performance_targets_met"] = None
    deployment_calibration["performance_targets_source"] = (
        "independent_blind_test_required"
    )
    if not deployment_calibration["valid"]:
        print(
            "[warning] 部署阈值组合未通过校准验收；模型仍会写出，"
            "但不能报告为已完成校准验收",
            flush=True,
        )
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
    training_identity = {
        "audio_ids": sorted(
            {
                str(entry.get("audio_id") or "")
                for entry in entries
                if entry.get("audio_id")
            }
        ),
        "source_sha256": sorted(
            {
                str(entry.get("source_sha256") or "")
                for entry in entries
                if entry.get("source_sha256")
            }
        ),
        "feeds": sorted({str(entry.get("feed") or "unknown") for entry in entries}),
        "feed_keys": sorted(
            {
                canonical_feed_key(
                    str(entry.get("feed") or ""),
                    str(entry.get("feed_url") or ""),
                )
                for entry in entries
            }
        ),
    }

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
        "training_identity": training_identity,
        "metrics": {
            "episodes": len(samples),
            "feeds": len(set(groups)),
            "windows": int(len(y_all)),
            "positives": int(y_all.sum()),
            "negatives": int(len(y_all) - y_all.sum()),
            "window": window_metrics,
            "event": event_metrics,
            "deployment_calibration": deployment_calibration,
            "cv_quality": cv_quality,
            "folds": fold_reports,
            "supervision": "weak_llm_candidates" if weak_mode else "full_annotations",
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[model] {args.model_out}")
    print(json.dumps({key: payload["metrics"][key] for key in ("window", "event")}, ensure_ascii=False))

    boundary_written = False
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
            onset_threshold = deployment_onset_threshold
            offset_threshold = deployment_offset_threshold
            onset_selection = deployment_onset_selection
            offset_selection = deployment_offset_selection
            onset_metrics = _aggregate_head_groups(boundary_onset_eval)
            offset_metrics = _aggregate_head_groups(boundary_offset_eval)
            onset_metrics["threshold_selection"] = onset_selection
            offset_metrics["threshold_selection"] = offset_selection

            deployment_calibration["onset_threshold"] = round(onset_threshold, 4)
            deployment_calibration["offset_threshold"] = round(offset_threshold, 4)
            if boundary_onset_eval:
                on_prob = np.concatenate([item[0] for item in boundary_onset_eval])
                on_labels = np.concatenate([item[1] for item in boundary_onset_eval])
                deployment_calibration["onset"] = threshold_metrics(
                    on_labels, on_prob, onset_threshold
                )
            if boundary_offset_eval:
                off_prob = np.concatenate([item[0] for item in boundary_offset_eval])
                off_labels = np.concatenate([item[1] for item in boundary_offset_eval])
                deployment_calibration["offset"] = threshold_metrics(
                    off_labels, off_prob, offset_threshold
                )

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
                    "deployment_calibration": deployment_calibration,
                },
            }
            args.boundary_out.parent.mkdir(parents=True, exist_ok=True)
            args.boundary_out.write_text(
                json.dumps(boundary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            boundary_written = True
            print(f"[boundary] {args.boundary_out}")
            print(json.dumps({"onset": onset_metrics, "offset": offset_metrics}, ensure_ascii=False))
    if not _training_artifacts_complete(boundary_written, weak_mode, args.skip_boundary):
        print(
            "[error] 训练要求生成 onset/offset 边界模型，但没有写出 boundary model；"
            "请检查边界标注数量/类别，或显式使用 --skip-boundary",
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "artifacts": {
                    "segment_model": True,
                    "boundary_model": boundary_written,
                    "boundary_skipped": bool(weak_mode or args.skip_boundary),
                },
                "model": str(args.model_out),
                "boundary": str(args.boundary_out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
