from __future__ import annotations

import numpy as np

from .models import Candidate, Config, EpisodeFeatures


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    low, high = np.percentile(values, [25, 75])
    scale = max(float(high - low), 1e-6)
    return (values - median) / scale


def novelty_curve(features: EpisodeFeatures) -> np.ndarray:
    count = len(features.loudness_db)
    if count < 20:
        return np.zeros(count, dtype=np.float32)
    loudness = _robust_normalize(features.loudness_db)
    flatness = _robust_normalize(features.flatness)
    flux = _robust_normalize(features.flux)
    mfcc = np.column_stack(
        [_robust_normalize(features.mfcc[:, index]) for index in range(features.mfcc.shape[1])]
    )
    delta_loud = np.abs(np.diff(loudness, prepend=loudness[0]))
    delta_flat = np.abs(np.diff(flatness, prepend=flatness[0]))
    delta_flux = np.abs(np.diff(flux, prepend=flux[0]))
    delta_mfcc = np.sqrt(np.mean(np.diff(mfcc, axis=0, prepend=mfcc[0:1]) ** 2, axis=1))
    raw = 0.40 * delta_loud + 0.15 * delta_flat + 0.15 * delta_flux + 0.30 * delta_mfcc
    padded = np.pad(raw, 2, mode="edge")
    return np.convolve(padded, np.ones(5) / 5.0, mode="valid").astype(np.float32)


def detect_boundaries(features: EpisodeFeatures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    novelty = novelty_curve(features)
    count = len(novelty)
    if count < 30:
        return np.zeros(0), np.zeros(0), novelty
    local = np.ones(count, dtype=bool)
    for offset in range(1, 6):
        local[offset:] &= novelty[offset:] >= novelty[:-offset]
        local[:-offset] &= novelty[:-offset] >= novelty[offset:]
    local[:10] = False
    local[-10:] = False
    high = np.percentile(novelty, 95)
    high = max(float(high), 1e-6)
    threshold = max(float(np.percentile(novelty, 88)), float(np.median(novelty) + 1.5))
    candidates = np.flatnonzero(local & (novelty >= threshold))
    if len(candidates) == 0:
        return np.zeros(0), np.zeros(0), novelty

    separated: list[int] = []
    minimum_gap = 30
    for index in candidates[np.argsort(-novelty[candidates])]:
        index = int(index)
        if all(abs(index - other) >= minimum_gap for other in separated):
            separated.append(index)
    separated.sort()
    indices = np.asarray(separated, dtype=np.int32)
    strengths = np.clip(novelty[indices] / high, 0.0, 1.0)
    return indices * features.grid_s, strengths, novelty


def _boundary_strength(indices: np.ndarray, strengths: np.ndarray, seconds: float) -> float:
    if len(indices) == 0:
        return 0.0
    distances = np.abs(indices - seconds)
    position = int(np.argmin(distances))
    if distances[position] > 2.5:
        return 0.0
    return float(strengths[position])


def segment_evidence(
    features: EpisodeFeatures,
    start_s: float,
    end_s: float,
    boundary_indices: np.ndarray | None = None,
    boundary_strengths: np.ndarray | None = None,
    novelty: np.ndarray | None = None,
) -> dict:
    count = len(features.loudness_db)
    if count == 0:
        return {
            "musicness": 0.0,
            "loudness_dev_db": 0.0,
            "boundary_strength": 0.0,
            "position": 0.0,
        }
    first = int(np.clip(round(start_s / features.grid_s), 0, count - 1))
    last = int(np.clip(round(end_s / features.grid_s), first + 1, count))
    loudness = features.loudness_db[first:last]
    flatness = features.flatness[first:last]
    episode_median = float(np.median(features.loudness_db))
    segment_median = float(np.median(loudness)) if len(loudness) else episode_median
    deviation = abs(segment_median - episode_median)
    tonality_rank = float(np.mean(features.flatness < (np.median(flatness) if len(flatness) else 0.0)))
    musicness = float(np.clip((0.5 - tonality_rank) / 0.5, 0.0, 1.0))
    if boundary_indices is None or boundary_strengths is None:
        boundary_indices, boundary_strengths, novelty = detect_boundaries(features)
    edge_start = _boundary_strength(boundary_indices, boundary_strengths, start_s)
    edge_end = _boundary_strength(boundary_indices, boundary_strengths, end_s)
    edges = [value for value in (edge_start, edge_end) if value > 0]
    boundary = float(np.mean(edges)) if edges else 0.2
    if start_s < 90:
        position = 1.0
    elif end_s > features.duration_s - 120:
        position = 0.9
    else:
        position = 0.35
    return {
        "musicness": round(musicness, 3),
        "loudness_dev_db": round(deviation, 2),
        "boundary_strength": round(boundary, 3),
        "position": position,
    }


def acoustic_candidates(features: EpisodeFeatures, cfg: Config) -> list[Candidate]:
    boundary_s, strength, novelty = detect_boundaries(features)
    points = [0.0, *[float(value) for value in boundary_s], float(features.duration_s)]
    points = sorted({round(value, 3) for value in points})
    candidates: list[Candidate] = []
    for start, end in zip(points, points[1:]):
        duration = end - start
        if duration < cfg.acoustic_min_s or duration > cfg.acoustic_max_s:
            continue
        evidence = segment_evidence(features, start, end, boundary_s, strength, novelty)
        score = (
            0.30 * evidence["musicness"]
            + 0.30 * min(evidence["loudness_dev_db"] / 6.0, 1.0)
            + 0.20 * evidence["boundary_strength"]
            + 0.20 * evidence["position"]
        )
        if score < cfg.min_acoustic_score:
            continue
        candidates.append(
            Candidate(
                start_s=round(start, 2),
                end_s=round(end, 2),
                score=round(min(score, 0.95), 3),
                tier="acoustic",
                evidence=evidence,
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[: cfg.max_acoustic_per_episode]
