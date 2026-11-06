from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundaries import musicness_from_flatness
from .models import Candidate, Config, EpisodeFeatures


@dataclass
class Boundary:
    time_s: float
    score: float
    kind: str
    level_db: float = 0.0
    timbre: float = 0.0


def _robust(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    low, high = np.percentile(values, [25, 75])
    return (values - median) / max(float(high - low), 1e-6)


def _smooth(values: np.ndarray, size: int = 3) -> np.ndarray:
    if len(values) < size or size <= 1:
        return values
    left = size // 2
    right = size - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(size) / size, mode="valid")


def _mfcc_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _normalized_mfcc(features: EpisodeFeatures) -> np.ndarray:
    mfcc = np.asarray(features.fine_mfcc, dtype=np.float64)
    normalized = np.zeros_like(mfcc)
    for dim in range(mfcc.shape[1]):
        normalized[:, dim] = _robust(mfcc[:, dim])
    return normalized


def _silence_edges(loudness: np.ndarray, grid: float) -> list[tuple[float, float]]:
    if len(loudness) == 0:
        return []
    noise_floor = float(np.percentile(loudness, 10))
    silent = loudness < noise_floor + 6.0
    edges: list[tuple[float, float]] = []
    start = None
    for index, value in enumerate(silent):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if 15 <= length <= 200:
                duration = length * grid
                strength = min(0.65 + duration * 0.15, 0.95)
                edges.append((start * grid, strength))
                edges.append((index * grid, strength))
            start = None
    return edges


def fine_boundaries(features: EpisodeFeatures) -> list[Boundary]:
    loudness = np.asarray(features.fine_loudness_db, dtype=np.float64)
    flux = np.asarray(features.fine_flux, dtype=np.float64)
    count = len(loudness)
    grid = features.fine_grid_s
    if count < 100 or grid <= 0:
        return []

    window = 50
    kernel = np.ones(window) / window
    pre_loudness = np.convolve(loudness, kernel, mode="full")[:count]
    post_loudness = np.convolve(loudness, kernel, mode="full")[window - 1:window - 1 + count]
    level = np.abs(post_loudness - pre_loudness)

    mfcc = _normalized_mfcc(features)
    pre_mfcc = np.empty_like(mfcc)
    post_mfcc = np.empty_like(mfcc)
    for dim in range(mfcc.shape[1]):
        convolved = np.convolve(mfcc[:, dim], kernel, mode="full")
        pre_mfcc[:, dim] = convolved[:count]
        post_mfcc[:, dim] = convolved[window - 1:window - 1 + count]
    timbre = np.sqrt(np.mean((post_mfcc - pre_mfcc) ** 2, axis=1))

    smoothed_flux = _smooth(flux, 10)
    flux_high = max(float(np.percentile(smoothed_flux, 95)), 1e-6)
    scores = (
        0.40 * np.clip(level / 6.0, 0.0, 1.0)
        + 0.30 * np.clip(timbre / 2.0, 0.0, 1.0)
        + 0.30 * np.clip(smoothed_flux / flux_high, 0.0, 1.0)
    )
    scores[:window] = 0.0
    scores[-window:] = 0.0

    radius = 10
    local = np.ones(count, dtype=bool)
    for offset in range(1, radius + 1):
        local[offset:] &= scores[offset:] >= scores[:-offset]
        local[:-offset] &= scores[:-offset] >= scores[offset:]
    local[:window] = False
    local[-window:] = False

    threshold = max(0.5, float(np.percentile(scores, 99.0)))
    indices = np.flatnonzero(local & (scores >= threshold))
    order = indices[np.argsort(-scores[indices])] if len(indices) else indices
    minimum_gap = int(round(8.0 / grid))
    chosen: list[int] = []
    silence_edges = _silence_edges(loudness, grid)
    for index in order:
        index = int(index)
        if all(abs(index - other) >= minimum_gap for other in chosen):
            chosen.append(index)
        if len(chosen) >= 80:
            break

    for time_s, strength in silence_edges:
        index = int(round(time_s / grid))
        if all(abs(index - other) >= minimum_gap for other in chosen):
            chosen.append(index)
            scores[index] = max(scores[index], strength)

    silence_times = [time_s for time_s, _ in silence_edges]
    chosen.sort()
    boundaries: list[Boundary] = []
    for index in chosen:
        kind = "silence" if any(abs(index * grid - edge) < grid * 2 for edge in silence_times) else "cut"
        boundaries.append(
            Boundary(
                time_s=round(index * grid, 3),
                score=float(np.clip(scores[index], 0.0, 1.0)),
                kind=kind,
                level_db=float(level[index]),
                timbre=float(np.clip(timbre[index] / 2.0, 0.0, 1.0)),
            )
        )
    return boundaries


def _nearest_boundary(boundaries: list[Boundary], time_s: float, limit: float = 3.0) -> tuple[float, float]:
    candidates = [boundary for boundary in boundaries if abs(boundary.time_s - time_s) <= limit and boundary.score >= 0.5]
    if not candidates:
        return time_s, 0.0
    best = min(candidates, key=lambda boundary: (abs(boundary.time_s - time_s), -boundary.score))
    return best.time_s, best.score


def refine_bounds(
    start_s: float,
    end_s: float,
    boundaries: list[Boundary],
) -> tuple[float, float, float]:
    start, start_strength = _nearest_boundary(boundaries, start_s)
    end, end_strength = _nearest_boundary(boundaries, end_s)
    if end - start < 5.0:
        return start_s, end_s, 0.0
    return start, end, (start_strength + end_strength) / 2.0


def _segment_stats(features: EpisodeFeatures, start_s: float, end_s: float) -> dict:
    count = len(features.loudness_db)
    if count == 0:
        return {"musicness": 0.0, "loudness_dev_db": 0.0, "position": 0.0}
    first = int(np.clip(round(start_s / features.grid_s), 0, count - 1))
    last = int(np.clip(round(end_s / features.grid_s), first + 1, count))
    loudness = features.loudness_db[first:last]
    flatness = features.flatness[first:last]
    episode_median = float(np.median(features.loudness_db))
    segment_median = float(np.median(loudness)) if len(loudness) else episode_median
    deviation = abs(segment_median - episode_median)
    musicness = musicness_from_flatness(flatness, features.flatness)
    if start_s < 120:
        position = 1.0
    elif end_s > features.duration_s - 150:
        position = 0.9
    else:
        position = 0.35
    return {
        "musicness": musicness,
        "flatness_median": float(np.median(flatness)) if len(flatness) else 0.0,
        "flatness_baseline": float(np.median(features.flatness)) if count else 0.0,
        "loudness_dev_db": deviation,
        "loudness_median": segment_median,
        "loudness_baseline": episode_median,
        "position": position,
    }


def _duration_prior(duration: float) -> float:
    if 15.0 <= duration <= 60.0:
        return 1.0
    if 60.0 < duration <= 120.0:
        return 0.8
    return 0.5


def insertion_candidates(
    features: EpisodeFeatures,
    cfg: Config,
    boundaries: list[Boundary] | None = None,
) -> list[Candidate]:
    if boundaries is None:
        boundaries = fine_boundaries(features)
    if not boundaries:
        return []
    points = [(-1, Boundary(0.0, 0.25, "episode_start"))]
    points.extend((index, boundary) for index, boundary in enumerate(boundaries))
    points.append((len(boundaries), Boundary(features.duration_s, 0.25, "episode_end")))
    points.sort(key=lambda item: item[1].time_s)

    candidates: list[Candidate] = []
    for (_, first), (_, second) in zip(points, points[1:]):
        start_s, end_s = first.time_s, second.time_s
        duration = end_s - start_s
        if duration < cfg.insertion_min_s or duration > cfg.insertion_max_s:
            continue
        starts_episode = first.kind == "episode_start"
        ends_episode = second.kind == "episode_end"
        if starts_episode and ends_episode:
            continue
        if not starts_episode and not ends_episode:
            if min(first.score, second.score) < 0.3:
                continue
        elif starts_episode:
            if second.score < 0.5 or duration > 120:
                continue
        elif ends_episode:
            if first.score < 0.5 or duration > 120:
                continue
        edge_boundaries = [boundary for boundary in (first, second) if boundary.kind not in {"episode_start", "episode_end"}]
        if not edge_boundaries:
            continue
        boundary_strength = float(np.mean([boundary.score for boundary in (first, second)]))
        timbre_strength = float(np.mean([boundary.timbre for boundary in edge_boundaries]))
        timbre_min = float(min(boundary.timbre for boundary in edge_boundaries))
        stats = _segment_stats(features, start_s, end_s)
        duration_prior = _duration_prior(duration)
        score = (
            0.34 * boundary_strength
            + 0.20 * timbre_strength
            + 0.14 * stats["musicness"]
            + 0.12 * min(stats["loudness_dev_db"] / 6.0, 1.0)
            + 0.10 * stats["position"]
            + 0.10 * duration_prior
        )
        if boundary_strength < 0.35:
            score *= 0.55
        host_like = (
            stats["musicness"] < 0.25
            and stats["loudness_dev_db"] < 3.0
            and timbre_min < 0.7
        )
        if score < cfg.min_insertion_score:
            continue
        kind = "cut"
        if any(boundary.kind == "silence" for boundary in edge_boundaries):
            kind = "silence"
        candidates.append(
            Candidate(
                start_s=round(start_s, 2),
                end_s=round(end_s, 2),
                score=round(min(score, 0.95), 3),
                tier="inserted",
                evidence={
                    "kind": kind,
                    "boundary_strength": round(boundary_strength, 3),
                    "boundary_start": round(first.score, 3),
                    "boundary_end": round(second.score, 3),
                    "timbre_change": round(timbre_strength, 3),
                    "timbre_min": round(timbre_min, 3),
                    "musicness": round(stats["musicness"], 3),
                    "flatness_median": round(stats["flatness_median"], 5),
                    "flatness_baseline": round(stats["flatness_baseline"], 5),
                    "loudness_dev_db": round(stats["loudness_dev_db"], 2),
                    "loudness_median": round(stats["loudness_median"], 2),
                    "loudness_baseline": round(stats["loudness_baseline"], 2),
                    "position": stats["position"],
                    "duration_prior": duration_prior,
                    "host_like": host_like,
                },
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = _merge_pods(candidates)
    return candidates[: cfg.max_insertions_per_episode]


def _merge_pods(candidates: list[Candidate], max_duration: float = 240.0) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: item.start_s)
    merged: list[Candidate] = []
    for candidate in ordered:
        if merged and candidate.start_s - merged[-1].end_s <= 0.3:
            current = merged[-1]
            if candidate.end_s - current.start_s <= max_duration:
                current.end_s = max(current.end_s, candidate.end_s)
                current.score = max(current.score, candidate.score)
                current.evidence["segments"] = current.evidence.get("segments", 1) + 1
                current.evidence["pod"] = True
                continue
        merged.append(candidate)
    return merged
