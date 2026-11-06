from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fingerprint import FPS
from .models import Config


@dataclass
class RepetitionRef:
    episode_index: int
    start_frame: int
    end_frame: int
    match_ratio: float


@dataclass
class Repetition:
    start_frame: int
    end_frame: int
    match_ratio: float
    votes: int
    refs: list[RepetitionRef] = field(default_factory=list)


def _dense_clusters(frames: np.ndarray, total_frames: int) -> list[tuple[int, int]]:
    if len(frames) == 0:
        return []
    bin_frames = max(1, int(round(FPS)))
    bin_count = max(1, total_frames // bin_frames + 1)
    histogram = np.bincount(
        np.clip(frames // bin_frames, 0, bin_count - 1).astype(np.int64),
        minlength=bin_count,
    ).astype(np.float32)
    smoothed = np.convolve(histogram, np.ones(3) / 3.0, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return []
    threshold = max(2.5, 0.25 * peak)
    active = np.flatnonzero(smoothed >= threshold)
    if len(active) == 0:
        return []
    clusters: list[tuple[int, int]] = []
    group_start = previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value - previous > 2:
            clusters.append((group_start, previous))
            group_start = value
        previous = value
    clusters.append((group_start, previous))
    return [
        (start_bin * bin_frames, min(total_frames, (end_bin + 1) * bin_frames))
        for start_bin, end_bin in clusters
    ]


def _candidate_bins(counts: np.ndarray, min_votes: int, limit: int = 24) -> list[int]:
    length = len(counts)
    if length < 13:
        return []
    peak = counts >= min_votes
    peak[:6] = False
    peak[-6:] = False
    for k in range(1, 6):
        peak[k:] &= counts[k:] >= counts[:-k]
        peak[:-k] &= counts[:-k] >= counts[k:]
    indices = np.flatnonzero(peak)
    if len(indices) == 0:
        return []
    order = np.argsort(-counts[indices], kind="stable")
    chosen: list[int] = []
    for position in order:
        candidate = int(indices[position])
        if all(abs(candidate - existing) >= 4 for existing in chosen):
            chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return chosen


def _build_index(features) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    max_frames = [max(1, feature.frame_count()) for feature in features]
    hashes = np.concatenate([feature.hashes for feature in features]).astype(np.int64)
    frames = np.concatenate([feature.hash_frames for feature in features]).astype(np.int32)
    episodes = np.concatenate(
        [np.full(len(feature.hashes), index, dtype=np.int16) for index, feature in enumerate(features)]
    )
    order = np.lexsort((frames, hashes))
    return hashes[order], frames[order], episodes[order], max_frames


def _query_episode(
    query_index: int,
    features,
    index_hashes: np.ndarray,
    index_frames: np.ndarray,
    index_episodes: np.ndarray,
    max_frames: list[int],
    cfg: Config,
) -> list[Repetition]:
    query = features[query_index]
    query_hashes = query.hashes.astype(np.int64)
    query_frames = query.hash_frames.astype(np.int64)
    if len(query_hashes) == 0:
        return []

    left = np.searchsorted(index_hashes, query_hashes, side="left")
    right = np.searchsorted(index_hashes, query_hashes, side="right")
    counts = right - left
    usable = (counts > 0) & (counts <= cfg.max_hash_df)
    if not usable.any():
        return []
    left, counts, query_positions = left[usable], counts[usable], np.flatnonzero(usable)
    total = int(counts.sum())
    query_slots = np.repeat(query_positions, counts)
    offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    postings = np.repeat(left, counts) + offsets

    episodes = index_episodes[postings]
    keep = episodes != query_index
    postings, query_slots = postings[keep], query_slots[keep]
    episodes = episodes[keep]
    if len(postings) == 0:
        return []

    posting_frames = index_frames[postings].astype(np.int64)
    deltas = query_frames[query_slots] - posting_frames
    min_run = int(round(cfg.min_run_s * FPS))
    max_gap = int(round(cfg.max_gap_s * FPS))
    detections: list[Repetition] = []

    for episode in np.unique(episodes):
        mask = episodes == episode
        sub_slots = query_slots[mask]
        sub_posting_frames = posting_frames[mask]
        sub_deltas = deltas[mask]
        span = max_frames[episode]
        shifted = sub_deltas + span
        inside = (shifted >= 0) & (shifted <= 2 * span)
        if not inside.any():
            continue
        sub_slots = sub_slots[inside]
        sub_posting_frames = sub_posting_frames[inside]
        sub_deltas = sub_deltas[inside]
        histogram = np.bincount(shifted[inside].astype(np.int64), minlength=2 * span + 1)

        for bin_index in _candidate_bins(histogram, cfg.min_votes):
            delta = bin_index - span
            aligned = np.abs(sub_deltas - delta) <= 1
            if int(aligned.sum()) < cfg.min_votes:
                continue
            matched_frames = query_frames[sub_slots[aligned]]
            for start, end in _dense_clusters(matched_frames, max_frames[query_index]):
                if end - start < min_run:
                    continue
                inside_run = (query_frames >= start) & (query_frames <= end)
                denominator = int(inside_run.sum())
                if denominator == 0:
                    continue
                in_window = aligned & (query_frames[sub_slots] >= start) & (query_frames[sub_slots] <= end)
                numerator = int(in_window.sum())
                ratio = numerator / denominator
                if ratio < cfg.min_match_ratio:
                    continue
                other_frames = sub_posting_frames[in_window]
                if len(other_frames) == 0:
                    continue
                detections.append(
                    Repetition(
                        start_frame=start,
                        end_frame=end,
                        match_ratio=ratio,
                        votes=numerator,
                        refs=[
                            RepetitionRef(
                                episode_index=int(episode),
                                start_frame=int(other_frames.min()),
                                end_frame=int(other_frames.max()),
                                match_ratio=ratio,
                            )
                        ],
                    )
                )

    return _merge_detections(detections, max_gap)


def _merge_detections(detections: list[Repetition], max_gap: int) -> list[Repetition]:
    if not detections:
        return []
    ordered = sorted(detections, key=lambda item: item.start_frame)
    merged: list[Repetition] = []
    for item in ordered:
        if merged and item.start_frame <= merged[-1].end_frame + max_gap:
            current = merged[-1]
            current.end_frame = max(current.end_frame, item.end_frame)
            current.match_ratio = max(current.match_ratio, item.match_ratio)
            current.votes = max(current.votes, item.votes)
            for ref in item.refs:
                _add_ref(current, ref, max_gap)
        else:
            merged.append(
                Repetition(
                    start_frame=item.start_frame,
                    end_frame=item.end_frame,
                    match_ratio=item.match_ratio,
                    votes=item.votes,
                    refs=[RepetitionRef(**vars(ref)) for ref in item.refs],
                )
            )
    for current in merged:
        current.refs = sorted(current.refs, key=lambda ref: ref.episode_index)
    return merged


def _add_ref(repetition: Repetition, ref: RepetitionRef, max_gap: int) -> None:
    for existing in repetition.refs:
        if existing.episode_index != ref.episode_index:
            continue
        if ref.start_frame <= existing.end_frame + max_gap and existing.start_frame <= ref.end_frame + max_gap:
            existing.start_frame = min(existing.start_frame, ref.start_frame)
            existing.end_frame = max(existing.end_frame, ref.end_frame)
            existing.match_ratio = max(existing.match_ratio, ref.match_ratio)
            return
    repetition.refs.append(RepetitionRef(**vars(ref)))


def find_repetitions(features, cfg: Config) -> list[list[Repetition]]:
    if len(features) < 2:
        return [[] for _ in features]
    index_hashes, index_frames, index_episodes, max_frames = _build_index(features)
    results: list[list[Repetition]] = []
    for query_index in range(len(features)):
        results.append(
            _query_episode(
                query_index,
                features,
                index_hashes,
                index_frames,
                index_episodes,
                max_frames,
                cfg,
            )
        )
    return results
