"""Boundary refinement with onset/offset-specific heads.

The decoder yields 0.5 s bounds. Refinement searches the 10 ms boundary
evidence grid with two directions: onset evidence for the start, offset
evidence for the end. Scores are first smoothed over ~0.2 s so cross-fades and
soft splices still form a peak, then snapped to the raw local maximum for
precision. A pair whose interior contains a stronger production change than
its edges is rejected rather than trusted.
"""

from __future__ import annotations

import numpy as np

from .temporal_decoder import DecodedEvent

SMOOTH_EDGE_S = 0.2
SNAP_SEARCH_S = 0.35


def _smooth(values: np.ndarray, half: int) -> np.ndarray:
    if half <= 0 or len(values) == 0:
        return np.asarray(values, dtype=np.float64)
    padded = np.pad(np.asarray(values, dtype=np.float64), half, mode="edge")
    kernel = np.ones(2 * half + 1, dtype=np.float64) / (2 * half + 1)
    return np.convolve(padded, kernel, mode="valid")


def _best_edge(
    times: np.ndarray,
    scores: np.ndarray,
    center: float,
    search_s: float,
    grid_s: float,
) -> tuple[float, float]:
    if len(times) == 0 or len(scores) == 0:
        return center, 0.0
    low = int(np.searchsorted(times, center - search_s, side="left"))
    high = int(np.searchsorted(times, center + search_s, side="right"))
    if high <= low:
        return center, 0.0
    local_raw = np.asarray(scores[low:high], dtype=np.float64)
    half = max(1, int(round(SMOOTH_EDGE_S / max(grid_s, 1e-6))))
    local_smooth = _smooth(local_raw, half)
    coarse_offset = int(np.argmax(local_smooth))
    coarse_time = float(times[low + coarse_offset])

    snap_low = int(np.searchsorted(times, coarse_time - SNAP_SEARCH_S, side="left"))
    snap_high = int(np.searchsorted(times, coarse_time + SNAP_SEARCH_S, side="right"))
    if snap_high <= snap_low:
        return coarse_time, float(local_smooth[coarse_offset])
    snap = np.asarray(scores[snap_low:snap_high], dtype=np.float64)
    offset = int(np.argmax(snap))
    return float(times[snap_low + offset]), float(snap[offset])


def _plausible_pair(
    times: np.ndarray,
    scores: np.ndarray,
    start: float,
    end: float,
    start_score: float,
    end_score: float,
    min_duration: float,
) -> bool:
    if end - start < min_duration:
        return False
    if start_score <= 0.0 or end_score <= 0.0:
        return False
    if len(times) == 0:
        return True
    interior_low = int(np.searchsorted(times, start + 1.0, side="left"))
    interior_high = int(np.searchsorted(times, end - 1.0, side="right"))
    if interior_high <= interior_low:
        return True
    interior = float(np.max(scores[interior_low:interior_high]))
    # If a normal cut inside the span is much stronger than either edge, the
    # detected pair probably brackets the wrong event.
    return interior <= 1.5 * min(start_score, end_score)


def refine_event(
    event: DecodedEvent,
    change: dict,
    search_s: float = 2.5,
    min_duration: float = 4.0,
) -> dict:
    times = np.asarray(change.get("times", []))
    onset_scores = np.asarray(change.get("onset_scores", change.get("scores", [])))
    offset_scores = np.asarray(change.get("offset_scores", change.get("scores", [])))
    combined = np.maximum(onset_scores, offset_scores)
    grid_s = float(times[1] - times[0]) if len(times) > 1 else 0.01

    original_start, original_end = event.start, event.end
    onset_ref = float(np.percentile(onset_scores, 95)) if len(onset_scores) else 0.0
    offset_ref = float(np.percentile(offset_scores, 95)) if len(offset_scores) else 0.0
    onset_threshold = max(0.20, 0.35 * onset_ref)
    offset_threshold = max(0.20, 0.35 * offset_ref)
    if change.get("onset_threshold") is not None:
        onset_threshold = max(0.20, float(change["onset_threshold"]))
    if change.get("offset_threshold") is not None:
        offset_threshold = max(0.20, float(change["offset_threshold"]))

    start, start_score = _best_edge(times, onset_scores, event.start, search_s, grid_s)
    end, end_score = _best_edge(times, offset_scores, event.end, search_s, grid_s)
    if start_score < onset_threshold:
        start, start_score = original_start, 0.0
    if end_score < offset_threshold:
        end, end_score = original_end, 0.0

    start = max(0.0, start)
    usable = _plausible_pair(
        times, combined, start, end, start_score, end_score, min_duration
    )
    if not usable:
        start, end = original_start, original_end
        start_score = end_score = 0.0

    boundary_strength = (start_score + end_score) / 2.0
    confidence = min(0.99, max(0.0, 0.7 * event.probability + 0.3 * boundary_strength))
    return {
        "start": round(start, 2),
        "end": round(end, 2),
        "start_s": round(start, 2),
        "end_s": round(end, 2),
        "duration_s": round(end - start, 2),
        "confidence": round(confidence, 3),
        "evidence": {
            "mean_probability": round(event.probability, 3),
            "peak_probability": round(event.peak, 3),
            "onset_score": round(start_score, 3),
            "offset_score": round(end_score, 3),
            "boundary_kind": change.get("kind", "none"),
            "decoded_start": round(original_start, 2),
            "decoded_end": round(original_end, 2),
            "refined": bool(start_score > 0 or end_score > 0),
        },
    }


def refine_events(
    events: list[DecodedEvent],
    change: dict,
    search_s: float = 2.5,
    min_duration: float = 4.0,
) -> list[dict]:
    return [
        refine_event(event, change, search_s=search_s, min_duration=min_duration)
        for event in events
    ]
