"""Boundary refinement.

The temporal decoder works on a 0.5 s window grid, so its bounds are coarse.
This module searches ±2.5 s around each decoded edge for the strongest
production change on the 10 ms forensic grid. If no change stands out, the
decoded bound is kept rather than snapped to noise.
"""

from __future__ import annotations

import numpy as np

from .temporal_decoder import DecodedEvent


def _best_edge(times: np.ndarray, scores: np.ndarray, center: float, search_s: float) -> tuple[float, float]:
    if len(times) == 0:
        return center, 0.0
    low = int(np.searchsorted(times, center - search_s, side="left"))
    high = int(np.searchsorted(times, center + search_s, side="right"))
    if high <= low:
        return center, 0.0
    local = scores[low:high]
    if len(local) == 0:
        return center, 0.0
    best_offset = int(np.argmax(local))
    return float(times[low + best_offset]), float(local[best_offset])


def refine_event(
    event: DecodedEvent,
    change: dict,
    search_s: float = 2.5,
    min_duration: float = 4.0,
) -> dict:
    times = np.asarray(change.get("times", []))
    scores = np.asarray(change.get("scores", []))
    original_start, original_end = event.start, event.end

    global_strength = float(np.percentile(scores, 95)) if len(scores) else 0.0
    threshold = max(0.20, 0.35 * global_strength)

    start, start_score = _best_edge(times, scores, event.start, search_s)
    end, end_score = _best_edge(times, scores, event.end, search_s)
    if start_score < threshold:
        start = original_start
        start_score = 0.0
    if end_score < threshold:
        end = original_end
        end_score = 0.0
    start = max(0.0, start)
    if end - start < min_duration:
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
            "boundary_start": round(start_score, 3),
            "boundary_end": round(end_score, 3),
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
