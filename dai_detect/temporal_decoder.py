"""Temporal decoding of window probabilities into ad intervals.

Per-window classifiers produce noisy probabilities. The decoder smooths them,
applies hysteresis (enter/exit thresholds), merges short gaps, enforces a
minimum ad duration and splits implausibly long runs at their weakest valley.
This is the "temporal context" layer: a single high window does not become an
ad, and a brief dip inside an ad does not split it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dai_detect.decode_config import DEFAULT_MAX_GAP_S


@dataclass
class DecodedEvent:
    start: float
    end: float
    probability: float
    peak: float
    first_index: int
    last_index: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float64)
    padded = np.pad(values, 1, mode="edge")
    median = np.median(
        np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1
    )
    padded2 = np.pad(median, 1, mode="edge")
    return np.convolve(padded2, np.ones(3) / 3.0, mode="valid")


def _merge(events: list[DecodedEvent], max_gap: float) -> list[DecodedEvent]:
    merged: list[DecodedEvent] = []
    for event in events:
        if merged and event.start - merged[-1].end <= max_gap:
            previous = merged[-1]
            total = previous.duration + event.duration
            weight = event.duration / max(total, 1e-6)
            previous.end = event.end
            previous.probability = (
                previous.probability * (1.0 - weight) + event.probability * weight
            )
            previous.peak = max(previous.peak, event.peak)
            previous.last_index = event.last_index
            continue
        merged.append(event)
    return merged


def decode_windows(
    times: np.ndarray,
    probabilities: np.ndarray,
    hop_s: float,
    span_s: float,
    min_ad_s: float = 10.0,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
    max_ad_s: float = 600.0,
    enter_threshold: float = 0.55,
    exit_threshold: float = 0.40,
    core_threshold: float | None = None,
) -> list[DecodedEvent]:
    if len(times) == 0 or len(probabilities) == 0:
        return []
    smooth = _smooth(np.asarray(probabilities, dtype=np.float64))
    events: list[DecodedEvent] = []
    active = False
    first = 0
    for index, probability in enumerate(smooth):
        if not active and probability >= enter_threshold:
            active = True
            first = index
        elif active and probability < exit_threshold:
            active = False
            events.append(_make_event(times, smooth, first, index - 1, span_s))
    if active:
        events.append(_make_event(times, smooth, first, len(smooth) - 1, span_s))

    events = _merge(events, max_gap_s)

    kept: list[DecodedEvent] = []
    for event in events:
        if event.duration < min_ad_s:
            continue
        for piece in _split(event, max_ad_s, smooth, times, span_s):
            trimmed = _trim_core(piece, smooth, times, span_s, core_threshold)
            if trimmed.duration >= min_ad_s:
                kept.append(trimmed)
    return kept


def _trim_core(
    event: DecodedEvent,
    smooth: np.ndarray,
    times: np.ndarray,
    span_s: float,
    core_threshold: float | None,
) -> DecodedEvent:
    """Pull event edges back to where the evidence is still strong.

    Hysteresis keeps an event alive through dips, so a break whose trailing
    content scores just above the exit threshold stays attached to the ad and
    pushes its end tens of seconds late. Trimming from each edge until the
    smoothed probability is back above ``core_threshold`` removes that tail
    without touching the interior of the break.
    """
    if core_threshold is None:
        return event
    first, last = event.first_index, event.last_index
    while first < last and smooth[first] < core_threshold:
        first += 1
    while last > first and smooth[last] < core_threshold:
        last -= 1
    if first == event.first_index and last == event.last_index:
        return event
    window = smooth[first: last + 1]
    return DecodedEvent(
        start=max(0.0, float(times[first] - span_s / 2.0)),
        end=float(times[last] + span_s / 2.0),
        probability=float(np.mean(window)) if len(window) else 0.0,
        peak=float(np.max(window)) if len(window) else 0.0,
        first_index=first,
        last_index=last,
    )


def _make_event(
    times: np.ndarray,
    smooth: np.ndarray,
    first: int,
    last: int,
    span_s: float,
) -> DecodedEvent:
    start = float(times[first] - span_s / 2.0)
    end = float(times[last] + span_s / 2.0)
    window = smooth[first: last + 1]
    return DecodedEvent(
        start=max(0.0, start),
        end=end,
        probability=float(np.mean(window)) if len(window) else 0.0,
        peak=float(np.max(window)) if len(window) else 0.0,
        first_index=first,
        last_index=last,
    )


def _split(
    event: DecodedEvent,
    max_ad_s: float,
    smooth: np.ndarray,
    times: np.ndarray,
    span_s: float,
) -> list[DecodedEvent]:
    if event.duration <= max_ad_s:
        return [event]
    pieces: list[DecodedEvent] = []
    remaining = event
    while remaining.duration > max_ad_s:
        length = remaining.last_index - remaining.first_index + 1
        if length < 6:
            break
        local = smooth[remaining.first_index: remaining.last_index + 1]
        low = int(length * 0.25)
        high = int(length * 0.75)
        if high <= low:
            break
        valley = int(np.argmin(local[low:high])) + low
        cut = remaining.first_index + valley
        pieces.append(
            DecodedEvent(
                start=remaining.start,
                end=float(times[cut - 1] + span_s / 2.0),
                probability=float(np.mean(smooth[remaining.first_index:cut])),
                peak=float(np.max(smooth[remaining.first_index:cut])),
                first_index=remaining.first_index,
                last_index=cut - 1,
            )
        )
        remaining = DecodedEvent(
            start=float(times[cut] - span_s / 2.0),
            end=event.end,
            probability=float(np.mean(smooth[cut: event.last_index + 1])),
            peak=float(np.max(smooth[cut: event.last_index + 1])),
            first_index=cut,
            last_index=event.last_index,
        )
    pieces.append(remaining)
    return pieces
