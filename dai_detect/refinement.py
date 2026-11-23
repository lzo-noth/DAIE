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
# 解码器给出的端点常常偏离真值几十秒（盲测 p95 约 38 s）：固定 ±6 s 的搜索
# 半径够不到，精修等于没做。半径随事件长度自适应，但设上限避免横跨整段内容。
EDGE_SEARCH_FRACTION = 0.30
EDGE_SEARCH_MAX_S = 15.0
# 只有当新端点明显强于当前端点时才移动，避免把本来正确的端点拽到更响的
# 节目音乐上（盲测里精修"改好 11 次、改坏 8 次"，几乎等于抛硬币）。
EDGE_IMPROVE_FACTOR = 1.25
# 单条赞助商创意很少超过 70 s；解码出的长块通常是整段广告位（2–4 家）。
MAX_SINGLE_EVENT_S = 70.0
SPLIT_SCORE_FACTOR = 1.5
# 广告起读几乎总带固定话术（"this message comes from…"），盲测里真值起点到最近
# 标记词的中位距离只有 0.8 s，比任何声学变化点都准。起点优先吸附到标记词。
CUE_SNAP_S = 12.0


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


def _score_near(times: np.ndarray, scores: np.ndarray, at: float) -> float:
    """Best raw score within one grid cell of ``at``."""
    if len(times) == 0 or len(scores) == 0:
        return 0.0
    index = int(np.clip(np.searchsorted(times, at), 0, len(times) - 1))
    low = max(0, index - 1)
    high = min(len(scores), index + 2)
    return float(np.max(scores[low:high]))


def _edge_search_s(start: float, end: float, search_s: float) -> float:
    span = max(0.0, end - start)
    return max(float(search_s), min(EDGE_SEARCH_FRACTION * span, EDGE_SEARCH_MAX_S))


def _split_span(
    times: np.ndarray,
    onset: np.ndarray,
    offset: np.ndarray,
    start: float,
    end: float,
    min_duration: float,
    threshold: float,
) -> list[tuple[float, float]]:
    """Cut an over-long decoded block at its strongest internal production change.

    The failed blind runs show the decoder keeping one long block across a whole
    ad pod while the labels are per sponsor creative. Boundary scores already
    learn those creative transitions, so a strong internal peak is the right
    place to cut; without a strong peak the block is left alone.
    """
    pieces: list[tuple[float, float]] = [(start, end)]
    if len(times) == 0 or end - start <= MAX_SINGLE_EVENT_S:
        return pieces
    for _ in range(3):
        result: list[tuple[float, float]] = []
        changed = False
        for low, high in pieces:
            if high - low <= MAX_SINGLE_EVENT_S:
                result.append((low, high))
                continue
            first = int(np.searchsorted(times, low + min_duration, side="left"))
            last = int(np.searchsorted(times, high - min_duration, side="left"))
            if last <= first:
                result.append((low, high))
                continue
            combined = np.maximum(onset[first:last], offset[first:last])
            offset_index = int(np.argmax(combined))
            if float(combined[offset_index]) < threshold:
                result.append((low, high))
                continue
            cut = float(times[first + offset_index])
            result.extend([(low, cut), (cut, high)])
            changed = True
        pieces = result
        if not changed:
            break
    return pieces


def _refine_span(
    times: np.ndarray,
    onset_scores: np.ndarray,
    offset_scores: np.ndarray,
    start: float,
    end: float,
    search_s: float,
    grid_s: float,
    onset_threshold: float,
    offset_threshold: float,
    pin_start: bool = False,
    pin_end: bool = False,
) -> tuple[float, float, float, float]:
    """Snap one span's edges to onset/offset peaks; returns the new edges+scores."""
    radius = _edge_search_s(start, end, search_s)
    if pin_start:
        new_start, start_score = start, 0.0
    else:
        new_start, start_score = _best_edge(times, onset_scores, start, radius, grid_s)
    if pin_end:
        new_end, end_score = end, 0.0
    else:
        new_end, end_score = _best_edge(times, offset_scores, end, radius, grid_s)

    # 只有明显更强的峰才值得移动端点；否则保留解码器给的位置。
    if not pin_start and start_score < max(
        onset_threshold, EDGE_IMPROVE_FACTOR * _score_near(times, onset_scores, start)
    ):
        new_start, start_score = start, 0.0
    if not pin_end and end_score < max(
        offset_threshold, EDGE_IMPROVE_FACTOR * _score_near(times, offset_scores, end)
    ):
        new_end, end_score = end, 0.0
    return new_start, new_end, start_score, end_score


def _cue_start(
    cue_starts: list[float], original_start: float, end: float, min_duration: float
) -> float | None:
    """Nearest marker time around the decoded start, if it keeps the event valid."""
    if not cue_starts:
        return None
    best = min(cue_starts, key=lambda value: abs(value - original_start))
    if abs(best - original_start) > CUE_SNAP_S:
        return None
    if best < 0.0 or end - best < min_duration:
        return None
    return float(best)


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

    start, end, start_score, end_score = _refine_span(
        times,
        onset_scores,
        offset_scores,
        original_start,
        original_end,
        search_s,
        grid_s,
        onset_threshold,
        offset_threshold,
        pin_start=bool(getattr(event, "fixed_start", False)),
        pin_end=bool(getattr(event, "fixed_end", False)),
    )
    if start_score <= 0.0:
        start = original_start
    if end_score <= 0.0:
        end = original_end

    cue_start = _cue_start(change.get("cue_starts") or [], original_start, end, min_duration)
    start = max(0.0, start)
    if cue_start is not None:
        # 标记词本身就是证据，不再要求声学端点分数为正。
        start = cue_start
        usable = end - start >= min_duration
    else:
        usable = _plausible_pair(
            times, combined, start, end, start_score, end_score, min_duration
        )
    if not usable:
        start, end = original_start, original_end
        start_score = end_score = 0.0
        cue_start = None

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
            "start_source": (
                "cue" if cue_start is not None else ("boundary" if start_score > 0 else "decoded")
            ),
            "refined": bool(start_score > 0 or end_score > 0 or cue_start is not None),
        },
    }


def _split_decoded_events(
    events: list[DecodedEvent], change: dict, min_duration: float
) -> list[DecodedEvent]:
    times = np.asarray(change.get("times", []))
    onset = np.asarray(change.get("onset_scores", change.get("scores", [])))
    offset = np.asarray(change.get("offset_scores", change.get("scores", [])))
    cue_starts = change.get("cue_starts") or []
    phrases = sorted(float(value) for value in (change.get("cue_phrases") or []))
    if len(times) == 0:
        return events
    onset_threshold = float(change.get("onset_threshold", 0.0) or 0.0)
    offset_threshold = float(change.get("offset_threshold", 0.0) or 0.0)
    threshold = SPLIT_SCORE_FACTOR * max(0.20, onset_threshold, offset_threshold)
    pieces: list[DecodedEvent] = []
    for event in events:
        start = event.start
        end = event.end
        snapped = _cue_start(cue_starts, start, end, min_duration)
        if snapped is not None:
            start = snapped
        # 先按“新赞助商开场语”切分：盲测里预测普遍比真值长 1.5–2 倍，正是把
        # 相邻创意并成一块；这些短语就是标注用的创意边界。
        cuts: list[float] = []
        for cut in phrases:
            if cut < start + min_duration or cut > end - min_duration:
                continue
            if cuts and cut - cuts[-1] < min_duration:
                continue
            cuts.append(cut)
        bounds = [start] + cuts + [end]
        spans = list(zip(bounds, bounds[1:]))
        if len(spans) == 1:
            spans = _split_span(
                times, onset, offset, start, end, min_duration, threshold
            )
        event_pieces: list[DecodedEvent] = []
        for low, high in spans:
            if high - low < min_duration:
                continue
            event_pieces.append(
                DecodedEvent(
                    start=low,
                    end=high,
                    probability=event.probability,
                    peak=event.peak,
                    first_index=event.first_index,
                    last_index=event.last_index,
                    # 因开场语/短语切出来的端点直接采信，不再交给声学峰。
                    fixed_start=low != event.start or snapped is not None,
                    fixed_end=abs(high - event.end) > 1e-6,
                )
            )
        # 切分结果必须覆盖到原事件终点，否则退回原事件，避免把检出弄丢。
        if event_pieces and abs(event_pieces[-1].end - end) < 1e-6:
            pieces.extend(event_pieces)
        else:
            pieces.append(event)
    return pieces


def refine_events(
    events: list[DecodedEvent],
    change: dict,
    search_s: float = 2.5,
    min_duration: float = 4.0,
) -> list[dict]:
    split = _split_decoded_events(events, change, min_duration)
    refined = [
        refine_event(event, change, search_s=search_s, min_duration=min_duration)
        for event in split
    ]
    kept = [item for item in refined if item["duration_s"] >= min_duration]
    if kept:
        return kept
    return refined
