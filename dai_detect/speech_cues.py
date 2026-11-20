"""Content cues from word timings.

Acoustic features alone cannot tell a dry sponsor read from ordinary narration:
the blind reports show the model firing on show opens (music + announcer) while
missing plain ad reads. Sponsor reads do carry content markers — "this message
comes from", "use code", "dot com", "terms and conditions" — so the detector is
allowed one lightweight content view: word timings, either from the annotation
pipeline's transcript or from a small offline ASR model at runtime.

This module only turns word timings into per-window numbers; it never reads
labels, and it is a pure function so training and inference stay identical.
"""

from __future__ import annotations

import math
import re

import numpy as np

CUE_SCHEMA = "speech_cue_v1"
CUE_FEATURES: list[str] = [
    "cue_marker_now",
    "cue_marker_near",
    "cue_since_marker",
    "cue_until_marker",
    "cue_code_near",
]

# 单词级标记：广告口播的固定话术。
_MARKER_WORDS = frozenset(
    {
        "sponsor",
        "sponsors",
        "sponsored",
        "sponsorship",
        "advertisement",
        "advertisements",
        "promo",
        "promos",
        "promotion",
        "promotions",
        "code",
        "codes",
        "offer",
        "offers",
        "discount",
        "discounts",
        "trial",
        "subscription",
        "subscriptions",
        "subscribe",
        "visit",
        "website",
        "deal",
        "coupon",
        "percent",
        "dotcom",
    }
)

# 站内常见但含义模糊的词只在短语里计分，避免把节目内容里的 "support" 当广告。
_MARKER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("this", "message", "comes", "from"),
    ("support", "for"),
    ("supported", "by"),
    ("brought", "to", "you", "by"),
    ("sponsored", "by"),
    ("our", "sponsors"),
    ("use", "code"),
    ("promo", "code"),
    ("terms", "and", "conditions"),
    ("privacy", "policy"),
    ("free", "trial"),
    ("off", "your", "first"),
    ("dot", "com"),
    ("sign", "up"),
    ("go", "to"),
)

_CODE_RE = re.compile(r"^\d+$|^\d+%$|%$|^\d+(\.\d+)?$")
_CODE_WORDS = frozenset({"code", "percent", "dollars", "free", "off", "deal", "discount"})

_TOKEN_RE = re.compile(r"[a-z0-9%$\.]+")
_NEAR_S = 8.0


def _words(segments: list[dict]) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Expand segment text into evenly spaced pseudo-word timings."""
    times: list[tuple[str, float]] = []
    for segment in segments or []:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment["start_s"])
            end = float(segment["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        tokens = _TOKEN_RE.findall(str(segment.get("text") or "").casefold())
        if not tokens:
            continue
        step = (end - start) / len(tokens)
        for index, token in enumerate(tokens):
            times.append((token, start + (index + 0.5) * step))
    times.sort(key=lambda item: item[1])
    return np.asarray([item[1] for item in times], dtype=np.float64), times


def _marker_hits(tokens: list[str], times: list[tuple[str, float]]) -> list[tuple[float, float]]:
    """Return (time, weight) marker evidence."""
    hits: list[tuple[float, float]] = []
    for token, at in times:
        if token in _MARKER_WORDS:
            hits.append((at, 1.0))
    lowered = [token for token, _ in times]
    starts = [at for _, at in times]
    for phrase in _MARKER_PHRASES:
        width = len(phrase)
        for index in range(len(lowered) - width + 1):
            if tuple(lowered[index : index + width]) == phrase:
                # 短语比单词更有指向性；时间取短语中点。
                where = (starts[index] + starts[index + width - 1]) / 2.0
                hits.append((where, 2.0))
    hits.sort()
    return hits


def _code_hits(times: list[tuple[str, float]]) -> list[float]:
    hits: list[float] = []
    for token, at in times:
        if token in _CODE_WORDS or _CODE_RE.search(token):
            hits.append(at)
    return hits


def cue_matrix(
    segments: list[dict],
    times: np.ndarray,
    span_s: float = 1.0,
    near_s: float = _NEAR_S,
) -> np.ndarray:
    """Build the (windows x len(CUE_FEATURES)) content-cue matrix."""
    times = np.asarray(times, dtype=np.float64)
    matrix = np.zeros((len(times), len(CUE_FEATURES)), dtype=np.float32)
    if len(times) == 0:
        return matrix

    _, words = _words(segments)
    if not words:
        # 没有转写时给出中性值：既不加分也不减分。
        matrix[:, 2] = 30.0
        matrix[:, 3] = 30.0
        return matrix

    markers = _marker_hits([token for token, _ in words], words)
    marker_times = np.asarray([at for at, _ in markers], dtype=np.float64)
    marker_weights = np.asarray([weight for _, weight in markers], dtype=np.float64)
    codes = np.asarray(_code_hits(words), dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(marker_weights)))

    for index, center in enumerate(times):
        window_start = center - span_s / 2.0
        window_end = center + span_s / 2.0
        low = int(np.searchsorted(marker_times, window_start, side="left"))
        high = int(np.searchsorted(marker_times, window_end, side="right"))
        matrix[index, 0] = float(cumulative[high] - cumulative[low])

        near_low = int(np.searchsorted(marker_times, center - near_s, side="left"))
        near_high = int(np.searchsorted(marker_times, center + near_s, side="right"))
        matrix[index, 1] = float(
            np.log1p(cumulative[near_high] - cumulative[near_low])
        )

        if low > 0:
            matrix[index, 2] = float(min(30.0, center - marker_times[low - 1]))
        else:
            matrix[index, 2] = 30.0
        if high < len(marker_times):
            matrix[index, 3] = float(min(30.0, marker_times[high] - center))
        else:
            matrix[index, 3] = 30.0

        if len(codes):
            code_low = int(np.searchsorted(codes, center - near_s, side="left"))
            code_high = int(np.searchsorted(codes, center + near_s, side="right"))
            matrix[index, 4] = float(np.log1p(code_high - code_low))

    return matrix


def marker_times(segments: list[dict]) -> list[float]:
    """Times of ad-marker words/phrases, used to anchor event starts."""
    _, words = _words(segments)
    if not words:
        return []
    return sorted(at for at, _ in _marker_hits([token for token, _ in words], words))


def attach_speech_cues(windows: dict, segments: list[dict]) -> dict:
    """Append content cues to a window payload, keeping names aligned."""
    times = np.asarray(windows.get("times", []), dtype=np.float64)
    cues = cue_matrix(segments, times, span_s=float(windows.get("span_s", 1.0)))
    features = np.asarray(windows["features"], dtype=np.float32)
    return {
        **windows,
        "features": np.hstack([features, cues]).astype(np.float32),
        "names": list(windows["names"]) + list(CUE_FEATURES),
    }
