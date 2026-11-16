"""Per-episode rolling context for the window classifier.

A 1.0 s window cannot tell whether the last half minute has sounded like a
sponsor read, which is the information that separates an ad from a one-off
music sting or a loud transition. These features are computed inside one
episode and from acoustic features only; they never look at labels, transcripts
or other episodes.
"""

from __future__ import annotations

import numpy as np

CONTEXT_SPANS: tuple[float, ...] = (4.0, 12.0, 30.0)
CONTEXT_SCHEMA = "window_v3_context_mean"


def context_feature_names(
    names: list[str], spans: tuple[float, ...] = CONTEXT_SPANS
) -> list[str]:
    return list(names) + [
        f"{name}__ctx{span:g}s" for span in spans for name in names
    ]


def window_context(
    features: np.ndarray,
    times: np.ndarray,
    spans: tuple[float, ...] = CONTEXT_SPANS,
) -> np.ndarray:
    """Append the mean of every feature over ±span seconds.

    ``times`` must be sorted, which window extraction guarantees.
    """
    matrix = np.asarray(features, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    cumulative = np.vstack(
        [
            np.zeros((1, matrix.shape[1]), dtype=np.float64),
            np.cumsum(matrix, axis=0, dtype=np.float64),
        ]
    )
    parts = [matrix]
    for span in spans:
        lower = np.searchsorted(times, times - span, side="left")
        upper = np.searchsorted(times, times + span, side="right")
        counts = np.maximum(upper - lower, 1).astype(np.float64)[:, None]
        parts.append(((cumulative[upper] - cumulative[lower]) / counts).astype(np.float32))
    return np.hstack(parts)
