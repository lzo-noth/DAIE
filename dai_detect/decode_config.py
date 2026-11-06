"""Single source of truth for decoder thresholds.

Training evaluation and runtime inference must use identical enter/exit rules;
otherwise reported event metrics describe a decoder that is never deployed.
"""

from __future__ import annotations

DEFAULT_MIN_ENTER = 0.45
DEFAULT_MAX_ENTER = 0.75
DEFAULT_EXIT_MARGIN = 0.15
DEFAULT_MIN_EXIT = 0.20


def resolve_thresholds(
    threshold: float,
    enter_threshold: float | None = None,
    exit_threshold: float | None = None,
) -> tuple[float, float]:
    if enter_threshold is not None:
        enter = float(enter_threshold)
    else:
        enter = min(DEFAULT_MAX_ENTER, max(DEFAULT_MIN_ENTER, float(threshold) + 0.05))
    if exit_threshold is not None:
        exit_value = float(exit_threshold)
    else:
        exit_value = max(DEFAULT_MIN_EXIT, enter - DEFAULT_EXIT_MARGIN)
    return enter, exit_value
