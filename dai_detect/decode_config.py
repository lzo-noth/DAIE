"""Single source of truth for decoder thresholds.

Training evaluation and runtime inference must use identical enter/exit rules;
otherwise reported event metrics describe a decoder that is never deployed.
"""

from __future__ import annotations

DEFAULT_MIN_ENTER = 0.45
DEFAULT_MAX_ENTER = 0.75
DEFAULT_EXIT_MARGIN = 0.15
DEFAULT_MIN_EXIT = 0.20
# 一个广告位里常连读多家赞助商，实测段间停顿 3.5–7.5 s；2 s 会把同一广告位
# 切成多段，事件级精度因此崩塌。
DEFAULT_MAX_GAP_S = 8.0
# 学习型边界头在真值边界 ±2 s 内能找到峰值，把搜索半径放宽到 6 s 才能纠正
# 解码器给出的粗糙端点。
DEFAULT_SEARCH_S = 6.0


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
