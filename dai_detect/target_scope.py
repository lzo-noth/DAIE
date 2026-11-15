"""Which annotated ads the detector is trained to find.

SPEC 的检测目标是第三方**预录**广告（``produced``）。主播口播（``host_read``）
在声学上最接近正文，如果混进正样本，分类器会学到"像正文的广告"，这既拉低
区分度，也和产品目标不一致，所以它作为明确的非目标负样本参与训练。

制作风格未知（``unknown`` 或缺失）既不能擅自当正样本，也不能当负样本：它
只从监督和评测中屏蔽，直到补完标注。
"""

from __future__ import annotations

TARGET_PRODUCTION_STYLE = "produced"
# 这些风格是明确非目标：窗口标签按普通内容处理，变成 hard negative。
NON_TARGET_PRODUCTION_STYLES = frozenset({"host_read"})
# 风格缺失/未标注时，区间两端都不使用。
IGNORED_PRODUCTION_STYLES = frozenset({"", "unknown"})


def ad_style(ad: dict) -> str:
    return str(ad.get("ad_production_style") or "").strip()


def is_target_ad(ad: dict) -> bool:
    return ad_style(ad) == TARGET_PRODUCTION_STYLE


def split_ads(ads: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return ``(target, non_target, ignored)`` under the produced-only policy."""
    target: list[dict] = []
    non_target: list[dict] = []
    ignored: list[dict] = []
    for ad in ads:
        style = ad_style(ad)
        if style == TARGET_PRODUCTION_STYLE:
            target.append(ad)
        elif style in NON_TARGET_PRODUCTION_STYLES:
            non_target.append(ad)
        else:
            ignored.append(ad)
    return target, non_target, ignored


def target_scope_payload() -> dict:
    """Model-facing description so a loaded model states what it detects."""
    return {
        "ad_production_style": TARGET_PRODUCTION_STYLE,
        "non_target_styles": sorted(NON_TARGET_PRODUCTION_STYLES),
        "ignored_styles": sorted(IGNORED_PRODUCTION_STYLES),
    }
