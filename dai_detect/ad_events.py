"""Split an annotated ad block into one event per sponsor creative.

一个广告位常连放 2–4 家赞助商，标注却把它记成一个 60–90 s 的区间。窗口分类器
只会在单条创意上响应，事件匹配因此同时产生漏检和误报。这里用逐段判定
（``host_read`` 与 ``brand``）把广告块切成与检测粒度一致的事件：

* ``host_read`` 变化 → 切开；
* 品牌变化 → 切开（空品牌视为延续，避免把一句话的停顿当成新广告）；
* 小于 ``min_duration_s`` 的碎片丢弃（它们更可能是过渡音效，不是独立创意）。

切分只依赖已有的转写判定，不引入新模型，也不改单音频推理路径。
"""

from __future__ import annotations

EVENT_GRANULARITY = "sponsor_creative"


def _brand(verdict: dict) -> str:
    return str(verdict.get("brand") or "").strip().casefold()


def _is_third_party_ad(verdict: dict) -> bool:
    return (
        verdict.get("is_ad") is True
        and str(verdict.get("ad_type") or "") == "third_party_ad"
    )


def split_ad_events(
    ad: dict,
    segments: list[dict],
    results: dict[str, dict],
    min_duration_s: float = 5.0,
) -> list[dict]:
    """Return per-creative events for one annotated ad block.

    A block whose style is unknown is returned untouched: the segment verdicts
    must not silently promote an unlabelled ad to ``produced``.
    """
    if str(ad.get("ad_production_style") or "") not in ("produced", "host_read"):
        return [dict(ad)]
    ad_start = float(ad["start_s"])
    ad_end = float(ad["end_s"])
    groups: list[dict] = []
    for index, segment in enumerate(segments):
        seg_start = float(segment.get("start_s") or 0.0)
        seg_end = float(segment.get("end_s") or 0.0)
        if min(ad_end, seg_end) - max(ad_start, seg_start) <= 0:
            continue
        verdict = results.get(str(index)) or {}
        if not _is_third_party_ad(verdict):
            continue
        style = "host_read" if verdict.get("host_read") is True else "produced"
        brand = _brand(verdict)
        start = max(ad_start, seg_start)
        end = min(ad_end, seg_end)
        if groups:
            current = groups[-1]
            brand_changed = bool(brand and current["brand"] and brand != current["brand"])
            if current["style"] == style and not brand_changed:
                current["start"] = min(current["start"], start)
                current["end"] = max(current["end"], end)
                current["brand"] = current["brand"] or brand
                continue
        groups.append(
            {"start": start, "end": end, "style": style, "brand": brand}
        )

    kept = [item for item in groups if item["end"] - item["start"] >= min_duration_s]
    if not kept:
        return [dict(ad)]

    merged: list[dict] = []
    for item in kept:
        last = merged[-1] if merged else None
        same_brand = bool(
            last
            and (
                not item["brand"]
                or not last["brand"]
                or item["brand"] == last["brand"]
            )
        )
        if (
            last is not None
            and last["style"] == item["style"]
            and same_brand
            and item["start"] - last["end"] <= 1.0
        ):
            last["end"] = max(last["end"], item["end"])
            last["brand"] = last["brand"] or item["brand"]
        else:
            merged.append(dict(item))

    events = []
    for item in merged:
        events.append(
            {
                **ad,
                "start_s": round(item["start"], 3),
                "end_s": round(item["end"], 3),
                "ad_production_style": item["style"],
                "event_granularity": EVENT_GRANULARITY,
            }
        )
    return events


def split_annotation_row(
    row: dict,
    transcript: dict | None,
    min_duration_s: float = 5.0,
) -> tuple[dict, int, int]:
    """Split every ad in an annotation row; return (row, before, after)."""
    ads = list(row.get("ads") or [])
    if not transcript or not ads:
        return row, len(ads), len(ads)
    segments = list(transcript.get("segments") or [])
    results = dict(transcript.get("results") or {})
    events: list[dict] = []
    for ad in ads:
        events.extend(split_ad_events(ad, segments, results, min_duration_s))
    events.sort(key=lambda item: float(item["start_s"]))
    return {**row, "ads": events}, len(ads), len(events)
