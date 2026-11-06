from __future__ import annotations

import json
import os
import re
from pathlib import Path

BASE_AD_TERMS = [
    "sponsored by",
    "brought to you by",
    "support for this podcast",
    "support comes from",
    "this episode is sponsored",
    "our sponsor",
    "advertisement",
    "ad break",
    "paid for by",
    "partner of",
    "use code",
    "promo code",
    "offer code",
    "discount code",
    "coupon code",
    "free trial",
    "terms apply",
    "percent off",
    "sign up",
    "subscribe",
    "本节目由",
    "赞助",
    "优惠码",
    "折扣",
    "广告",
]

_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def load_terms(path: str | Path | None = None) -> list[str]:
    target = Path(path or os.environ.get("DAI_KEYWORDS", "feedback/keywords.json"))
    learned: list[str] = []
    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            learned = [str(term).lower() for term in payload.get("terms") or [] if str(term).strip()]
        except (OSError, ValueError):
            learned = []
    return sorted(set(BASE_AD_TERMS) | set(learned))


def _single_word_pattern(term: str) -> re.Pattern[str]:
    pattern = _PATTERN_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        _PATTERN_CACHE[term] = pattern
    return pattern


def _hit(term: str, text: str) -> bool:
    variants = [term]
    if "-" in term:
        variants.append(term.replace("-", " "))
    for variant in variants:
        if " " in variant or "-" in variant or "." in variant:
            if variant in text:
                return True
        elif _single_word_pattern(variant).search(text):
            return True
    return False


def match_terms(text: str, terms: list[str]) -> list[str]:
    if not text:
        return []
    lowered = text.lower()
    return sorted({term for term in terms if _hit(term, lowered)})
