#!/usr/bin/env python3
"""Mine sponsor/brand terms from LLM-labelled ad transcripts.

The result is used by scripts/asr_clips.py as a deterministic (non-LLM) ad
language spotter on the next round.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE_TERMS = [
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

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "you", "your", "have", "from",
    "are", "was", "were", "will", "can", "just", "like", "about", "there",
    "they", "what", "when", "where", "which", "who", "how", "not", "but", "all",
    "our", "out", "one", "two", "get", "got", "now", "new", "more", "most",
    "some", "any", "its", "it's", "i'm", "don't", "didn't", "doesn't", "isn't",
    "yeah", "okay", "right", "well", "really", "actually", "going", "gonna",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9.+'\-]{2,}")


def _tokens(text: str) -> set[str]:
    words = TOKEN_RE.findall(text.lower())
    words = [word.strip(".'-") for word in words]
    words = [word for word in words if word and word not in STOPWORDS]
    phrases = set(words)
    for size in (2, 3):
        for index in range(len(words) - size + 1):
            phrase = " ".join(words[index:index + size])
            if all(word not in STOPWORDS for word in words[index:index + size]):
                phrases.add(phrase)
    return phrases


GENERIC_BRAND_WORDS = {
    "plus", "max", "com", "www", "http", "https", "app", "tv", "one", "two",
    "zero", "episode", "podcast", "show", "today", "year", "years", "time",
    "people", "thing", "things", "way", "lot", "little", "good", "great",
    "much", "many", "back", "first", "last", "next", "other", "same", "own",
    "networks", "network", "media", "group", "systems", "system", "solutions",
    "technologies", "technology", "services", "products", "company", "brand",
    "studio", "music", "news", "sport", "sports", "games", "food", "travel",
    "hotel", "hotels", "health", "healthcare", "financial", "capital", "bank",
    "insurance", "energy", "university", "college", "school", "institute",
    "center", "centre", "foundation", "partners", "ventures", "labs", "lab",
    "cloud", "data", "digital", "global", "national", "international",
    "american", "world", "daily", "weekly", "review", "report", "press",
}


def _brands(text: str) -> set[str]:
    pieces = re.split(r"[,/&|()]+", text.lower())
    brands = set()
    for piece in pieces:
        cleaned = " ".join(TOKEN_RE.findall(piece)).strip()
        if len(cleaned) >= 3:
            brands.add(cleaned)
            brands.update(
                word
                for word in cleaned.split()
                if len(word) >= 5 and word not in GENERIC_BRAND_WORDS
            )
    return {brand for brand in brands if brand not in GENERIC_BRAND_WORDS}


def load_rows(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def mine(rows: list[dict], existing: dict, max_terms: int = 240) -> dict:
    ads = [row for row in rows if row.get("is_ad")]
    non_ads = [row for row in rows if not row.get("is_ad")]
    ad_counts: dict[str, int] = {}
    non_counts: dict[str, int] = {}
    for row in ads:
        for term in _tokens(str(row.get("asr_text") or "")):
            ad_counts[term] = ad_counts.get(term, 0) + 1
        for term in (row.get("ad_terms") or []):
            ad_counts[str(term).lower()] = ad_counts.get(str(term).lower(), 0) + 1
    for row in non_ads:
        for term in _tokens(str(row.get("asr_text") or "")):
            non_counts[term] = non_counts.get(term, 0) + 1

    discovered: dict[str, float] = {}
    for term, count in ad_counts.items():
        if len(term) < 4 or term in STOPWORDS:
            continue
        non_count = non_counts.get(term, 0)
        odds = (count + 1) / (non_count + 1)
        if count >= 2 and odds >= 3.0:
            discovered[term] = odds
    brands: set[str] = set()
    for row in ads:
        llm = row.get("llm") or {}
        brands.update(_brands(str(llm.get("brand") or "")))
        brands.update(_brands(str(row.get("asr_text") or "")) if row.get("ad_language") else set())
    existing_terms = [str(term) for term in existing.get("terms") or []]
    existing_brands = [str(term) for term in existing.get("brands") or []]

    ranked = sorted(discovered.items(), key=lambda item: (-item[1], item[0]))
    terms = list(dict.fromkeys([*BASE_TERMS, *existing_terms, *[term for term, _ in ranked], *sorted(brands)]))
    terms = [term for term in terms if len(term) >= 3 and term not in GENERIC_BRAND_WORDS][:max_terms]
    all_brands = sorted(set(existing_brands) | brands)
    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": len(rows),
        "ads": len(ads),
        "non_ads": len(non_ads),
        "terms": terms,
        "brands": all_brands,
        "discovered": {term: round(odds, 2) for term, odds in ranked[:120]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=Path("feedback/labels.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("feedback/keywords.json"))
    parser.add_argument("--max-terms", type=int, default=240)
    args = parser.parse_args()
    rows = load_rows(args.store)
    existing = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    payload = mine(rows, existing, args.max_terms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "ads": payload["ads"],
                "non_ads": payload["non_ads"],
                "terms": len(payload["terms"]),
                "brands": payload["brands"][:12],
                "wrote": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
