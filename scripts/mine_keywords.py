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

TOKEN_RE = re.compile(r"[a-z][a-z0-9.+'\-]{2,}")


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
        if not cleaned or len(cleaned) > 40 or len(cleaned.split()) > 4:
            continue
        brands.add(cleaned)
        if len(cleaned.split()) <= 2:
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


def mine(rows: list[dict], existing: dict, max_terms: int = 120) -> dict:
    ads = [row for row in rows if row.get("is_ad")]
    non_ads = [row for row in rows if not row.get("is_ad")]
    brands: set[str] = set()
    for row in ads:
        llm = row.get("llm") or {}
        brands.update(_brands(str(llm.get("brand") or "")))
        for term in (row.get("ad_terms") or []):
            cleaned = str(term).lower().strip()
            if cleaned in BASE_TERMS:
                brands.add(cleaned)
    # Only the LLM brand field and curated ad phrases are used as learned
    # keywords. Mining arbitrary n-grams from transcripts produced too many
    # false positives ("that's", "everything") in the first rounds.
    terms = list(dict.fromkeys([*BASE_TERMS, *sorted(brands)]))
    terms = [term for term in terms if len(term) >= 3 and term not in GENERIC_BRAND_WORDS][:max_terms]
    return {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": len(rows),
        "ads": len(ads),
        "non_ads": len(non_ads),
        "terms": terms,
        "brands": sorted(brands),
        "discovered": {},
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
