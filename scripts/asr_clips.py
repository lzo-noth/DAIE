#!/usr/bin/env python3
"""Transcribe candidate clips and flag ad-like language.

Runs after dai-detect in Actions; never used during audio analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AD_TERMS = [
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
    "sign up",
    "percent off",
    "dot com",
    ".com",
    "本节目由",
    "赞助",
    "优惠码",
    "折扣",
    "广告",
]


def ad_terms(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in AD_TERMS if term in lowered})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--clips-dir", required=True, type=Path)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    report = json.loads(args.report.read_text(encoding="utf-8"))
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    total = 0
    confirmed = 0
    for episode in report.get("episodes", []):
        for candidate in episode.get("candidates", []):
            evidence = candidate.get("evidence", {})
            clip_name = evidence.get("clip")
            if not clip_name:
                continue
            clip_path = args.clips_dir / clip_name
            if not clip_path.exists():
                continue
            segments, _ = model.transcribe(
                str(clip_path),
                language=args.language,
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            terms = ad_terms(text)
            evidence["asr_text"] = text[:400]
            evidence["ad_terms"] = terms
            evidence["ad_language"] = bool(terms)
            if terms:
                candidate["score"] = round(min(0.99, float(candidate.get("score", 0.0)) + 0.06), 3)
                confirmed += 1
            total += 1
    report["asr"] = {"model": args.model, "language": args.language, "clips": total, "ad_language": confirmed}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ASR clips={total} ad_language={confirmed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
