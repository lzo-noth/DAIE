#!/usr/bin/env python3
"""Classify candidate ASR transcripts as ads/content with an LLM."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.llm import apply_classifications, classify_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keep-weak", action="store_true", help="保留 LLM 未确认且无 slot 的插入候选")
    parser.add_argument("--keep-host-read", action="store_true", help="保留主播口播广告")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not gemini_key and not openai_key:
        print("LLM key not configured; skipping ad classification")
        return 0

    if gemini_key:
        api_key = gemini_key
        base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = args.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    else:
        api_key = openai_key
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    from openai import OpenAI

    report = json.loads(args.report.read_text(encoding="utf-8"))
    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        results = classify_items(report, client, model, batch_size=args.batch_size)
        apply_classifications(
            report,
            results,
            drop_content=not args.keep_weak,
            drop_host_read=not args.keep_host_read,
        )
        report.setdefault("llm", {})["model"] = model
        report.setdefault("llm", {})["classified"] = len(results)
        print(
            f"LLM classified={len(results)} "
            f"ads={report['llm'].get('verified_third_party_ads', 0)} "
            f"host_read={report['llm'].get('host_read', 0)}"
        )
    except Exception as exc:  # noqa: BLE001 - keep the analysis artifact usable
        report.setdefault("llm", {})["error"] = f"{type(exc).__name__}: {exc}"[:500]
        print(f"LLM classification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
