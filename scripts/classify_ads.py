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


def _targets(report: Path | None, root: Path | None) -> list[Path]:
    if root is not None:
        return sorted(root.rglob("report.json"))
    if report is not None:
        return [report]
    return []


def _client_and_model(args):
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not gemini_key and not openai_key:
        return None, ""
    if gemini_key:
        base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        model = args.model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        api_key = gemini_key
    else:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        api_key = openai_key
    from openai import OpenAI

    client = (
        OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
        if base_url
        else OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
    )
    return client, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keep-weak", action="store_true", help="保留 LLM 未确认且无 slot 的插入候选")
    parser.add_argument("--keep-host-read", action="store_true", help="保留主播口播广告")
    parser.add_argument("--keep-repeated-content", action="store_true", help="保留 LLM 判为正片的重复候选")
    args = parser.parse_args()

    client, model = _client_and_model(args)
    if client is None:
        print("LLM key not configured; skipping ad classification")
        return 0
    targets = _targets(args.report, args.reports_root)
    if not targets:
        print("no report targets")
        return 0

    total_classified = 0
    total_ads = 0
    for report_path in targets:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"{report_path}: read failed {exc}")
            continue
        try:
            results = classify_items(report, client, model, batch_size=args.batch_size)
            apply_classifications(
                report,
                results,
                drop_content=not args.keep_weak,
                drop_host_read=not args.keep_host_read,
                keep_repeated_content=args.keep_repeated_content,
            )
            stats = report.setdefault("llm", {})
            stats["model"] = model
            stats["classified"] = len(results)
            total_classified += len(results)
            total_ads += int(stats.get("verified_third_party_ads", 0))
        except Exception as exc:  # noqa: BLE001 - keep the analysis artifact usable
            report.setdefault("llm", {})["error"] = f"{type(exc).__name__}: {exc}"[:500]
            print(f"{report_path.parent.name}: LLM failed {type(exc).__name__}: {exc}", file=sys.stderr)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"{report_path.parent.name}: classified={report.get('llm', {}).get('classified', 0)} "
            f"ads={report.get('llm', {}).get('verified_third_party_ads', 0)} "
            f"host_read={report.get('llm', {}).get('host_read', 0)}"
        )
    print(f"LLM total classified={total_classified} ads={total_ads} model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
