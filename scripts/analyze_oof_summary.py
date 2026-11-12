#!/usr/bin/env python3
"""Attribute event-level false positives using per-episode OOF window counts.

Answers the question the event metrics cannot: are false positives spread over
every show (a feature-capacity problem) or concentrated in a few feeds (a
domain-shift / data problem)?

Usage:
    python scripts/analyze_oof_summary.py out/release/oof_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _thresholds(rows: list[dict]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update((row.get("thresholds") or {}).keys())
    return sorted(keys, key=float)


def _overall(rows: list[dict], thresholds: list[str]) -> None:
    print("窗口级（OOF 汇总）")
    print(f"{'阈值':>6} {'TP':>7} {'FP':>7} {'FN':>7} {'召回':>7} {'特异度':>8} {'FP/小时':>9}")
    for threshold in thresholds:
        tp = sum(int(row["thresholds"][threshold]["tp"]) for row in rows if threshold in row["thresholds"])
        fp = sum(int(row["thresholds"][threshold]["fp"]) for row in rows if threshold in row["thresholds"])
        fn = sum(int(row["thresholds"][threshold]["fn"]) for row in rows if threshold in row["thresholds"])
        negatives = sum(int(row["windows"]) - int(row["positive_windows"]) for row in rows)
        hours = sum(float(row["duration_s"]) for row in rows) / 3600.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        specificity = 1.0 - fp / negatives if negatives else float("nan")
        print(
            f"{float(threshold):6.2f} {tp:7d} {fp:7d} {fn:7d} "
            f"{recall:7.3f} {specificity:8.4f} {fp / hours if hours else float('nan'):9.2f}"
        )


def _per_feed(rows: list[dict], threshold: str, top: int) -> None:
    by_feed: dict[str, dict[str, float]] = defaultdict(lambda: {"fp": 0.0, "tp": 0.0, "hours": 0.0, "episodes": 0.0})
    for row in rows:
        counts = row["thresholds"].get(threshold)
        if not counts:
            continue
        entry = by_feed[str(row.get("feed") or "unknown")]
        entry["fp"] += int(counts["fp"])
        entry["tp"] += int(counts["tp"])
        entry["hours"] += float(row["duration_s"]) / 3600.0
        entry["episodes"] += 1
    total_fp = sum(entry["fp"] for entry in by_feed.values()) or 1.0
    ranked = sorted(by_feed.items(), key=lambda item: item[1]["fp"], reverse=True)
    print(f"\n按 feed 拆分（阈值 {threshold}，共 {len(by_feed)} 个 feed）")
    print(f"{'feed':34s} {'集数':>4} {'FP':>7} {'占比':>7} {'FP/小时':>9}")
    for name, entry in ranked[:top]:
        share = entry["fp"] / total_fp
        rate = entry["fp"] / entry["hours"] if entry["hours"] else float("nan")
        print(f"{name[:34]:34s} {int(entry['episodes']):4d} {int(entry['fp']):7d} {share:7.1%} {rate:9.2f}")
    for count in (3, 5, 10):
        share = sum(entry["fp"] for _, entry in ranked[:count]) / total_fp
        print(f"  前 {count:2d} 个 feed 占全部 FP 的 {share:.1%}")


def _per_episode(rows: list[dict], threshold: str, top: int) -> None:
    ranked = []
    for row in rows:
        counts = row["thresholds"].get(threshold)
        if not counts:
            continue
        hours = float(row["duration_s"]) / 3600.0
        rate = int(counts["fp"]) / hours if hours else 0.0
        ranked.append((rate, row))
    ranked.sort(key=lambda item: item[0], reverse=True)
    print(f"\n误报最多的单集（阈值 {threshold}）")
    for rate, row in ranked[:top]:
        print(
            f"  {rate:7.2f} FP/小时  {str(row.get('feed'))[:24]:24s} "
            f"{str(row.get('episode'))[:40]:40s} 窗口={row['windows']} 正例={row['positive_windows']}"
        )
    blank = [row for row in rows if int(row.get("positive_windows") or 0) == 0]
    if blank:
        rates = []
        for row in blank:
            counts = row["thresholds"].get(threshold) or {}
            hours = float(row["duration_s"]) / 3600.0
            rates.append(int(counts.get("fp") or 0) / hours if hours else 0.0)
        rates.sort()
        print(
            f"\n无广告集（{len(blank)} 集）FP/小时 中位数 "
            f"{rates[len(rates) // 2]:.2f}，最高 {rates[-1]:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    rows = _load(args.summary)
    if not rows:
        print("汇总文件为空")
        return 1
    thresholds = _thresholds(rows)
    print(f"集数={len(rows)} feed 数={len({row.get('feed') for row in rows})}")
    _overall(rows, thresholds)
    focus = "0.90" if "0.90" in thresholds else thresholds[-1]
    _per_feed(rows, focus, args.top)
    _per_episode(rows, focus, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
