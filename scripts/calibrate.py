#!/usr/bin/env python3
"""Summarize LLM-labeled candidates and suggest acoustic gate parameters."""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path


def _rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for episode in report.get("episodes", []):
            for candidate in episode.get("candidates", []):
                evidence = candidate.get("evidence", {})
                llm = evidence.get("llm")
                if not llm:
                    continue
                rows.append(
                    {
                        "ad": bool(evidence.get("llm_verified")),
                        "musicness": float(evidence.get("musicness") or 0.0),
                        "loudness_dev": float(evidence.get("loudness_dev_db") or 0.0),
                        "timbre": float(evidence.get("timbre_min") or 0.0),
                        "boundary": float(evidence.get("boundary_strength") or 0.0),
                        "slot": int(evidence.get("slot_matches") or 0),
                        "duration": float(candidate.get("end_s", 0.0)) - float(candidate.get("start_s", 0.0)),
                        "score": float(candidate.get("score") or 0.0),
                    }
                )
    return rows


def _median(rows: list[dict], key: str) -> str:
    if not rows:
        return "-"
    values = [row[key] for row in rows]
    return f"min={min(values):.2f} med={stats.median(values):.2f} max={max(values):.2f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path, help="report.json 或包含 report.json 的目录")
    parser.add_argument("--out", type=Path, help="写入 calibration.json")
    args = parser.parse_args()
    paths = []
    for path in args.reports:
        paths.extend(path.rglob("report.json") if path.is_dir() else [path])
    rows = _rows(sorted(set(paths)))
    ads = [row for row in rows if row["ad"]]
    non_ads = [row for row in rows if not row["ad"]]
    print(f"samples={len(rows)} ads={len(ads)} non_ads={len(non_ads)}")
    for key in ("musicness", "loudness_dev", "timbre", "boundary", "slot", "duration", "score"):
        print(f"ads     {key:12} {_median(ads, key)}")
        print(f"non-ads {key:12} {_median(non_ads, key)}")

    def rule(row: dict) -> bool:
        return row["musicness"] >= 0.3 or row["loudness_dev"] >= 5.0

    true_positive = sum(1 for row in ads if rule(row))
    false_positive = sum(1 for row in non_ads if rule(row))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, len(ads))
    print(f"gate music>=0.3 or loud>=5.0: precision={precision:.2f} recall={recall:.2f}")
    slot_ads = sum(1 for row in ads if row["slot"])
    slot_non = sum(1 for row in non_ads if row["slot"])
    print(f"slot matches: ads={slot_ads}/{len(ads)} non_ads={slot_non}/{len(non_ads)}")

    if args.out:
        payload = {
            "samples": len(rows),
            "ads": len(ads),
            "ad_evidence": {"musicness_min": 0.3, "loudness_dev_min": 5.0},
            "slot_boost": {"ad_like_only": True, "single": 0.05, "multi": 0.08},
            "observed": {
                "gate_precision": round(precision, 3),
                "gate_recall": round(recall, 3),
                "slot_ads": slot_ads,
                "slot_non_ads": slot_non,
            },
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
