#!/usr/bin/env python3
"""Batch blind test across feeds and episodes.

Each manifest line points at one audio file and its ground truth::

    {"audio": "...", "feed": "Show A", "episode": "Ep 1", "ground_truth": "..."}

The runner calls ``scripts/blind_test.py`` per episode so identity checks,
strict calibration checks and independence checks are identical to the
single-episode path. It then aggregates TP/FP/FN, boundary errors and FP/hour
globally and per feed. Performance targets are only marked met when every row
is an independent unseen-feed test.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND_TEST = ROOT / "scripts" / "blind_test.py"

ACCEPTANCE = {
    "event_precision": 0.90,
    "event_recall": 0.90,
    "boundary_p95_s": 0.50,
    "false_positives_per_hour": 15.0,
}


def _wilson(successes: int, total: int) -> dict:
    if total <= 0:
        return {"estimate": None, "lower": None, "upper": None}
    ratio = successes / total
    z = 1.96
    denominator = 1.0 + z * z / total
    center = (ratio + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(ratio * (1.0 - ratio) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "estimate": round(ratio, 4),
        "lower": round(max(0.0, center - margin), 4),
        "upper": round(min(1.0, center + margin), 4),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return round(float(ordered[low]), 3)
    weight = index - low
    return round(float(ordered[low] * (1.0 - weight) + ordered[high] * weight), 3)


def _aggregate(reports: list[dict]) -> dict:
    tp = fp = fn = predicted = ground_truth = 0
    total_duration = 0.0
    boundary_errors: list[float] = []
    label_sources: set[str] = set()
    feeds: dict[str, dict] = {}
    for report in reports:
        metrics = report["metrics"]
        feed = report.get("feed") or "unknown"
        label_sources.add(str(report.get("labels_source") or "unverified"))
        bucket = feeds.setdefault(
            feed,
            {
                "episodes": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "duration_s": 0.0,
                "boundary_errors": [],
                "independent": True,
            },
        )
        matched = int(metrics.get("matched") or 0)
        false_positives = int(metrics.get("false_positives") or 0)
        false_negatives = int(metrics.get("false_negatives") or 0)
        tp += matched
        fp += false_positives
        fn += false_negatives
        predicted += int(metrics.get("predicted_events") or 0)
        ground_truth += int(metrics.get("ground_truth_events") or 0)
        duration = float(report.get("duration_s") or 0.0)
        total_duration += duration
        bucket["episodes"] += 1
        bucket["tp"] += matched
        bucket["fp"] += false_positives
        bucket["fn"] += false_negatives
        bucket["duration_s"] += duration
        bucket["independent"] = bucket["independent"] and bool(
            (report.get("independence") or {}).get("independent")
        )
        for match in report.get("matches") or []:
            boundary_errors.extend(
                (float(match["start_error"]), float(match["end_error"]))
            )
            bucket["boundary_errors"].extend(
                (float(match["start_error"]), float(match["end_error"]))
            )

    precision = tp / predicted if predicted else None
    recall = tp / ground_truth if ground_truth else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    per_feed = {}
    for feed, bucket in sorted(feeds.items()):
        feed_pred = bucket["tp"] + bucket["fp"]
        feed_gt = bucket["tp"] + bucket["fn"]
        feed_precision = bucket["tp"] / feed_pred if feed_pred else None
        feed_recall = bucket["tp"] / feed_gt if feed_gt else None
        per_feed[feed] = {
            "episodes": bucket["episodes"],
            "true_positives": bucket["tp"],
            "false_positives": bucket["fp"],
            "false_negatives": bucket["fn"],
            "event_precision": round(feed_precision, 4) if feed_precision is not None else None,
            "event_recall": round(feed_recall, 4) if feed_recall is not None else None,
            "false_positives_per_hour": (
                round(bucket["fp"] / (bucket["duration_s"] / 3600.0), 3)
                if bucket["duration_s"] > 0
                else None
            ),
            "boundary_p50_s": _percentile(bucket["boundary_errors"], 50),
            "boundary_p90_s": _percentile(bucket["boundary_errors"], 90),
            "boundary_p95_s": _percentile(bucket["boundary_errors"], 95),
            "independent": bucket["independent"],
        }
    return {
        "episodes": len(reports),
        "feeds": len(feeds),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_events": predicted,
        "ground_truth_events": ground_truth,
        "event_precision": round(precision, 4) if precision is not None else None,
        "event_recall": round(recall, 4) if recall is not None else None,
        "event_f1": round(f1, 4) if f1 is not None else None,
        "precision_ci95": _wilson(tp, tp + fp),
        "recall_ci95": _wilson(tp, tp + fn),
        "false_positives_per_hour": (
            round(fp / (total_duration / 3600.0), 3) if total_duration > 0 else None
        ),
        "boundary_p50_s": _percentile(boundary_errors, 50),
        "boundary_p90_s": _percentile(boundary_errors, 90),
        "boundary_p95_s": _percentile(boundary_errors, 95),
        "per_feed": per_feed,
        "all_independent": all(bucket["independent"] for bucket in feeds.values()),
        "labels_sources": sorted(label_sources),
        "all_human_labels": label_sources == {"human"},
    }


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dai-batch"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        handle.write(response.read())


def _run_row(row: dict, args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="dai-batch-") as directory:
        metrics_path = Path(directory) / "metrics.json"
        ground_truth = str(row["ground_truth"])
        if ground_truth.startswith(("http://", "https://")):
            local_truth = Path(directory) / "ground_truth.json"
            _download(ground_truth, local_truth)
            ground_truth = str(local_truth)
        command = [
            sys.executable,
            str(BLIND_TEST),
            "--audio",
            str(row["audio"]),
            "--ground-truth",
            ground_truth,
            "--feed",
            str(row.get("feed") or ""),
            "--metrics",
            str(metrics_path),
            "--quiet",
            "--max-fp-per-hour",
            str(args.max_fp_per_hour),
        ]
        if args.model:
            command += ["--model", str(args.model)]
        if args.boundary_model:
            command += ["--boundary-model", str(args.boundary_model)]
        if args.require_models:
            command.append("--require-models")
        if args.require_calibrated_models:
            command.append("--require-calibrated-models")
        if args.require_independent_test:
            command.append("--require-independent-test")
        if args.allow_training_overlap:
            command.append("--allow-training-overlap")
        if args.allow_audio_mismatch:
            command.append("--allow-audio-mismatch")
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            return {
                "feed": str(row.get("feed") or ""),
                "episode": str(row.get("episode") or ""),
                "audio": str(row["audio"]),
                "error": (process.stderr or process.stdout).strip()[-800:],
                "returncode": process.returncode,
            }
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        report["feed"] = str(row.get("feed") or "")
        report["episode"] = str(row.get("episode") or "")
        report["duration_s"] = (
            report.get("metrics", {}).get("duration_s")
            or report.get("duration_s")
            or 0.0
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="批量盲测清单 JSONL")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--boundary-model", type=Path)
    parser.add_argument("--out", type=Path, help="汇总报告 JSON")
    parser.add_argument("--reports-out-dir", type=Path, help="逐集盲测报告目录")
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-calibrated-models", action="store_true")
    parser.add_argument("--require-independent-test", action="store_true")
    parser.add_argument("--allow-training-overlap", action="store_true")
    parser.add_argument("--allow-audio-mismatch", action="store_true")
    parser.add_argument("--max-fp-per-hour", type=float, default=ACCEPTANCE["false_positives_per_hour"])
    parser.add_argument("--min-events", type=int, default=10)
    parser.add_argument("--min-feeds", type=int, default=3)
    parser.add_argument(
        "--allow-nonhuman-ground-truth",
        action="store_true",
        help="允许 LLM_weak/unverified 真值跑通流程，但不写 performance_targets_met=true",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reports = [_run_row(row, args) for row in rows]
    if args.reports_out_dir:
        args.reports_out_dir.mkdir(parents=True, exist_ok=True)
        for index, report in enumerate(reports):
            safe = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in f"{report.get('feed', 'unknown')}_{report.get('episode', index)}"
            )
            (args.reports_out_dir / f"{safe}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    failures = [report for report in reports if "error" in report]
    successful = [report for report in reports if "error" not in report]
    if not successful:
        summary = {
            "manifest": str(args.manifest),
            "episodes": len(rows),
            "failed_episodes": failures,
            "passed": False,
            "error": "没有成功的盲测行",
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2

    aggregate = _aggregate(successful)
    checks = {
        "event_precision": (
            aggregate["event_precision"] is not None
            and aggregate["event_precision"] >= ACCEPTANCE["event_precision"]
        ),
        "event_recall": (
            aggregate["event_recall"] is not None
            and aggregate["event_recall"] >= ACCEPTANCE["event_recall"]
        ),
        "boundary_p95_s": (
            aggregate["boundary_p95_s"] is not None
            and aggregate["boundary_p95_s"] <= ACCEPTANCE["boundary_p95_s"]
        ),
        "false_positives_per_hour": (
            args.max_fp_per_hour <= 0
            or (
                aggregate["false_positives_per_hour"] is not None
                and aggregate["false_positives_per_hour"] <= args.max_fp_per_hour
            )
        ),
        "minimum_events": aggregate["ground_truth_events"] >= args.min_events,
        "minimum_feeds": aggregate["feeds"] >= args.min_feeds,
        "all_independent": bool(aggregate["all_independent"]),
        "human_ground_truth": bool(
            aggregate["all_human_labels"] or args.allow_nonhuman_ground_truth
        ),
        "no_failed_episodes": not failures,
    }
    passed = all(checks.values())
    summary = {
        "manifest": str(args.manifest),
        "episodes": len(rows),
        "successful_episodes": len(successful),
        "failed_episodes": failures,
        "metrics": aggregate,
        "checks": checks,
        "passed": passed,
        "performance_targets_met": bool(
            passed
            and aggregate["all_human_labels"]
            and aggregate["all_independent"]
        ),
        "performance_targets_source": (
            "independent_human_ground_truth_batch"
            if passed and aggregate["all_human_labels"]
            else "not_independent_or_non_human_labels"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
