#!/usr/bin/env python3
"""Update feedback/state.json and stop markers after a self-improvement round."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("feedback/state.json"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration.json"))
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--target-precision", type=float, default=0.0)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--target-recall-at-k", type=int, default=40)
    parser.add_argument("--target-recall-at", type=float, default=0.95)
    parser.add_argument("--target-auc", type=float, default=0.8)
    parser.add_argument("--target-fp-per-hour", type=float, default=15.0)
    parser.add_argument("--target-candidates-per-hour", type=float, default=30.0)
    parser.add_argument("--ground-truth-metrics", type=Path)
    parser.add_argument("--target-event-precision", type=float, default=0.9)
    parser.add_argument("--target-event-recall", type=float, default=0.9)
    parser.add_argument("--target-boundary-p95", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    metrics = _read_json(args.metrics)
    previous = _read_json(args.state)
    active = _read_json(args.calibration)
    latest = metrics.get("latest") or {}
    learned = metrics.get("learned") or {}
    latest_learned = latest.get("learned") or learned
    precision = float(learned.get("precision") or 0.0)
    recall = float(latest_learned.get("recall") or learned.get("recall") or 0.0)
    auc = float(learned.get("auc") or 0.0)
    fp_per_hour = latest.get("fp_per_hour", metrics.get("fp_per_hour"))
    candidates_per_hour = latest.get("candidates_per_hour", metrics.get("candidates_per_hour"))
    recall_at = latest.get("recall_at") or metrics.get("recall_at") or {}
    recall_at_value = float(recall_at.get(str(args.target_recall_at_k)) or 0.0)
    latest_ads = int(latest.get("ads") or 0)
    checks = {
        "auc": auc >= args.target_auc,
    }
    if latest_ads > 0:
        checks["recall"] = recall >= args.target_recall
        checks["recall_at"] = recall_at_value >= args.target_recall_at
        if args.target_precision > 0:
            checks["precision"] = (
                float(latest_learned.get("precision") or precision) >= args.target_precision
            )
    if fp_per_hour is not None:
        checks["fp_per_hour"] = float(fp_per_hour) <= args.target_fp_per_hour
    if candidates_per_hour is not None:
        checks["candidates_per_hour"] = float(candidates_per_hour) <= args.target_candidates_per_hour

    ground_truth = _read_json(args.ground_truth_metrics) if args.ground_truth_metrics else {}
    gt_checks = {}
    if ground_truth and not ground_truth.get("error"):
        gt_checks["event_precision"] = (
            float(ground_truth.get("event_precision") or 0.0) >= args.target_event_precision
        )
        gt_checks["event_recall"] = (
            float(ground_truth.get("event_recall") or 0.0) >= args.target_event_recall
        )
        boundary_p95 = ground_truth.get("boundary_p95_s")
        gt_checks["boundary_p95"] = (
            boundary_p95 is not None and float(boundary_p95) <= args.target_boundary_p95
        )
    checks["ground_truth"] = bool(gt_checks) and all(gt_checks.values())
    complete = bool(metrics) and all(checks.values())

    active_metrics = active.get("metrics") or {}
    active_learned = active_metrics.get("learned") or {}
    previous_active = (previous.get("active_calibration") or {}).get("learned") or {}
    baseline_learned = previous_active or active_learned
    baseline_auc = float(baseline_learned.get("auc") or 0.0)
    baseline_precision = float(baseline_learned.get("precision") or 0.0)
    improved = auc >= baseline_auc + 0.005 or (
        auc >= baseline_auc and precision >= baseline_precision + 0.01
    )
    stale_rounds = 0 if improved else int(previous.get("stale_rounds") or 0) + 1
    stopped = (not complete) and (
        args.round >= args.max_rounds or stale_rounds >= args.patience
    )
    reason = ""
    if stopped:
        reason = (
            "max rounds" if args.round >= args.max_rounds else f"no improvement for {stale_rounds} rounds"
        )

    state = {
        "version": 1,
        "round": args.round,
        "max_rounds": args.max_rounds,
        "complete": complete,
        "stopped": stopped,
        "stop_reason": reason,
        "stale_rounds": stale_rounds,
        "patience": args.patience,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "targets": {
            "precision": args.target_precision,
            "recall": args.target_recall,
            "recall_at_k": args.target_recall_at_k,
            "recall_at": args.target_recall_at,
            "auc": args.target_auc,
            "fp_per_hour": args.target_fp_per_hour,
            "candidates_per_hour": args.target_candidates_per_hour,
        },
        "checks": checks,
        "metrics": metrics,
        "ground_truth": ground_truth,
        "active_calibration": {
            "enabled": active.get("enabled"),
            "mode": active.get("mode"),
            "learned": active_learned,
            "fp_per_hour": active_metrics.get("fp_per_hour"),
        },
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_marker = args.state.parent / "COMPLETE"
    stopped_marker = args.state.parent / "STOPPED"
    if complete:
        complete_marker.write_text(
            f"target reached at round {args.round}: "
            f"precision={precision:.3f} recall={recall:.3f} auc={auc:.3f} fp_per_hour={fp_per_hour}\n",
            encoding="utf-8",
        )
    elif stopped:
        stopped_marker.write_text(
            f"stopped at round {args.round} ({reason}): "
            f"precision={precision:.3f} recall={recall:.3f} auc={auc:.3f} fp_per_hour={fp_per_hour}\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "round": args.round,
                "complete": complete,
                "stopped": stopped,
                "stop_reason": reason,
                "stale_rounds": stale_rounds,
                "checks": checks,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
