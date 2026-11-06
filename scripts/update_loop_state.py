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
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--target-precision", type=float, default=0.3)
    parser.add_argument("--target-recall", type=float, default=0.8)
    parser.add_argument("--target-auc", type=float, default=0.85)
    parser.add_argument("--target-fp-per-hour", type=float, default=5.0)
    args = parser.parse_args()

    metrics = _read_json(args.metrics)
    learned = metrics.get("learned") or {}
    precision = float(learned.get("precision") or 0.0)
    recall = float(learned.get("recall") or 0.0)
    auc = float(learned.get("auc") or 0.0)
    fp_per_hour = metrics.get("fp_per_hour")
    checks = {
        "precision": precision >= args.target_precision,
        "recall": recall >= args.target_recall,
        "auc": auc >= args.target_auc,
    }
    if fp_per_hour is not None:
        checks["fp_per_hour"] = float(fp_per_hour) <= args.target_fp_per_hour
    complete = bool(metrics) and all(checks.values())
    stopped = (not complete) and args.round >= args.max_rounds

    state = {
        "version": 1,
        "round": args.round,
        "max_rounds": args.max_rounds,
        "complete": complete,
        "stopped": stopped,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "targets": {
            "precision": args.target_precision,
            "recall": args.target_recall,
            "auc": args.target_auc,
            "fp_per_hour": args.target_fp_per_hour,
        },
        "checks": checks,
        "metrics": metrics,
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
            f"stopped at max round {args.round}: "
            f"precision={precision:.3f} recall={recall:.3f} auc={auc:.3f} fp_per_hour={fp_per_hour}\n",
            encoding="utf-8",
        )
    print(json.dumps({"round": args.round, "complete": complete, "stopped": stopped, "checks": checks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
