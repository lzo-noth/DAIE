#!/usr/bin/env python3
"""Evaluate detector candidates against independent human ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ground_truth(root: Path) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            ads = item.get("ads") or []
            episode = item.get("episode") or item.get("title")
            episode_id = item.get("episode_id") or item.get("id")
            if ads and (episode or episode_id):
                entries.append(
                    {
                        "episode": episode,
                        "episode_id": episode_id,
                        "ads": [
                            {
                                "start_s": float(ad["start_s"]),
                                "end_s": float(ad["end_s"]),
                                "type": ad.get("type", "third_party_inserted"),
                            }
                            for ad in ads
                            if ad.get("start_s") is not None and ad.get("end_s") is not None
                        ],
                    }
                )
    return entries


def load_candidates(root: Path) -> dict[str, list[dict]]:
    by_episode: dict[str, list[dict]] = {}
    for path in sorted(root.rglob("report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for episode in report.get("episodes") or []:
            keys = [episode.get("title"), episode.get("id")]
            candidates = episode.get("candidates") or []
            for key in keys:
                if key:
                    by_episode[str(key)] = candidates
    return by_episode


def _overlap(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _iou(first_start: float, first_end: float, second_start: float, second_end: float) -> float:
    overlap = _overlap(first_start, first_end, second_start, second_end)
    union = max(first_end, second_end) - min(first_start, second_start)
    return overlap / union if union > 0 else 0.0


def evaluate(gt_entries: list[dict], candidates_by_episode: dict[str, list[dict]]) -> dict:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    boundary_errors: list[float] = []
    ious: list[float] = []
    per_episode: list[dict] = []
    for entry in gt_entries:
        candidates = (
            candidates_by_episode.get(str(entry.get("episode")))
            or candidates_by_episode.get(str(entry.get("episode_id")))
            or []
        )
        used: set[int] = set()
        episode_tp = 0
        episode_fn = 0
        for ad in entry["ads"]:
            best_index = -1
            best_iou = 0.0
            best_overlap = 0.0
            for index, candidate in enumerate(candidates):
                if index in used:
                    continue
                start = float(candidate.get("start_s", 0.0))
                end = float(candidate.get("end_s", 0.0))
                overlap = _overlap(start, end, ad["start_s"], ad["end_s"])
                iou = _iou(start, end, ad["start_s"], ad["end_s"])
                if overlap > best_overlap or (overlap == best_overlap and iou > best_iou):
                    best_index, best_iou, best_overlap = index, iou, overlap
            ad_duration = max(1e-6, ad["end_s"] - ad["start_s"])
            if best_index >= 0 and best_overlap / ad_duration >= 0.5:
                candidate = candidates[best_index]
                used.add(best_index)
                true_positive += 1
                episode_tp += 1
                start = float(candidate.get("start_s", 0.0))
                end = float(candidate.get("end_s", 0.0))
                boundary_errors.append(
                    (abs(start - ad["start_s"]) + abs(end - ad["end_s"])) / 2.0
                )
                ious.append(best_iou)
            else:
                false_negative += 1
                episode_fn += 1
        episode_fp = len(candidates) - len(used)
        false_positive += episode_fp
        per_episode.append(
            {
                "episode": entry.get("episode") or entry.get("episode_id"),
                "ads": len(entry["ads"]),
                "true_positive": episode_tp,
                "false_negative": episode_fn,
                "false_positive": episode_fp,
            }
        )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    boundary_p95 = (
        float(np.percentile(boundary_errors, 95)) if boundary_errors else None
    )
    return {
        "episodes": len(gt_entries),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "event_precision": round(precision, 3),
        "event_recall": round(recall, 3),
        "boundary_p95_s": round(boundary_p95, 3) if boundary_p95 is not None else None,
        "iou_mean": round(float(np.mean(ious)), 3) if ious else None,
        "per_episode": per_episode,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=Path("ground_truth"))
    parser.add_argument("--out", type=Path, default=Path("out/learn/ground_truth.json"))
    args = parser.parse_args()
    gt_entries = load_ground_truth(args.ground_truth)
    if not gt_entries:
        print("no human ground truth files; skip event evaluation")
        return 1
    metrics = evaluate(gt_entries, load_candidates(args.reports_root))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
