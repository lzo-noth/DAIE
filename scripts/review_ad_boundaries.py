#!/usr/bin/env python3
"""Human audio boundary review: worklist out, reviewed spans back in.

The release labels come from ASR segment timestamps, so a span marked
``human`` can still start at the first *word* of an ad instead of the actual
splice point. This script separates the two claims:

  * ``--worklist`` exports every candidate ad with its original AI start/end,
    so a reviewer can open the audio, listen around the boundaries and write
    back exact times.
  * ``--apply`` merges a reviewed file and records how the boundary was
    established. ``labels_source`` only becomes ``human`` when every ad in the
    episode carries ``label_method=human_audio_review`` and both edges are
    explicitly verified.

Review file format (one JSON object per line)::

    {
      "feed": "...", "episode": "...",
      "reviewer": "name", "reviewed_at": "2026-11-09T...",
      "listened": true,
      "ambiguous_boundary": false,
      "ads": [
        {"start_s": 120.0, "end_s": 151.42, "type": "third_party_inserted",
         "start_verified": true, "end_verified": true}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))


def build_worklist(rows: list[dict], labels_source: str) -> list[dict]:
    """Export candidate spans with the identity needed to find the audio again."""
    worklist: list[dict] = []
    for row in rows:
        if str(row.get("labels_source") or "") == "human":
            continue
        ads = [
            {
                "candidate_start_s": float(ad.get("start_s") or 0.0),
                "candidate_end_s": float(ad.get("end_s") or 0.0),
                "candidate_type": str(ad.get("type") or ""),
                "start_s": None,
                "end_s": None,
            }
            for ad in (row.get("ads") or [])
        ]
        worklist.append(
            {
                "feed": str(row.get("feed") or "unknown"),
                "episode": str(row.get("episode") or ""),
                "audio": row.get("audio"),
                "source_sha256": row.get("source_sha256"),
                "audio_id": row.get("audio_id"),
                "duration_s": row.get("duration_s"),
                "current_labels_source": row.get("labels_source") or labels_source,
                "ads": ads,
                "reviewer": "",
                "reviewed_at": "",
                "listened": False,
                "ambiguous_boundary": False,
            }
        )
    return worklist


def _verified_ads(review: dict) -> tuple[list[dict], list[str]]:
    problems: list[str] = []
    if review.get("listened") is not True:
        problems.append("未标记听过边界前后音频")
    verified: list[dict] = []
    for ad in review.get("ads") or []:
        start = ad.get("start_s")
        end = ad.get("end_s")
        if start is None or end is None:
            problems.append("存在未填写的边界")
            continue
        if float(end) <= float(start):
            problems.append(f"边界非法 {start}-{end}")
            continue
        if ad.get("start_verified") is not True or ad.get("end_verified") is not True:
            problems.append(f"边界 {start}-{end} 未逐条确认")
            continue
        verified.append(
            {
                "start_s": round(float(start), 3),
                "end_s": round(float(end), 3),
                "type": str(ad.get("type") or "third_party_inserted"),
                "label_method": "human_audio_review",
                "start_verified": True,
                "end_verified": True,
                "ambiguous_boundary": bool(
                    ad.get("ambiguous_boundary") or review.get("ambiguous_boundary")
                ),
                "candidate_start_s": ad.get("candidate_start_s"),
                "candidate_end_s": ad.get("candidate_end_s"),
                "reviewer": str(review.get("reviewer") or ""),
                "reviewed_at": str(review.get("reviewed_at") or ""),
            }
        )
    return verified, problems


def apply_review(
    rows: list[dict], reviews: list[dict], labels_source: str = "human"
) -> tuple[list[dict], list[dict]]:
    """Merge reviewed spans; only fully verified episodes claim ``human``."""
    by_key = {_key(review): review for review in reviews}
    applied: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        key = _key(row)
        review = by_key.get(key)
        if review is None:
            continue
        ads, problems = _verified_ads(review)
        if problems:
            rejected.append(
                {"feed": key[0], "episode": key[1], "problems": problems}
            )
            continue
        row["ads"] = ads
        row["labels_source"] = labels_source
        row["boundary_review"] = {
            "reviewer": str(review.get("reviewer") or ""),
            "reviewed_at": str(review.get("reviewed_at") or ""),
            "listened": True,
            "ambiguous_boundary": bool(review.get("ambiguous_boundary") or False),
            "labels_source": labels_source,
        }
        applied.append({"feed": key[0], "episode": key[1], "ads": len(ads)})
    return rows, rejected


def _review_pass(rows_path: Path, review_path: Path, labels_source: str) -> dict:
    rows = _load_jsonl(rows_path)
    reviews = _load_jsonl(review_path)
    updated, rejected = apply_review(rows, reviews, labels_source)
    _write_jsonl(rows_path, updated)
    return {
        "annotations": str(rows_path),
        "reviewed_episodes": len(updated),
        "rejected": rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, help="训练标注 JSONL")
    parser.add_argument("--blind-annotations", type=Path, help="盲测标注 JSONL")
    parser.add_argument("--ground-truth-dir", type=Path, help="盲测真值目录")
    parser.add_argument("--blind-generated", type=Path, help="盲测清单")
    parser.add_argument("--worklist-out", type=Path, help="导出待人工复核清单")
    parser.add_argument("--apply", type=Path, help="人工复核结果 JSONL")
    parser.add_argument(
        "--labels-source",
        default="human",
        help="复核通过后写入的标签来源（默认 human）",
    )
    parser.add_argument(
        "--now",
        default="",
        help="覆盖 reviewed_at（测试用）",
    )
    args = parser.parse_args()

    if args.worklist_out:
        train = _load_jsonl(args.annotations) if args.annotations else []
        blind = _load_jsonl(args.blind_annotations) if args.blind_annotations else []
        worklist = build_worklist(train, "ai_reviewed") + build_worklist(blind, "ai_reviewed")
        _write_jsonl(args.worklist_out, worklist)
        print(
            json.dumps(
                {
                    "worklist": str(args.worklist_out),
                    "episodes": len(worklist),
                    "ads": sum(len(item["ads"]) for item in worklist),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    if not args.apply:
        print("需要 --worklist-out 或 --apply", flush=True)
        return 1
    results: list[dict] = []
    if args.annotations and args.annotations.exists():
        results.append(_review_pass(args.annotations, args.apply, args.labels_source))
    if args.blind_annotations and args.blind_annotations.exists():
        results.append(
            _review_pass(args.blind_annotations, args.apply, args.labels_source)
        )
    if args.ground_truth_dir and args.ground_truth_dir.is_dir() and args.blind_generated:
        annotations = {
            _key(row): row
            for row in _load_jsonl(args.blind_annotations or Path("/nonexistent"))
        }
        for item in _load_jsonl(args.blind_generated):
            annotation = annotations.get(_key(item))
            if annotation is None:
                continue
            gt_path = args.ground_truth_dir / Path(
                str(item.get("ground_truth") or "")
            ).name
            if not gt_path.exists():
                continue
            payload = json.loads(gt_path.read_text(encoding="utf-8"))
            payload["ads"] = annotation.get("ads") or []
            payload["labels_source"] = annotation.get("labels_source")
            gt_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    rejected = sum(len(item["rejected"]) for item in results)
    return 2 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
