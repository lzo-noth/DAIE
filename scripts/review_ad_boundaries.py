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


def _identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("feed") or "unknown"),
        str(row.get("episode") or ""),
        str(row.get("source_sha256") or ""),
    )


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
                # 复核结果必须绑定这份音频字节，否则同一集的另一个 DAI
                # 变体可能套用为它人工确认过的边界。
                "source_sha256": row.get("source_sha256"),
                "audio_id": row.get("audio_id"),
                "duration_s": row.get("duration_s"),
                "current_labels_source": row.get("labels_source") or labels_source,
                "ads": ads,
                "reviewer": "",
                "reviewed_at": "",
                "listened": False,
                "no_ads_confirmed": False,
                "ambiguous_boundary": False,
            }
        )
    return worklist


def _verify_review(review: dict, row: dict) -> tuple[list[dict], list[str]]:
    """Validate one episode review against the audio it claims to describe."""
    problems: list[str] = []
    if not str(review.get("reviewer") or "").strip():
        problems.append("缺少 reviewer")
    if not str(review.get("reviewed_at") or "").strip():
        problems.append("缺少 reviewed_at")
    if review.get("listened") is not True:
        problems.append("未标记听过整集音频")

    row_sha = str(row.get("source_sha256") or "")
    review_sha = str(review.get("source_sha256") or "")
    if not review_sha:
        problems.append("复核结果缺少 source_sha256，无法证明绑定的是这份音频")
    elif review_sha != row_sha:
        problems.append(
            f"source_sha256 不匹配（复核 {review_sha[:16]}… vs 音频 {row_sha[:16]}…）"
        )
    row_audio_id = str(row.get("audio_id") or "")
    review_audio_id = str(review.get("audio_id") or "")
    if row_audio_id and review_audio_id and row_audio_id != review_audio_id:
        problems.append("audio_id 不匹配，复核对应的是另一条录音")

    if review.get("ambiguous_boundary") is True:
        problems.append("标记为模糊边界，不能直接作为人工作真值")

    raw_ads = list(review.get("ads") or [])
    if not raw_ads:
        # 无广告集同样是误报率的分母，必须显式确认"整集确实没有广告"。
        if review.get("no_ads_confirmed") is not True:
            problems.append("空广告集需要 no_ads_confirmed=true 的整集确认")

    duration = float(row.get("duration_s") or review.get("duration_s") or 0.0)
    verified: list[dict] = []
    for ad in raw_ads:
        start = ad.get("start_s")
        end = ad.get("end_s")
        if start is None or end is None:
            problems.append("存在未填写的边界")
            continue
        start, end = float(start), float(end)
        if end <= start:
            problems.append(f"边界非法 {start}-{end}")
            continue
        if duration > 0 and (start < 0 or end > duration + 0.5):
            problems.append(f"边界 {start}-{end} 超出音频时长 {duration:.1f}s")
            continue
        if ad.get("start_verified") is not True or ad.get("end_verified") is not True:
            problems.append(f"边界 {start}-{end} 未逐条确认")
            continue
        if ad.get("ambiguous_boundary") is True:
            problems.append(f"边界 {start}-{end} 被标记为模糊边界")
            continue
        verified.append(
            {
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "type": str(ad.get("type") or "third_party_ad"),
                "ad_content_type": str(ad.get("ad_content_type") or "third_party_ad"),
                "ad_production_style": str(ad.get("ad_production_style") or "unknown"),
                "insertion_mechanism": str(ad.get("insertion_mechanism") or "unknown"),
                "label_method": "human_audio_review",
                "start_verified": True,
                "end_verified": True,
                "ambiguous_boundary": False,
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
    """Merge reviewed spans; only fully verified episodes claim ``human``.

    Reviews are matched by episode first and then by audio hash: a review that
    was written for another DAI variant of the same episode must surface as a
    rejection instead of silently disappearing.
    """
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for review in reviews:
        by_pair.setdefault((*_identity(review)[:2],), []).append(review)
    applied: list[dict] = []
    rejected: list[dict] = []
    for row in rows:
        key = _identity(row)
        candidates = by_pair.get(key[:2], [])
        if not candidates:
            continue
        review = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("source_sha256") or "") == key[2]
            ),
            candidates[0],
        )
        ads, problems = _verify_review(review, row)
        if problems:
            rejected.append(
                {
                    "feed": key[0],
                    "episode": key[1],
                    "source_sha256": key[2],
                    "problems": problems,
                }
            )
            continue
        row["ads"] = ads
        row["labels_source"] = labels_source
        row["boundary_review"] = {
            "reviewer": str(review.get("reviewer") or ""),
            "reviewed_at": str(review.get("reviewed_at") or ""),
            "source_sha256": key[2],
            "listened": True,
            "no_ads_confirmed": bool(review.get("no_ads_confirmed") or False),
            "ambiguous_boundary": False,
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
            _identity(row): row
            for row in _load_jsonl(args.blind_annotations or Path("/nonexistent"))
        }
        for item in _load_jsonl(args.blind_generated):
            annotation = annotations.get(_identity(item))
            if annotation is None:
                continue
            gt_path = args.ground_truth_dir / Path(
                str(item.get("ground_truth") or "")
            ).name
            if not gt_path.exists():
                continue
            payload = json.loads(gt_path.read_text(encoding="utf-8"))
            # 真值文件与标注必须是同一份音频字节，否则人工边界会被写到
            # 另一个 DAI 变体的真值上。
            if str(payload.get("source_sha256") or "") != str(
                annotation.get("source_sha256") or ""
            ):
                continue
            payload["ads"] = annotation.get("ads") or []
            payload["labels_source"] = annotation.get("labels_source")
            payload["boundary_review"] = annotation.get("boundary_review")
            gt_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    rejected = sum(len(item["rejected"]) for item in results)
    return 2 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
