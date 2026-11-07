#!/usr/bin/env python3
"""Re-derive ad spans from reviewed transcripts and apply review decisions.

The first-pass labels came from ``merge_ad_segments``, which drops host-read
ads. Real podcast ad reads are mostly host-read, so that filter removed almost
every inserted ad (e.g. TWiT had 0 of 9 ad breaks kept). This script rebuilds
``ads`` from the stored per-segment LLM verdicts using the policy recorded in
``review_decisions.json``, then applies the explicit per-span drops that a text
review found (content mentions, sponsor disclosures, cross-promos).

It rewrites ``annotations.jsonl``, ``blind_annotations.jsonl``,
``blind_generated.jsonl`` and ``blind_gt/*.json`` in place. ``labels_source``
comes from the decisions file and must stay non-``human`` unless a person
actually reviewed the audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DECISIONS = Path("datasets/release_annotations/review_decisions.json")
MATCH_TOLERANCE_S = 0.6


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_ads(
    segments: list[dict],
    results: dict[str, dict],
    policy: dict,
    drop_starts: list[float],
) -> list[dict]:
    min_confidence = float(policy.get("min_confidence", 0.5))
    max_gap = float(policy.get("max_gap_s", 3.0))
    min_span = float(policy.get("min_span_s", 2.0))
    ad_types = set(policy.get("ad_types") or ["third_party_ad"])

    spans: list[list[float]] = []
    for index, segment in enumerate(segments):
        result = results.get(str(index)) or {}
        if result.get("is_ad") is not True:
            continue
        if str(result.get("ad_type") or "") not in ad_types:
            continue
        if float(result.get("confidence") or 0.0) < min_confidence:
            continue
        start = float(segment.get("start_s") or 0.0)
        end = float(segment.get("end_s") or 0.0)
        if spans and start - spans[-1][1] <= max_gap:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])

    ads = []
    for start, end in spans:
        if end - start < min_span:
            continue
        if any(abs(start - drop) <= MATCH_TOLERANCE_S for drop in drop_starts):
            continue
        ads.append(
            {
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "type": "third_party_inserted",
            }
        )
    return ads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/release_annotations"))
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    args = parser.parse_args()

    decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
    policy = decisions.get("policy") or {}
    labels_source = str(decisions.get("labels_source") or "ai_reviewed")
    drop_all = {
        (str(item.get("feed") or ""), str(item.get("episode") or ""))
        for item in decisions.get("drop_all") or []
    }
    drops: dict[tuple[str, str], list[float]] = {}
    for item in decisions.get("drops") or []:
        key = (str(item.get("feed") or ""), str(item.get("episode") or ""))
        drops.setdefault(key, []).append(float(item.get("start_s") or 0.0))

    stats = []
    for transcripts_name, annotations_name in (
        ("transcripts.jsonl", "annotations.jsonl"),
        ("blind_transcripts.jsonl", "blind_annotations.jsonl"),
    ):
        transcripts_path = args.data_dir / transcripts_name
        if not transcripts_path.exists():
            continue
        rebuilt = []
        for row in load_jsonl(transcripts_path):
            key = (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))
            if key in drop_all:
                ads = []
            else:
                ads = derive_ads(
                    list(row.get("segments") or []),
                    dict(row.get("results") or {}),
                    policy,
                    drops.get(key, []),
                )
            identity = {
                field: row.get(field)
                for field in (
                    "audio",
                    "feed",
                    "episode",
                    "source_sha256",
                    "audio_id",
                    "audio_id_kind",
                    "duration_s",
                )
            }
            rebuilt.append({**identity, "labels_source": labels_source, "ads": ads})
            stats.append(
                {
                    "feed": key[0],
                    "episode": key[1],
                    "ads": len(ads),
                    "ad_s": round(sum(a["end_s"] - a["start_s"] for a in ads), 1),
                }
            )
        write_jsonl(args.data_dir / annotations_name, rebuilt)

    generated_path = args.data_dir / "blind_generated.jsonl"
    gt_dir = args.data_dir / "blind_gt"
    if generated_path.exists() and gt_dir.is_dir():
        annotations = {
            (str(row.get("feed") or "unknown"), str(row.get("episode") or "")): row
            for row in load_jsonl(args.data_dir / "blind_annotations.jsonl")
        }
        rebuilt_generated = []
        for row in load_jsonl(generated_path):
            key = (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))
            annotation = annotations.get(key)
            if annotation is None:
                rebuilt_generated.append(row)
                continue
            gt_path = gt_dir / Path(str(row.get("ground_truth") or "")).name
            gt_path.write_text(
                json.dumps(
                    {
                        "duration_s": annotation.get("duration_s"),
                        "ads": annotation.get("ads") or [],
                        "source_sha256": annotation.get("source_sha256"),
                        "audio_id": annotation.get("audio_id"),
                        "audio_id_kind": annotation.get("audio_id_kind"),
                        "labels_source": labels_source,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            rebuilt_generated.append({**row, "ground_truth": str(row.get("ground_truth"))})
        write_jsonl(generated_path, rebuilt_generated)

    print(
        json.dumps(
            {
                "labels_source": labels_source,
                "episodes": len(stats),
                "ads": sum(item["ads"] for item in stats),
                "details": stats,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
