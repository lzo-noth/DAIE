#!/usr/bin/env python3
"""Split released ad blocks into one event per sponsor creative.

Runs after the label review steps, on both the training annotations and the
blind ground truth, so training, boundary heads and blind evaluation share one
event definition. The split only uses the stored per-segment verdicts; it does
not touch audio or the runtime detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.ad_events import split_annotation_row


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


def _index(rows: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    by_sha: dict[str, dict] = {}
    by_pair: dict[tuple[str, str], dict] = {}
    for row in rows:
        sha = str(row.get("source_sha256") or "")
        if sha:
            by_sha[sha] = row
        by_pair[(str(row.get("feed") or ""), str(row.get("episode") or ""))] = row
    return by_sha, by_pair


def _split_file(
    annotations: Path, transcripts: Path, min_duration_s: float
) -> dict:
    rows = _load_jsonl(annotations)
    if not rows:
        return {"file": str(annotations), "skipped": "missing"}
    by_sha, by_pair = _index(_load_jsonl(transcripts))
    before = after = 0
    updated = []
    for row in rows:
        transcript = by_sha.get(str(row.get("source_sha256") or "")) or by_pair.get(
            (str(row.get("feed") or ""), str(row.get("episode") or ""))
        )
        new_row, old_count, new_count = split_annotation_row(
            row, transcript, min_duration_s
        )
        before += old_count
        after += new_count
        updated.append(new_row)
    _write_jsonl(annotations, updated)
    return {
        "file": str(annotations),
        "episodes": len(updated),
        "ads_before": before,
        "ads_after": after,
    }


def _sync_ground_truth(data_dir: Path, min_duration_s: float) -> dict:
    generated = _load_jsonl(data_dir / "blind_generated.jsonl")
    annotations = _load_jsonl(data_dir / "blind_annotations.jsonl")
    if not generated or not annotations:
        return {"ground_truth": "skipped"}
    by_sha, by_pair = _index(annotations)
    updated = 0
    for item in generated:
        row = by_sha.get(str(item.get("source_sha256") or "")) or by_pair.get(
            (str(item.get("feed") or ""), str(item.get("episode") or ""))
        )
        if row is None:
            continue
        target = data_dir / "blind_gt" / Path(str(item.get("ground_truth") or "")).name
        if not target.exists():
            continue
        payload = json.loads(target.read_text(encoding="utf-8"))
        if str(payload.get("source_sha256") or "") != str(row.get("source_sha256") or ""):
            continue
        payload["ads"] = row.get("ads") or []
        payload["event_granularity"] = "sponsor_creative"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        updated += 1
    return {"ground_truth": str(data_dir / "blind_gt"), "episodes": updated}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("out/release"))
    parser.add_argument("--min-duration", type=float, default=5.0)
    args = parser.parse_args()

    results = [
        _split_file(
            args.data_dir / "annotations.jsonl",
            args.data_dir / "transcripts.jsonl",
            args.min_duration,
        ),
        _split_file(
            args.data_dir / "blind_annotations.jsonl",
            args.data_dir / "blind_transcripts.jsonl",
            args.min_duration,
        ),
        _sync_ground_truth(args.data_dir, args.min_duration),
    ]
    print(json.dumps({"results": results}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
