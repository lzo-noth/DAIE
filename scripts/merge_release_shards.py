#!/usr/bin/env python3
"""Merge annotation shards into one pipeline output directory.

Each shard writes its own ``annotations.jsonl`` / ``transcripts.jsonl`` /
``blind_*`` files plus the audio snapshots it downloaded. The merged directory
must be a drop-in replacement for a single-job run so training and publishing
work unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

JSONL_FILES = (
    "annotations.jsonl",
    "transcripts.jsonl",
    "blind_annotations.jsonl",
    "blind_transcripts.jsonl",
    "blind_generated.jsonl",
)


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))


def _merge_jsonl(sources: list[Path], target: Path) -> int:
    rows: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for source in sources:
        if not source.exists():
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = _key(row)
            if key not in rows:
                order.append(key)
            rows[key] = row
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for key in order:
            handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
    return len(order)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", type=Path, required=True, help="含 shard-*/ 的目录")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    shards = sorted(path for path in args.shards_dir.glob("shard-*") if path.is_dir())
    if not shards:
        print(f"{args.shards_dir} 没有 shard-* 目录", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in JSONL_FILES:
        count = _merge_jsonl([shard / name for shard in shards], args.out_dir / name)
        if count:
            print(f"[merge] {name}: {count} rows", flush=True)

    gt_rows = 0
    for shard in shards:
        source_dir = shard / "blind_gt"
        if not source_dir.is_dir():
            continue
        target_dir = args.out_dir / "blind_gt"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob("*.json"):
            shutil.copyfile(source, target_dir / source.name)
            gt_rows += 1
    if gt_rows:
        print(f"[merge] blind_gt: {gt_rows} files", flush=True)

    snapshots = 0
    for shard in shards:
        for source_dir in (shard / "snapshots", shard / "blind_snapshots"):
            if not source_dir.is_dir():
                continue
            target_dir = args.out_dir / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source in source_dir.glob("*"):
                target = target_dir / source.name
                if source.is_file() and not target.exists():
                    shutil.copyfile(source, target)
                    snapshots += 1
    print(f"[merge] snapshots: {snapshots} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
