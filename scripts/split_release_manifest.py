#!/usr/bin/env python3
"""Split a sampled manifest into train and blind sets with disjoint feeds.

Blind feeds never appear in training, so the blind batch measures generalization
to unseen shows. Existing blind rows are kept as-is (their audio is already
annotated) and only new feeds are added until the target size is reached.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="合并后的采样清单")
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--blind-out", type=Path, required=True)
    parser.add_argument("--existing-blind", type=Path, help="保留的既有盲测清单")
    parser.add_argument("--blind-episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = _load(args.manifest)
    if not rows:
        print(f"{args.manifest} 为空", flush=True)
        return 1

    existing_blind = _load(args.existing_blind) if args.existing_blind else []
    blind_feeds = {str(row.get("feed") or "") for row in existing_blind}
    blind_keys = {(str(row.get("feed") or ""), str(row.get("episode") or "")) for row in existing_blind}
    blind_rows = [row for row in rows if (str(row.get("feed") or ""), str(row.get("episode") or "")) in blind_keys]

    by_feed: dict[str, list[dict]] = {}
    for row in rows:
        by_feed.setdefault(str(row.get("feed") or ""), []).append(row)

    candidates = sorted(feed for feed in by_feed if feed and feed not in blind_feeds)
    random.Random(args.seed).shuffle(candidates)
    blind_count = sum(len(by_feed[feed]) for feed in blind_feeds)
    for feed in candidates:
        if blind_count >= args.blind_episodes:
            break
        blind_feeds.add(feed)
        blind_count += len(by_feed[feed])

    blind_rows = [row for row in rows if str(row.get("feed") or "") in blind_feeds]
    train_rows = [
        row
        for row in rows
        if str(row.get("feed") or "") not in blind_feeds
    ]
    _write(args.train_out, train_rows)
    _write(args.blind_out, blind_rows)

    train_feeds = {str(row.get("feed") or "") for row in train_rows}
    print(
        json.dumps(
            {
                "rows": len(rows),
                "train_episodes": len(train_rows),
                "train_feeds": len(train_feeds),
                "blind_episodes": len(blind_rows),
                "blind_feeds": len(blind_feeds),
                "overlap": sorted(train_feeds & blind_feeds),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not (train_feeds & blind_feeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
