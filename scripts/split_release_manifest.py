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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.feed_identity import canonical_feed_key


def _group(row: dict) -> str:
    """Isolate by canonical feed identity, not by display name.

    The same show can appear under different names (``This Week in Tech`` vs
    ``TWiT``) and a named feed can move to a new URL; hashing the feed URL keeps
    the train/blind boundary honest across both.
    """
    return str(
        row.get("feed_key")
        or canonical_feed_key(
            str(row.get("feed") or ""), str(row.get("feed_url") or "")
        )
    )


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _backfill_feed_urls(rows: list[dict], platforms: Path | None) -> int:
    """Fill ``feed_url`` on historical rows so every row groups by URL.

    Older manifests only carried display names. Grouping those by name while
    newer rows group by URL would split one show into two identities and could
    put half of it on each side of the train/blind boundary.
    """
    if platforms is None or not platforms.exists():
        return 0
    payload = json.loads(platforms.read_text(encoding="utf-8"))
    by_name: dict[str, str] = {}
    for platform in payload:
        for feed in platform.get("feeds") or []:
            if feed.get("name") and feed.get("url"):
                by_name[str(feed["name"])] = str(feed["url"])
    filled = 0
    for row in rows:
        if row.get("feed_url"):
            continue
        url = by_name.get(str(row.get("feed") or ""))
        if url:
            row["feed_url"] = url
            filled += 1
    return filled


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
    parser.add_argument(
        "--platforms",
        type=Path,
        help="platforms.json；用于给缺少 feed_url 的历史行回填规范化 feed 身份",
    )
    args = parser.parse_args()

    rows = _load(args.manifest)
    if not rows:
        print(f"{args.manifest} 为空", flush=True)
        return 1
    filled = _backfill_feed_urls(rows, args.platforms)

    existing_blind = _load(args.existing_blind) if args.existing_blind else []
    filled += _backfill_feed_urls(existing_blind, args.platforms)
    if filled:
        print(f"[feeds] 回填 feed_url: {filled} 行", flush=True)
    blind_feeds = {_group(row) for row in existing_blind}

    by_feed: dict[str, list[dict]] = {}
    for row in rows:
        by_feed.setdefault(_group(row), []).append(row)

    candidates = sorted(feed for feed in by_feed if feed and feed not in blind_feeds)
    random.Random(args.seed).shuffle(candidates)
    blind_count = sum(len(by_feed[feed]) for feed in blind_feeds if feed in by_feed)
    for feed in candidates:
        if blind_count >= args.blind_episodes:
            break
        blind_feeds.add(feed)
        blind_count += len(by_feed[feed])

    blind_rows = [row for row in rows if _group(row) in blind_feeds]
    train_rows = [
        row
        for row in rows
        if _group(row) not in blind_feeds
    ]
    _write(args.train_out, train_rows)
    _write(args.blind_out, blind_rows)

    train_feeds = {_group(row) for row in train_rows}
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
