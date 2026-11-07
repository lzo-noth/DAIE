#!/usr/bin/env python3
"""Sample episodes from platforms.json RSS feeds and download their audio.

Unlike the archived-report sampler, this resolves current episode enclosure
URLs directly from each feed in ``platforms.json``. It prioritizes feed
diversity, downloads the audio, hashes the bytes, and writes a manifest whose
``audio`` field points at the future GitHub Release asset URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "dai-platform-sampler"
AUDIO_SUFFIXES = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac", ".mp4"}


def _load_feeds(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feeds: list[dict] = []
    for platform in payload:
        for feed in platform.get("feeds") or []:
            if feed.get("url"):
                feeds.append(
                    {
                        "name": str(feed.get("name") or ""),
                        "slug": str(feed.get("slug") or ""),
                        "url": str(feed["url"]),
                        "platform": str(platform.get("platform") or ""),
                    }
                )
    return feeds


def _is_audio_url(url: str, content_type: str = "") -> bool:
    ctype = (content_type or "").lower()
    if ctype.startswith("audio/"):
        return True
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    return suffix in AUDIO_SUFFIXES


def _entry_audio(entry) -> str:
    for enclosure in entry.get("enclosures") or []:
        href = str(enclosure.get("href") or enclosure.get("url") or "")
        if href.startswith(("http://", "https://")) and _is_audio_url(
            href, str(enclosure.get("type") or "")
        ):
            return href
    for link in entry.get("links") or []:
        href = str(link.get("href") or "")
        if (
            link.get("rel") == "enclosure"
            and href.startswith(("http://", "https://"))
            and _is_audio_url(href, str(link.get("type") or ""))
        ):
            return href
    return ""


def _collect_feed(feed: dict, latest: int, timeout: int) -> list[dict]:
    import feedparser

    request = urllib.request.Request(feed["url"], headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    parsed = feedparser.parse(payload)
    episodes = []
    for entry in parsed.entries[:latest]:
        audio = _entry_audio(entry)
        if not audio:
            continue
        episodes.append(
            {
                "audio_url": audio,
                "feed": feed["name"] or feed["slug"],
                "feed_slug": feed["slug"],
                "platform": feed["platform"],
                "episode": str(entry.get("title") or audio.rsplit("/", 1)[-1]),
                "episode_id": str(entry.get("id") or entry.get("guid") or ""),
                "published": str(entry.get("published") or entry.get("updated") or ""),
                "duration_s": _duration(entry),
                "feed_url": feed["url"],
            }
        )
    return episodes


def _duration(entry) -> float:
    raw = entry.get("itunes_duration") or ""
    if ":" in str(raw):
        parts = [float(part) for part in str(raw).split(":")]
        total = 0.0
        for part in parts:
            total = total * 60.0 + part
        return total
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _safe_name(item: dict, index: int) -> str:
    raw = f"{item.get('feed_slug') or item['feed']}_{item['episode']}"
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("_")[:140] or f"episode_{index}"
    suffix = Path(urllib.parse.urlparse(item["audio_url"]).path).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".mp3"
    return name + suffix


def _download(url: str, target: Path, timeout: int, retries: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        digest = hashlib.sha256()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    handle.write(block)
            return digest.hexdigest()
        except Exception as exc:  # noqa: BLE001 - retry transient feed/CDN failures
            last_error = exc
            target.unlink(missing_ok=True)
    raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _ordered(items: list[dict], seed: int, max_per_feed: int) -> list[dict]:
    rng = random.Random(seed)
    by_feed: dict[str, list[dict]] = {}
    for item in items:
        by_feed.setdefault(item["feed"], []).append(item)
    for values in by_feed.values():
        rng.shuffle(values)
    feeds = list(by_feed)
    rng.shuffle(feeds)
    ordered: list[dict] = []
    for round_index in range(max_per_feed):
        for feed in feeds:
            if round_index < len(by_feed[feed]):
                ordered.append(by_feed[feed][round_index])
    for feed in feeds:
        ordered.extend(by_feed[feed][max_per_feed:])
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platforms", type=Path, default=Path("platforms.json"))
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-feed", type=int, default=1)
    parser.add_argument("--latest-per-feed", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("out/sample_audio"))
    parser.add_argument("--manifest", type=Path, default=Path("out/sample_audio_manifest.jsonl"))
    parser.add_argument("--release-tag", default="1.0")
    parser.add_argument("--repo", default="lzo-noth/DAIE")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    feeds = _load_feeds(args.platforms)
    if not feeds:
        print("platforms.json 没有可用 feed", file=sys.stderr)
        return 1
    items: list[dict] = []
    for feed in feeds:
        print(f"[feed] {feed['name']} {feed['url']}", flush=True)
        try:
            episode_rows = _collect_feed(feed, args.latest_per_feed, 60)
        except Exception as exc:  # noqa: BLE001 - one dead feed must not stop sampling
            print(f"[feed] skipped {feed['name']}: {exc}", flush=True)
            continue
        print(f"[feed] {feed['name']}: {len(episode_rows)} episodes", flush=True)
        items.extend(episode_rows)
    if not items:
        print("所有 feed 都没有解析到带 enclosure 的 episode", file=sys.stderr)
        return 1

    candidates = _ordered(items, args.seed, args.max_per_feed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    used_names: set[str] = set()
    for item in candidates:
        if len(rows) >= args.count:
            break
        name = _safe_name(item, len(rows) + 1)
        if name in used_names:
            name = f"{Path(name).stem}_{len(rows) + 1}{Path(name).suffix}"
        target = args.out_dir / name
        try:
            digest = _download(item["audio_url"], target, args.timeout)
        except Exception as exc:  # noqa: BLE001 - keep sampling other episodes
            failures.append({**item, "error": str(exc)[:200]})
            print(f"[skip] {item['feed']} / {item['episode']}: {exc}", flush=True)
            continue
        used_names.add(name)
        rows.append(
            {
                "audio": f"https://github.com/{args.repo}/releases/download/{args.release_tag}/{urllib.parse.quote(name)}",
                "feed": item["feed"],
                "episode": item["episode"],
                "episode_id": item["episode_id"],
                "duration_s": item["duration_s"],
                "published": item["published"],
                "platform": item["platform"],
                "source_url": item["audio_url"],
                "source_sha256": digest,
            }
        )
        print(f"[sample] {item['feed']} / {item['episode']} -> {name}", flush=True)

    args.manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "feeds_resolved": len({row['feed'] for row in rows}),
                "sampled": len(rows),
                "failures": failures,
                "manifest": str(args.manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
