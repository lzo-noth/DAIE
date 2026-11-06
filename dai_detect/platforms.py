from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .rss import fetch_feed, parse_feed


def load_platforms(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("platforms.json 顶层必须是数组")
    return data


def iter_feed_entries(platforms: list[dict]):
    for platform in platforms:
        for feed in platform.get("feeds", []):
            yield platform, feed


def verify_platforms(path: str | Path, proxy: Optional[str] = None) -> dict:
    platforms = load_platforms(path)
    results = []
    for platform in platforms:
        entry = {
            "platform": platform.get("platform"),
            "owner": platform.get("owner"),
            "delivery": platform.get("delivery"),
            "feeds": [],
        }
        for feed in platform.get("feeds", []):
            record = {
                "name": feed.get("name"),
                "slug": feed.get("slug"),
                "url": feed.get("url"),
                "status": "ok",
            }
            try:
                payload = fetch_feed(feed["url"], proxy=proxy, timeout=45)
                title, episodes = parse_feed(payload, feed["url"])
                record.update(
                    {
                        "title": title,
                        "episodes": len(episodes),
                        "latest_title": episodes[0].title if episodes else None,
                        "latest_duration_s": episodes[0].duration_s if episodes else None,
                        "latest_audio_url": episodes[0].audio_url if episodes else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - verification records failures
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            entry["feeds"].append(record)
        results.append(entry)
    return {
        "tool": "dai-detect",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platforms": results,
    }
