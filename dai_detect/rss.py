from __future__ import annotations

import gzip
import hashlib
import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from .models import Episode

ATOM = "http://www.w3.org/2005/Atom"
ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
NS = {"itunes": ITUNES, "atom": ATOM}


class FeedError(RuntimeError):
    pass


def fetch_feed(url: str, proxy: Optional[str] = None, timeout: float = 30.0) -> bytes:
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"User-Agent": "dai-detect/0.1"})
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read()
    except Exception as exc:  # noqa: BLE001 - surfaced as FeedError
        raise FeedError(f"抓取 RSS 失败: {exc}") from exc
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data


def parse_duration(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        if ":" in raw:
            total = 0.0
            for part in raw.split(":"):
                total = total * 60 + float(part)
            return total
        return float(raw)
    except ValueError:
        return None


def _text(node: Optional[ET.Element]) -> str:
    if node is None or node.text is None:
        return ""
    return html.unescape(node.text.strip())


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(node: ET.Element, name: str) -> Optional[ET.Element]:
    for child in node:
        if _local_name(child.tag) == name:
            return child
    return None


def _parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _episode_id(guid: str, audio_url: str) -> str:
    seed = guid or audio_url
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _is_audio(enclosure: ET.Element) -> bool:
    kind = (enclosure.get("type") or "").lower()
    if not kind:
        return True
    return kind.startswith("audio") or kind == "application/octet-stream"


def _rss_episodes(channel: ET.Element, feed_url: str) -> tuple[str, list[Episode]]:
    feed_title = _text(_child(channel, "title"))
    episodes: list[Episode] = []
    for item in channel:
        if _local_name(item.tag) != "item":
            continue
        enclosure = _child(item, "enclosure")
        if enclosure is None:
            continue
        url = enclosure.get("url") or ""
        if not url or not _is_audio(enclosure):
            continue
        url = urllib.parse.urljoin(feed_url, url)
        title = _text(_child(item, "title")) or url.rsplit("/", 1)[-1]
        guid = _text(_child(item, "guid")) or _text(_child(item, "link"))
        published = _text(_child(item, "pubDate"))
        duration = parse_duration(_text(_child(item, "duration")))
        episodes.append(
            Episode(
                id=_episode_id(guid, url),
                title=title,
                audio_url=url,
                published=published or None,
                duration_s=duration,
                source=feed_title or None,
            )
        )
    return feed_title, episodes


def _atom_episodes(feed: ET.Element, feed_url: str) -> tuple[str, list[Episode]]:
    feed_title = _text(_child(feed, "title"))
    episodes: list[Episode] = []
    for entry in feed:
        if _local_name(entry.tag) != "entry":
            continue
        audio_url = ""
        audio_type = ""
        for link in entry:
            if _local_name(link.tag) != "link":
                continue
            href = link.get("href") or ""
            rel = (link.get("rel") or "").lower()
            kind = (link.get("type") or "").lower()
            if rel == "enclosure" and href:
                audio_url = href
                audio_type = kind
                break
        if not audio_url:
            continue
        if audio_type and not audio_type.startswith("audio"):
            continue
        audio_url = urllib.parse.urljoin(feed_url, audio_url)
        title = _text(_child(entry, "title")) or audio_url.rsplit("/", 1)[-1]
        guid = _text(_child(entry, "id"))
        published = _text(_child(entry, "updated")) or _text(_child(entry, "published"))
        episodes.append(
            Episode(
                id=_episode_id(guid, audio_url),
                title=title,
                audio_url=audio_url,
                published=published or None,
                source=feed_title or None,
            )
        )
    return feed_title, episodes


def parse_feed(data: bytes, feed_url: str = "") -> tuple[str, list[Episode]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise FeedError(f"RSS 解析失败: {exc}") from exc

    root_name = _local_name(root.tag)
    if root_name == "rss":
        channel = _child(root, "channel")
        if channel is None:
            raise FeedError("RSS 缺少 channel 节点")
        feed_title, episodes = _rss_episodes(channel, feed_url)
    elif root_name == "feed":
        feed_title, episodes = _atom_episodes(root, feed_url)
    else:
        raise FeedError(f"不支持的订阅格式: {root_name}")

    seen = set()
    unique: list[Episode] = []
    for episode in episodes:
        if episode.audio_url in seen:
            continue
        seen.add(episode.audio_url)
        unique.append(episode)

    fallback = datetime.min.replace(tzinfo=timezone.utc)
    if any(_parse_date(ep.published) for ep in unique):
        unique.sort(key=lambda ep: _parse_date(ep.published) or fallback, reverse=True)
    return feed_title, unique
