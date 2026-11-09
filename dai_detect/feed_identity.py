"""Stable feed identity for training/blind isolation checks.

Display names are not stable identities: the same show can be listed as
``This Week in Tech`` and ``TWiT``, and a feed can move to a new host or URL.
Isolation must key off something that survives renaming, so the canonical key
is a hash of the normalized feed URL when one is known, and only falls back to
a normalized display name when it is not.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse

_SCHEME = re.compile(r"^[a-z]+://", re.IGNORECASE)


def normalize_feed_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    text = _SCHEME.sub("", text)
    parsed = urllib.parse.urlsplit("//" + text if "://" not in text else text)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def canonical_feed_key(name: str, url: str = "") -> str:
    normalized = normalize_feed_url(url)
    if normalized:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"url:{digest}"
    cleaned = re.sub(r"\s+", " ", str(name or "").strip()).lower()
    return f"name:{cleaned}"
