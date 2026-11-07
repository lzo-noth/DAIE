#!/usr/bin/env python3
"""Sample episodes from archived reports, download audio, and build a Release manifest.

The archived ``artifact/**/report.json`` files contain episode metadata
including ``audio_url``. This script deterministically samples episodes (one
per feed first), downloads the audio, hashes the bytes, and writes a manifest
whose ``audio`` field points at the future GitHub Release asset URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "dai-sample-audio"


def _collect(reports_root: Path) -> list[dict]:
    episodes: dict[str, dict] = {}
    for report_path in sorted(reports_root.rglob("report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        feed = str(payload.get("feed") or report_path.parent.name)
        for episode in payload.get("episodes") or []:
            url = str(episode.get("audio_url") or "")
            if not url:
                continue
            key = url
            if key in episodes:
                continue
            episodes[key] = {
                "audio_url": url,
                "feed": feed,
                "episode": str(episode.get("title") or episode.get("id") or ""),
                "episode_id": str(episode.get("id") or ""),
                "duration_s": float(episode.get("duration_s") or 0.0),
                "report": str(report_path),
            }
    return list(episodes.values())


def _ordered_candidates(items: list[dict], seed: int, max_per_feed: int) -> list[dict]:
    rng = random.Random(seed)
    rng.shuffle(items)
    picked: list[dict] = []
    seen: dict[str, int] = {}
    for item in items:
        feed = item["feed"]
        if seen.get(feed, 0) < max_per_feed:
            picked.append(item)
            seen[feed] = seen.get(feed, 0) + 1
    rest = [item for item in items if item not in picked]
    picked.extend(rest)
    return picked


def _safe_name(item: dict, index: int, url: str) -> str:
    raw = f"{item['feed']}_{item['episode']}".strip() or f"episode_{index}"
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("_")[:140] or f"episode_{index}"
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix not in {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".flac", ".mp4"}:
        suffix = ".mp3"
    return name + suffix


def _download(url: str, target: Path, timeout: int) -> str:
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            handle.write(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("artifact"))
    parser.add_argument("--index", type=Path, help="可选的 episode 索引 JSONL（替代扫描 artifact）")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-feed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("out/sample_audio"))
    parser.add_argument("--manifest", type=Path, default=Path("out/sample_audio_manifest.jsonl"))
    parser.add_argument("--release-tag", default="1.0")
    parser.add_argument("--repo", default="lzo-noth/DAIE")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    items = (
        [
            json.loads(line)
            for line in args.index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.index
        else _collect(args.reports_root)
    )
    if not items:
        print(f"没有在 {args.reports_root} 找到带 audio_url 的 episode", file=sys.stderr)
        return 1
    candidates = _ordered_candidates(items, args.seed, args.max_per_feed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[dict] = []
    used_names: set[str] = set()
    for item in candidates:
        if len(rows) >= args.count:
            break
        name = _safe_name(item, len(rows) + 1, item["audio_url"])
        if name in used_names:
            name = f"{Path(name).stem}_{len(rows) + 1}{Path(name).suffix}"
        target = args.out_dir / name
        try:
            digest = _download(item["audio_url"], target, args.timeout)
        except Exception as exc:  # noqa: BLE001 - keep sampling other episodes
            target.unlink(missing_ok=True)
            failures.append({**item, "error": f"{type(exc).__name__}: {exc}"[:200]})
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
                "source_sha256": digest,
                "source_url": item["audio_url"],
                "report": item["report"],
            }
        )
        print(f"[sample] {item['feed']} / {item['episode']} -> {name}", flush=True)

    args.manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "candidates": len(items),
        "sampled": len(rows),
        "feeds": len({row["feed"] for row in rows}),
        "failures": failures,
        "manifest": str(args.manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
