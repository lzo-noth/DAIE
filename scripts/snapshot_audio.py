#!/usr/bin/env python3
"""Create an exact audio snapshot and its identity record.

DAI rewrites the audio behind the same episode URL on every request. A label
is only valid for the exact variant it was created on, so the variant must be
archived. This script downloads (or copies) the bytes once, names the snapshot
by its SHA-256, and decodes it to obtain the decoded-PCM content id.

It prints one annotation row; pass ``--manifest`` to append it to a JSONL file
that ``scripts/train_single_episode.py`` can verify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.net import HTTP_USER_AGENT
from dai_detect.single_episode import extract_features

CHUNK = 1024 * 1024
AUDIO_SUFFIXES = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".flac", ".opus", ".mp4"}


def _is_remote(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _suffix(source: str) -> str:
    suffix = Path(urlparse(source).path).suffix.lower()
    return suffix if suffix in AUDIO_SUFFIXES else ".bin"


def _copy_local(source: str, target: Path) -> str:
    digest = hashlib.sha256()
    with Path(source).open("rb") as reader, target.open("wb") as writer:
        for chunk in iter(lambda: reader.read(CHUNK), b""):
            digest.update(chunk)
            writer.write(chunk)
    return digest.hexdigest()


def _download(source: str, target: Path, proxy: str | None) -> str:
    request = Request(source, headers={"User-Agent": HTTP_USER_AGENT})
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy})) if proxy else None
    digest = hashlib.sha256()
    response = opener.open(request, timeout=120) if opener else urlopen(request, timeout=120)
    try:
        with response, target.open("wb") as writer:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                writer.write(chunk)
    finally:
        response.close()
    return digest.hexdigest()


def snapshot(source: str, out_dir: Path, proxy: str | None, force: bool) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="snapshot-", dir=out_dir)
    os.close(fd)
    temp = Path(temp_name)
    try:
        if _is_remote(source):
            digest = _download(source, temp, proxy)
        else:
            digest = _copy_local(source, temp)
        target = out_dir / f"{digest}{_suffix(source)}"
        if target.exists() and not force:
            temp.unlink(missing_ok=True)
            return target, digest
        if target.exists():
            target.unlink()
        temp.replace(target)
        return target, digest
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="音频 URL 或本地路径")
    parser.add_argument("--out-dir", type=Path, default=Path("audio_snapshots"))
    parser.add_argument("--feed", default="")
    parser.add_argument("--episode", default="")
    parser.add_argument("--manifest", type=Path, help="追加一行的 JSONL 清单")
    parser.add_argument("--proxy")
    parser.add_argument("--force", action="store_true", help="即使快照已存在也重新复制/下载")
    args = parser.parse_args()

    target, digest = snapshot(args.source, args.out_dir, args.proxy, args.force)
    mono, forensic = extract_features(str(target), proxy=None)
    duration = min(float(mono["duration_s"]), float(forensic["duration_s"]))
    row = {
        "audio": str(target),
        "snapshot_of": args.source,
        "feed": args.feed,
        "episode": args.episode or target.name,
        "source_sha256": digest,
        "audio_id": str(mono.get("audio_id", "")),
        "audio_id_kind": str(mono.get("audio_id_kind", "")),
        "duration_s": round(duration, 2),
    }
    print(json.dumps(row, ensure_ascii=False))
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
