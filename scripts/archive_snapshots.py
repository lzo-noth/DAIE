#!/usr/bin/env python3
"""Mirror training audio snapshots to an immutable Release.

Snapshots are content-addressed (``<sha256>.<ext>``), so a Release asset is a
permanent, re-downloadable copy. Actions artifacts expire after 30 days; without
this step the repo could keep labels and hashes while the exact DAI variant of
the audio is gone, and the same URL cannot be trusted to return the same ad
insertion later.

Only training snapshots are archived by default. Blind audio stays in the
artifact so the holdout is not published next to its own labels.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_TAG = "audio-snapshots"


def _gh(arguments: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *arguments], capture_output=True, text=True, check=check
    )


def _ensure_release(tag: str, repo: str) -> None:
    view = _gh(["release", "view", tag, "--repo", repo], check=False)
    if view.returncode == 0:
        return
    _gh(
        [
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--title",
            "DAIE audio snapshots",
            "--notes",
            "Immutable content-addressed training snapshots (never clobbered).",
        ]
    )


def archive(directory: Path, tag: str, repo: str, dry_run: bool = False) -> dict:
    entries: list[dict] = []
    if not directory.is_dir():
        return {"directory": str(directory), "files": 0, "entries": []}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        entries.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                # 快照文件名就是内容哈希，不需要重算。
                "sha256": path.stem,
                "asset_url": (
                    f"https://github.com/{repo}/releases/download/{tag}/{path.name}"
                ),
            }
        )
    if dry_run or not entries:
        return {"directory": str(directory), "files": len(entries), "entries": entries}

    _ensure_release(tag, repo)
    uploaded = 0
    skipped = 0
    for entry in entries:
        path = directory / entry["name"]
        # 不使用 --clobber：内容寻址的快照一旦存在就不应该被替换。
        result = _gh(
            ["release", "upload", tag, str(path), "--repo", repo],
            check=False,
        )
        if result.returncode == 0:
            uploaded += 1
        elif "already exists" in (result.stderr or "").lower():
            skipped += 1
        else:
            raise RuntimeError(
                f"上传 {entry['name']} 失败: {(result.stderr or '').strip()[:300]}"
            )
    return {
        "directory": str(directory),
        "files": len(entries),
        "uploaded": uploaded,
        "skipped_existing": skipped,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, action="append", required=True)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--repo", default="")
    parser.add_argument("--index-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo
    if not repo:
        repo = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    results = [archive(directory, args.tag, repo, args.dry_run) for directory in args.dir]
    payload = {"tag": args.tag, "repo": repo, "directories": results}
    if args.index_out:
        args.index_out.parent.mkdir(parents=True, exist_ok=True)
        args.index_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
