#!/usr/bin/env python3
"""Seed a resume run with a previous artifact, filtered to the current manifests.

A resume run reuses earlier ASR/LLM work, but the manifest may have changed:
episodes can be promoted from blind to training, and a new blind holdout may
replace the old one. Copying the whole cache would (a) skip episodes that are
no longer in the manifest and (b) leave stale ``blind_generated`` rows that the
coverage gate correctly rejects.

This copies only rows whose ``(feed, episode, source_sha256)`` identity is
still part of the current train or blind manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_FILES = ("annotations.jsonl", "transcripts.jsonl")
BLIND_FILES = (
    "blind_annotations.jsonl",
    "blind_transcripts.jsonl",
    "blind_generated.jsonl",
)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("feed") or "unknown"),
        str(row.get("episode") or ""),
        str(row.get("source_sha256") or ""),
    )


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def seed(
    base_dir: Path,
    out_dir: Path,
    train_manifest: Path,
    blind_manifest: Path,
) -> dict:
    train_allowed = {_identity(row) for row in _load(train_manifest)}
    blind_allowed = {_identity(row) for row in _load(blind_manifest)}
    report: dict[str, dict] = {}
    for names, allowed in (
        (TRAIN_FILES, train_allowed),
        (BLIND_FILES, blind_allowed),
    ):
        for name in names:
            merged: dict[tuple[str, str, str], dict] = {}
            for source in (out_dir / name, base_dir / name):
                for row in _load(source):
                    merged[_identity(row)] = row
            kept = [row for key, row in merged.items() if key in allowed]
            _write(out_dir / name, kept)
            report[name] = {
                "candidates": len(merged),
                "kept": len(kept),
                "dropped": len(merged) - len(kept),
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    args = parser.parse_args()

    report = seed(args.base_dir, args.out_dir, args.train_manifest, args.blind_manifest)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
