#!/usr/bin/env python3
"""Seed a resume run with a previous artifact, filtered to the current manifests.

A resume run reuses earlier ASR/LLM work, but the manifest may have changed:
episodes can be promoted from blind to training, and a new blind holdout may
replace the old one. Copying the whole cache would (a) skip episodes that are
no longer in the manifest and (b) leave stale ``blind_generated`` rows that the
coverage gate correctly rejects.

This copies only rows whose ``(feed, episode, source_sha256)`` identity is
still part of the current train or blind manifest. Reuse across the train/blind
split is allowed only between files that carry the same payload: a demoted
blind episode's transcript may seed ``transcripts.jsonl``, but it must never
leak into ``blind_generated.jsonl`` -- that file is the list of ground-truth
files, and a transcript row there has no ``ground_truth`` path at all.
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

# Output file -> the same-schema file on the other side of the split.
COUNTERPARTS = {
    "annotations.jsonl": "blind_annotations.jsonl",
    "transcripts.jsonl": "blind_transcripts.jsonl",
    "blind_annotations.jsonl": "annotations.jsonl",
    "blind_transcripts.jsonl": "transcripts.jsonl",
    # blind_generated 只有一个用途（指向真值文件），没有可复用的对端；
    # 混入别的文件会让覆盖率门禁读到没有 ground_truth 的行。
    "blind_generated.jsonl": None,
}

# 人工级标签比机器标签更可信：续跑时若仓库里已有复核过的行，不能被
# artifact 里的旧机器标注覆盖回去。
LABEL_PRIORITY = {"human": 3, "ai_reviewed": 2, "llm_weak": 1}


def _rank(row: dict) -> int:
    return LABEL_PRIORITY.get(str(row.get("labels_source") or ""), 0)


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
            own_keys: set[tuple[str, str, str]] = set()
            seen: set[tuple[str, str, str]] = set()
            counterpart = COUNTERPARTS.get(name)
            sources = [
                (out_dir / name, True),
                (base_dir / name, True),
                (out_dir / counterpart, False) if counterpart else (None, False),
                (base_dir / counterpart, False) if counterpart else (None, False),
            ]
            for path, is_own in sources:
                if path is None:
                    continue
                for row in _load(path):
                    key = _identity(row)
                    seen.add(key)
                    if key not in allowed:
                        continue
                    if key in merged:
                        # 同一身份出现两次时不能"先到先得"：续跑 artifact 里
                        # 的旧机器标注会覆盖仓库里已经复核过的行。
                        if _rank(merged[key]) >= _rank(row):
                            continue
                        merged[key] = row
                        if is_own:
                            own_keys.add(key)
                        continue
                    merged[key] = row
                    if is_own:
                        own_keys.add(key)
            # 同一份转写可能因盲测轮换而换了侧，复用时不重复消耗 ASR；
            # 但只有同 schema 的对端文件才允许补位。
            kept_keys = list(merged)
            kept_rows = [merged[key] for key in kept_keys]
            if name == "blind_generated.jsonl":
                # 该文件是"真值索引"，没有 ground_truth 的行是历史污染的
                # 转写行。留着它会占住身份键，让正式索引行再也写不回来。
                kept_rows = [row for row in kept_rows if row.get("ground_truth")]
            _write(out_dir / name, kept_rows)
            report[name] = {
                "candidates": len(seen),
                "kept": len(kept_rows),
                "dropped": len(seen) - len(kept_rows),
                "from_other_set": len([key for key in kept_keys if key not in own_keys]),
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
