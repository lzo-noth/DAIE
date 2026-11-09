#!/usr/bin/env python3
"""Copy non-audio *training* annotation data from a pipeline output into the repo.

Blind-holdout labels are deliberately not published. Committing the blind
ground truth turns the final holdout into a de-facto training signal: anyone
editing features or thresholds can inspect the exact start/end times they are
being scored against. Feed-disjoint splits alone do not prevent that.

What is published:
  - ``annotations.jsonl`` / ``transcripts.jsonl`` for training episodes
  - ``index.json`` with counts and dataset identity

What stays private (Actions artifact / separate evaluation repo):
  - ``blind_annotations.jsonl`` / ``blind_transcripts.jsonl``
  - ``blind_generated.jsonl`` and ``blind_gt/*.json``

Audio snapshots stay in the Actions artifact because training needs the exact
bytes; local snapshot paths inside the JSONL rows are rewritten to
``snapshots/<file>`` so the files stay relocatable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

PUBLISHED_JSONL = (
    "annotations.jsonl",
    "transcripts.jsonl",
)
HOLDOUT_JSONL = (
    "blind_annotations.jsonl",
    "blind_transcripts.jsonl",
    "blind_generated.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relocate(row: dict, snapshot_dir: str = "snapshots") -> dict:
    audio = row.get("audio")
    if isinstance(audio, str) and audio:
        row["audio"] = f"{snapshot_dir}/{Path(audio).name}"
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, str) and ground_truth:
        row["ground_truth"] = f"blind_gt/{Path(ground_truth).name}"
    return row


def _publish_jsonl(source: Path, target: Path, snapshot_dir: str = "snapshots") -> dict:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(_relocate(json.loads(line), snapshot_dir))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "sha256": _sha256(target)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="pipeline 输出目录（含 out/release）")
    parser.add_argument("--dest", type=Path, required=True, help="仓库内数据集目录")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    args = parser.parse_args()

    published: dict[str, dict] = {}
    for name in PUBLISHED_JSONL:
        source = args.source / name
        if not source.exists():
            continue
        # 合并后的 artifact 把所有音频都放在 snapshots/。
        published[name] = _publish_jsonl(source, args.dest / name, "snapshots")

    # 撤回历史版本里已经公开的盲测真值，避免旧提交继续泄露 holdout。
    removed = []
    for name in HOLDOUT_JSONL:
        target = args.dest / name
        if target.exists():
            target.unlink()
            removed.append(str(target))
    blind_gt = args.dest / "blind_gt"
    if blind_gt.is_dir():
        shutil.rmtree(blind_gt)
        removed.append(str(blind_gt))

    holdout_rows = 0
    blind_generated = args.source / "blind_generated.jsonl"
    if blind_generated.exists():
        holdout_rows = sum(
            1 for line in blind_generated.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    if not published:
        print(f"{args.source} 没有可发布的数据", flush=True)
        return 1

    labels_sources: set[str] = set()
    feeds: list[str] = []
    ads = 0
    annotations = args.source / "annotations.jsonl"
    if annotations.exists():
        for line in annotations.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("labels_source"):
                labels_sources.add(str(row["labels_source"]))
            if row.get("feed"):
                feeds.append(str(row["feed"]))
            ads += len(row.get("ads") or [])

    index = {
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": args.run_id,
        "llm_model": args.model,
        "labels_sources": sorted(labels_sources),
        "files": published,
        "episodes": published.get("annotations.jsonl", {}).get("rows", 0),
        "feeds": len(set(feeds)),
        "ads": ads,
        "holdout": {
            "published": False,
            "episodes": holdout_rows,
            "location": "actions_artifact",
            "note": "盲测标签不进入公开仓库，避免评测集被逐步适配",
        },
        "removed_holdout_files": removed,
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
