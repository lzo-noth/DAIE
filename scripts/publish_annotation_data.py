#!/usr/bin/env python3
"""Copy non-audio annotation data from a pipeline output dir into the repo.

Audio snapshots stay in the Actions artifact because training needs the exact
bytes; labels and transcripts are committed so the dataset can be reviewed and
diffed in git. Local snapshot paths inside the JSONL rows are rewritten to
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

JSONL_FILES = (
    "annotations.jsonl",
    "transcripts.jsonl",
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


def _relocate(row: dict) -> dict:
    audio = row.get("audio")
    if isinstance(audio, str) and audio:
        row["audio"] = f"snapshots/{Path(audio).name}"
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, str) and ground_truth:
        row["ground_truth"] = f"blind_gt/{Path(ground_truth).name}"
    return row


def _publish_jsonl(source: Path, target: Path) -> dict:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(_relocate(json.loads(line)))
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
    for name in JSONL_FILES:
        source = args.source / name
        if not source.exists():
            continue
        published[name] = _publish_jsonl(source, args.dest / name)

    ground_truth_dir = args.source / "blind_gt"
    if ground_truth_dir.is_dir():
        target_dir = args.dest / "blind_gt"
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(ground_truth_dir.glob("*.json")):
            target = target_dir / source.name
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        published["blind_gt"] = {
            "files": len(list(target_dir.glob("*.json"))),
        }

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
    }
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
