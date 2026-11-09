#!/usr/bin/env python3
"""Refuse to publish a release dataset unless every manifest row is complete.

The annotation pipeline is a matrix of shards. Each shard can partially
succeed, so the existence of a merged artifact proves nothing about coverage:
a dead audio URL, a failed LLM call or a damaged snapshot can leave rows
missing while every job still exits successfully. This gate compares the
manifests against the merged output and exits non-zero unless 100% of the
rows are either annotated or explicitly recorded as unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _key(row: dict) -> tuple[str, str]:
    return (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))


def _load_coverage(directories: list[Path]) -> dict:
    unavailable: list[dict] = []
    failed: list[dict] = []
    shards: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.coverage.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            shards.append(str(path))
            unavailable.extend(payload.get("unavailable") or [])
            failed.extend(payload.get("failed") or [])
    return {"shards": shards, "unavailable": unavailable, "failed": failed}


def verify(
    train_manifest: Path,
    blind_manifest: Path,
    data_dir: Path,
    shuffle_dirs: list[Path],
    allow_unavailable: bool,
) -> dict:
    coverage = _load_coverage(shuffle_dirs)
    unavailable_keys = {_key(row) for row in coverage["unavailable"]}
    failed_keys = {_key(row) for row in coverage["failed"]}

    train_expected = {_key(row) for row in _load_jsonl(train_manifest)}
    train_actual = {_key(row) for row in _load_jsonl(data_dir / "annotations.jsonl")}
    train_missing = sorted(train_expected - train_actual)

    blind_expected = {_key(row) for row in _load_jsonl(blind_manifest)}
    blind_actual = {
        _key(row) for row in _load_jsonl(data_dir / "blind_annotations.jsonl")
    }
    blind_missing = sorted(blind_expected - blind_actual)

    generated = _load_jsonl(data_dir / "blind_generated.jsonl")
    gt_dir = data_dir / "blind_gt"
    gt_missing = [
        _key(row)
        for row in generated
        if not (gt_dir / Path(str(row.get("ground_truth") or "")).name).exists()
    ]

    accounted = unavailable_keys | failed_keys
    unresolved = [
        key
        for key in train_missing + blind_missing
        if key not in accounted
    ]
    unresolved_gt = [key for key in gt_missing if key not in accounted]

    complete = not unresolved and not unresolved_gt
    if not allow_unavailable and (unavailable_keys or failed_keys):
        complete = False

    return {
        "train_manifest_rows": len(train_expected),
        "train_annotated_rows": len(train_expected & train_actual),
        "train_missing": train_missing,
        "blind_manifest_rows": len(blind_expected),
        "blind_annotated_rows": len(blind_expected & blind_actual),
        "blind_missing": blind_missing,
        "blind_ground_truth_missing": gt_missing,
        "unresolved": unresolved,
        "unresolved_ground_truth": unresolved_gt,
        "unavailable_rows": sorted(unavailable_keys),
        "failed_rows": sorted(failed_keys),
        "coverage_reports": coverage["shards"],
        "allow_unavailable": allow_unavailable,
        "complete": complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        action="append",
        default=[],
        help="包含 *.coverage.json 的目录（可重复）",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--allow-unavailable",
        action="store_true",
        help="允许清单中存在显式记录为不可用的行（默认拒绝发布）",
    )
    args = parser.parse_args()

    report = verify(
        args.train_manifest,
        args.blind_manifest,
        args.data_dir,
        args.coverage_dir,
        args.allow_unavailable,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not report["complete"]:
        print(
            "[coverage] 拒绝发布：清单存在未标注、缺真值或明确不可用的行",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
