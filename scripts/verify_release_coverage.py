#!/usr/bin/env python3
"""Refuse to publish a release dataset unless every manifest row is complete.

The annotation pipeline is a matrix of shards. Each shard can partially
succeed, so the existence of a merged artifact proves nothing about coverage:
a dead audio URL, a failed LLM call or a damaged snapshot can leave rows
missing while every job still exits successfully.

Coverage is checked by *content identity* ``(feed, episode, source_sha256)``,
not by display names. DAI serves different ad variants behind the same URL, so
a label set that covers every episode name can still be attached to the wrong
audio bytes. The blind side gets two extra checks:

  * ``blind_manifest`` and ``blind_generated`` must describe the same episode
    set — otherwise the evaluation silently shrinks;
  * every generated ground-truth file must carry the same ``source_sha256`` as
    the manifest row it is scored against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

Identity = tuple[str, str, str]


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


def _identity(row: dict) -> Identity:
    return (
        str(row.get("feed") or "unknown"),
        str(row.get("episode") or ""),
        str(row.get("source_sha256") or ""),
    )


def _short(identity: Identity) -> dict:
    return {"feed": identity[0], "episode": identity[1], "source_sha256": identity[2]}


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
    unavailable_keys = {_identity(row) for row in coverage["unavailable"]}
    failed_keys = {_identity(row) for row in coverage["failed"]}

    train_expected = {_identity(row) for row in _load_jsonl(train_manifest)}
    train_actual = {_identity(row) for row in _load_jsonl(data_dir / "annotations.jsonl")}
    train_missing = sorted(train_expected - train_actual)

    blind_expected = {_identity(row) for row in _load_jsonl(blind_manifest)}
    blind_actual = {
        _identity(row) for row in _load_jsonl(data_dir / "blind_annotations.jsonl")
    }
    blind_missing = sorted(blind_expected - blind_actual)

    generated_rows = _load_jsonl(data_dir / "blind_generated.jsonl")
    generated = {_identity(row) for row in generated_rows}
    generated_missing = sorted(blind_expected - generated)
    generated_extra = sorted(generated - blind_expected)

    gt_dir = data_dir / "blind_gt"
    gt_missing: list[Identity] = []
    gt_sha_mismatch: list[dict] = []
    for row in generated_rows:
        gt_path = gt_dir / Path(str(row.get("ground_truth") or "")).name
        if not gt_path.exists():
            gt_missing.append(_identity(row))
            continue
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        audio_sha = str(payload.get("source_sha256") or "")
        if audio_sha != str(row.get("source_sha256") or ""):
            gt_sha_mismatch.append(
                {
                    "feed": row.get("feed"),
                    "episode": row.get("episode"),
                    "manifest_sha256": row.get("source_sha256"),
                    "ground_truth_sha256": audio_sha,
                }
            )

    accounted = unavailable_keys | failed_keys

    def unresolved(items: list[Identity]) -> list[Identity]:
        return [item for item in items if item not in accounted]

    unresolved_train = unresolved(train_missing)
    unresolved_blind = unresolved(blind_missing)
    unresolved_generated = unresolved(generated_missing)
    unresolved_gt = unresolved(gt_missing)

    complete = not (
        unresolved_train
        or unresolved_blind
        or unresolved_generated
        or unresolved_gt
        or generated_extra
        or gt_sha_mismatch
    )
    if not allow_unavailable and (unavailable_keys or failed_keys):
        complete = False

    return {
        "train_manifest_rows": len(train_expected),
        "train_annotated_rows": len(train_expected & train_actual),
        "train_missing": [_short(item) for item in train_missing],
        "blind_manifest_rows": len(blind_expected),
        "blind_annotated_rows": len(blind_expected & blind_actual),
        "blind_missing": [_short(item) for item in blind_missing],
        "blind_generated_rows": len(generated),
        "blind_generated_missing": [_short(item) for item in generated_missing],
        "blind_generated_extra": [_short(item) for item in generated_extra],
        "blind_ground_truth_missing": [_short(item) for item in gt_missing],
        "blind_ground_truth_sha_mismatch": gt_sha_mismatch,
        "unresolved": [_short(item) for item in unresolved_train + unresolved_blind],
        "unresolved_ground_truth": [
            _short(item) for item in unresolved_generated + unresolved_gt
        ],
        "unavailable_rows": [_short(item) for item in sorted(unavailable_keys)],
        "failed_rows": [_short(item) for item in sorted(failed_keys)],
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
            "[coverage] 拒绝发布：清单存在未标注、缺真值、真值哈希不符或明确不可用的行",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
