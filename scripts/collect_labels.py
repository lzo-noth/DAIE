#!/usr/bin/env python3
"""Collect LLM-labelled candidates into a persistent, deduplicated label store."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LABEL_STORE_VERSION = 1


def _episode_index(report: dict) -> tuple[dict[str, dict], list[dict]]:
    episodes = report.get("episodes") or []
    return {episode.get("title"): episode for episode in episodes}, episodes


def _candidate_for(label: dict, episodes: list[dict]) -> dict:
    raw_id = str(label.get("id") or "")
    if ":" not in raw_id:
        return {}
    try:
        episode_index, candidate_index = (int(part) for part in raw_id.split(":", 1))
    except ValueError:
        return {}
    if 0 <= episode_index < len(episodes):
        candidates = episodes[episode_index].get("candidates") or []
        if 0 <= candidate_index < len(candidates):
            return candidates[candidate_index]
    return {}


def _row(report: dict, label: dict, run_id: str) -> dict | None:
    feed = (report.get("feed") or {}).get("title") or ""
    by_title, episodes = _episode_index(report)
    episode = by_title.get(label.get("episode")) or {}
    candidate = _candidate_for(label, episodes)
    evidence = candidate.get("evidence") or {}
    features = label.get("features") or {}
    start = label.get("start_s")
    end = label.get("end_s")
    if start is None or end is None:
        return None
    duration = episode.get("duration_s")
    return {
        "version": LABEL_STORE_VERSION,
        "run_id": run_id,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feed": feed,
        "episode": label.get("episode") or episode.get("title") or "",
        "episode_id": episode.get("id"),
        "episode_duration_s": duration,
        "candidate_id": str(label.get("id") or ""),
        "tier": label.get("tier"),
        "start_s": round(float(start), 2),
        "end_s": round(float(end), 2),
        "score_before_llm": label.get("score_before_llm"),
        "heuristic_score": label.get("heuristic_score"),
        "features": features,
        "ad_language": bool(evidence.get("ad_language") or label.get("ad_language")),
        "ad_terms": evidence.get("ad_terms") or label.get("ad_terms") or [],
        "asr_text": evidence.get("asr_text") or label.get("asr_text") or "",
        "matched_episodes": int(features.get("matched_episodes") or evidence.get("matched_episodes") or 0),
        "votes": int(features.get("votes") or evidence.get("votes") or 0),
        "llm": label.get("llm"),
        "is_ad": bool(label.get("is_ad")),
    }


def _key(row: dict) -> str:
    return "|".join(
        [
            str(row.get("feed") or ""),
            str(row.get("episode") or ""),
            f"{float(row.get('start_s') or 0.0):.1f}",
            f"{float(row.get('end_s') or 0.0):.1f}",
            str(row.get("tier") or ""),
        ]
    )


def _load_store(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("is_ad") is not None:
                rows[_key(row)] = row
    except OSError:
        return rows
    return rows


def collect(reports_root: Path, store: Path, run_id: str) -> dict:
    rows = _load_store(store)
    reports = sorted(reports_root.rglob("report.json"))
    added = 0
    updated = 0
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for label in report.get("llm_labels") or []:
            row = _row(report, label, run_id)
            if row is None:
                continue
            key = _key(row)
            previous = rows.get(key)
            if previous is None:
                added += 1
            elif previous.get("is_ad") != row.get("is_ad"):
                updated += 1
            rows[key] = row
    store.parent.mkdir(parents=True, exist_ok=True)
    tmp = store.with_name(f"{store.stem}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for key in sorted(rows):
                handle.write(json.dumps(rows[key], ensure_ascii=False) + "\n")
        os.replace(tmp, store)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    ads = sum(1 for row in rows.values() if row.get("is_ad"))
    return {
        "reports": len(reports),
        "store_rows": len(rows),
        "added": added,
        "updated": updated,
        "ads": ads,
        "non_ads": len(rows) - ads,
        "feeds": len({row.get("feed") for row in rows.values()}),
        "episodes": len({_key({**row, "tier": ""}) for row in rows.values()}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("feedback/labels.jsonl"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args()
    stats = collect(args.reports_root, args.store, args.run_id)
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
