from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .models import Config, EpisodeResult


def hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _candidate_payload(candidate) -> dict:
    return {
        "start_s": round(candidate.start_s, 2),
        "end_s": round(candidate.end_s, 2),
        "start": hms(candidate.start_s),
        "end": hms(candidate.end_s),
        "duration_s": round(candidate.duration_s, 2),
        "score": candidate.score,
        "tier": candidate.tier,
        "evidence": candidate.evidence,
        "matches": [asdict(match) for match in candidate.matches],
    }


def build_report(
    feed_title: str,
    feed_url: str,
    results: list[EpisodeResult],
    cfg: Config,
) -> dict:
    episodes = []
    for result in results:
        entry = {
            "id": result.episode.id,
            "title": result.episode.title,
            "audio_url": result.episode.audio_url,
            "published": result.episode.published,
            "duration_s": round(result.duration_s, 2),
            "duration": hms(result.duration_s),
            "audio": {
                "id": result.audio_id,
                "id_kind": result.audio_id_kind,
                "source_sha256": result.source_sha256,
            },
            "error": result.error,
            "candidates": [_candidate_payload(candidate) for candidate in result.candidates],
        }
        episodes.append(entry)
    return {
        "tool": "dai-detect",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feed": {"title": feed_title, "url": feed_url},
        "config": asdict(cfg),
        "episodes": episodes,
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(feed_title: str, results: list[EpisodeResult]) -> None:
    titles = {result.episode.id: result.episode.title for result in results}
    print("")
    print(f"=== DAI 广告检测结果: {feed_title or '本地文件'} ===")
    for index, result in enumerate(results, start=1):
        print("")
        print(f"[{index}] {result.episode.title}  ({hms(result.duration_s)})")
        if result.error:
            print(f"    失败: {result.error}")
            continue
        if not result.candidates:
            print("    未发现候选区间")
            continue
        for candidate in result.candidates:
            label = "重复指纹" if candidate.tier == "repeated" else "插入段(未复核)"
            line = (
                f"    * {hms(candidate.start_s)} - {hms(candidate.end_s)}"
                f"  {candidate.duration_s:5.1f}s  score {candidate.score:.2f}  [{label}]"
            )
            if candidate.matches:
                first = candidate.matches[0]
                matched_title = titles.get(first.episode_id, first.episode_id)
                extra = f" 等 {len(candidate.matches)} 处" if len(candidate.matches) > 1 else ""
                line += f"  <- {matched_title[:24]} {hms(first.start_s)}{extra}"
            print(line)
    failures = sum(1 for result in results if result.error)
    total = sum(len(result.candidates) for result in results)
    print("")
    print(f"共 {len(results)} 集，{total} 个候选，{failures} 集失败。")
