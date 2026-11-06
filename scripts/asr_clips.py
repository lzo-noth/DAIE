#!/usr/bin/env python3
"""Transcribe candidate clips and flag ad-like language."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

AD_TERMS = [
    "sponsored by",
    "brought to you by",
    "support for this podcast",
    "support comes from",
    "this episode is sponsored",
    "our sponsor",
    "advertisement",
    "ad break",
    "paid for by",
    "partner of",
    "use code",
    "promo code",
    "offer code",
    "discount code",
    "coupon code",
    "free trial",
    "terms apply",
    "sign up",
    "percent off",
    "dot com",
    ".com",
    "本节目由",
    "赞助",
    "优惠码",
    "折扣",
    "广告",
]


def ad_terms(text: str) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in AD_TERMS if term in lowered})


def load_audio(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-200:])
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def process_report(report: dict, clips_dir: Path, model, language: str) -> tuple[int, int]:
    total = 0
    confirmed = 0
    for episode in report.get("episodes", []):
        for candidate in episode.get("candidates", []):
            evidence = candidate.get("evidence", {})
            clip_name = evidence.get("clip")
            if not clip_name:
                continue
            clip_path = clips_dir / clip_name
            if not clip_path.exists():
                continue
            try:
                audio = load_audio(clip_path)
                segments, _ = model.transcribe(
                    audio,
                    language=language,
                    beam_size=1,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except Exception as exc:  # noqa: BLE001 - one bad clip must not fail the feed
                evidence["asr_error"] = f"{type(exc).__name__}: {exc}"[:300]
                continue
            terms = ad_terms(text)
            evidence["asr_text"] = text[:400]
            evidence["ad_terms"] = terms
            evidence["ad_language"] = bool(terms)
            if terms:
                candidate["score"] = round(min(0.99, float(candidate.get("score", 0.0)) + 0.06), 3)
                confirmed += 1
            total += 1
    report["asr"] = {"model": getattr(model, "model_size_or_path", ""), "language": language, "clips": total, "ad_language": confirmed}
    return total, confirmed


def _targets(report: Path | None, root: Path | None, clips_dir: Path | None) -> list[tuple[Path, Path]]:
    if root is not None:
        return [(path, path.parent / "clips") for path in sorted(root.rglob("report.json"))]
    if report is not None:
        return [(report, clips_dir or report.parent / "clips")]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--model", default="base.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    targets = _targets(args.report, args.reports_root, args.clips_dir)
    if not targets:
        print("no report targets")
        return 0

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)
    for report_path, clips_dir in targets:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"{report_path}: read failed {exc}")
            continue
        total, confirmed = process_report(report, clips_dir, model, args.language)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report_path.parent.name}: clips={total} ad_language={confirmed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
