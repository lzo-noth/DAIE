#!/usr/bin/env python3
"""Annotate release-hosted podcast audio with ASR + LLM ad spans.

Input manifest lines::

    {"audio_url": "https://github.com/.../release/...mp3", "feed": "Show A", "episode": "Ep 1"}

Each audio is snapshotted to preserve its exact DAI variant, transcribed with
faster-whisper, classified by an LLM, and written as a full-supervision training
row with audio identity. ``labels_source`` defaults to ``llm_weak``: these
labels are useful for training and pre-annotation, but they are NOT human
ground truth and must not be reported as independent performance acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.llm import classify_transcript_segments, merge_ad_segments
from scripts.snapshot_audio import snapshot


def _client_and_model(model: str):
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if gemini_key:
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        api_key = gemini_key
    elif openai_key:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        api_key = openai_key
    else:
        return None, ""
    from openai import OpenAI

    client = (
        OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0)
        if base_url
        else OpenAI(api_key=api_key, timeout=120.0, max_retries=0)
    )
    return client, model_name


GEMINI_MODEL_CANDIDATES = (
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
)


def _probe_model(client, model: str) -> str:
    """Return "" when the model answers, else a short error string."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def _transcribe(path: Path, model, language: str) -> list[dict]:
    segments, _ = model.transcribe(
        str(path),
        language=language,
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    return [
        {
            "start_s": round(float(segment.start), 3),
            "end_s": round(float(segment.end), 3),
            "text": segment.text.strip(),
        }
        for segment in segments
        if segment.text.strip()
    ]


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
        except json.JSONDecodeError:
            # 中断写入的残行；其余行仍然可用。
            continue
    return rows


def _episode_key(row: dict) -> tuple[str, str]:
    return (str(row.get("feed") or "unknown"), str(row.get("episode") or ""))


def _existing_snapshot(snapshot_dir: Path, digest: str) -> Path | None:
    if not digest:
        return None
    for candidate in snapshot_dir.glob(f"{digest}*"):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, help="release 音频清单 JSONL（--pick-model 时可省略）")
    parser.add_argument("--annotations-out", type=Path, default=Path("out/release_annotations.jsonl"))
    parser.add_argument("--transcripts-out", type=Path, default=Path("out/release_transcripts.jsonl"))
    parser.add_argument("--snapshot-dir", type=Path, default=Path("out/release_snapshots"))
    parser.add_argument("--ground-truth-dir", type=Path, help="为盲测集写出 LLM/人工标签真值 JSON")
    parser.add_argument("--blind-manifest-out", type=Path, help="写出 batch blind 清单")
    parser.add_argument("--labels-source", default="llm_weak", help="llm_weak 或 human")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--model", default="", help="LLM 模型名")
    parser.add_argument(
        "--pick-model", action="store_true", help="探测可用 LLM 模型，成功时打印 MODEL=<name>"
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--request-interval", type=float, default=0.0, help="LLM 批次之间等待秒数（限速）"
    )
    parser.add_argument(
        "--resume", action="store_true", help="复用已有 transcripts/annotations，跳过已完成集"
    )
    parser.add_argument("--max-gap", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--allow-empty-ads", action="store_true")
    args = parser.parse_args()

    client, llm_model = _client_and_model(args.model)
    if client is None:
        print("缺少 GEMINI_API_KEY/OPENAI_API_KEY，无法生成 LLM 标注", file=sys.stderr)
        return 2
    if args.pick_model:
        candidates = [llm_model]
        if os.environ.get("GEMINI_API_KEY", "").strip():
            candidates += [name for name in GEMINI_MODEL_CANDIDATES if name != llm_model]
        for candidate in candidates:
            error = _probe_model(client, candidate)
            if not error:
                print(f"model {candidate}: ok", flush=True)
                print(f"MODEL={candidate}", flush=True)
                return 0
            print(f"model {candidate}: {error}", flush=True)
        print("没有可用的 LLM 模型", file=sys.stderr)
        return 1
    if args.manifest is None:
        parser.error("--manifest 必填")

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    from faster_whisper import WhisperModel

    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type=args.compute_type)
    args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
    args.transcripts_out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    done_annotations = (
        {_episode_key(row) for row in _load_jsonl(args.annotations_out)}
        if args.resume
        else set()
    )
    cached_transcripts = (
        {_episode_key(row): row for row in _load_jsonl(args.transcripts_out)}
        if args.resume
        else {}
    )
    annotation_handle = args.annotations_out.open(mode, encoding="utf-8")
    transcript_handle = args.transcripts_out.open(mode, encoding="utf-8")
    if args.blind_manifest_out:
        args.blind_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        existing_blind = (
            {_episode_key(row) for row in _load_jsonl(args.blind_manifest_out)}
            if args.resume
            else set()
        )
        blind_handle = args.blind_manifest_out.open(mode, encoding="utf-8")
    else:
        existing_blind = set()
        blind_handle = None
    if args.ground_truth_dir:
        args.ground_truth_dir.mkdir(parents=True, exist_ok=True)

    total_ads = 0
    total_episodes = 0
    failures: list[dict] = []
    for row in rows:
        source = str(row.get("audio_url") or row.get("audio") or "")
        if not source:
            continue
        feed = str(row.get("feed") or "unknown")
        episode = str(row.get("episode") or Path(source).stem)
        key = (feed, episode)
        if key in done_annotations:
            print(f"[resume] 跳过已完成: {feed}/{episode}", flush=True)
            continue
        print(f"[annotate] {feed} / {episode}", flush=True)
        cached = cached_transcripts.get(key)
        if cached:
            segments = list(cached.get("segments") or [])
            audio_path = str(cached.get("audio") or "")
            digest = str(cached.get("source_sha256") or "")
            audio_id = str(cached.get("audio_id") or "")
            audio_id_kind = str(cached.get("audio_id_kind") or "")
            duration = float(cached.get("duration_s") or 0.0)
            classified = not cached.get("llm_error")
            if classified:
                results = dict(cached.get("results") or {})
                ads = list(cached.get("ads") or [])
                print(f"[resume] 复用转写: {feed}/{episode}", flush=True)
            else:
                print(f"[resume] 复用转写并重试 LLM 分类: {feed}/{episode}", flush=True)
        else:
            digest_hint = str(row.get("source_sha256") or "")
            reused = _existing_snapshot(args.snapshot_dir, digest_hint)
            if reused:
                target, digest = reused, digest_hint
                print(f"[resume] 复用音频快照: {target.name}", flush=True)
            else:
                target, digest = snapshot(source, args.snapshot_dir, None, False)
            # Reuse snapshot_audio's identity extraction without a second download.
            from dai_detect.single_episode import extract_features

            mono, forensic = extract_features(str(target), proxy=None)
            duration = min(
                float(mono.get("duration_s") or 0.0),
                float(forensic.get("duration_s") or 0.0),
            )
            audio_path = str(target)
            audio_id = str(mono.get("audio_id") or "")
            audio_id_kind = str(mono.get("audio_id_kind") or "")
            segments = _transcribe(target, whisper, args.language)
            classified = False
        identity = {
            "audio": audio_path,
            "feed": feed,
            "episode": episode,
            "source_sha256": digest,
            "audio_id": audio_id,
            "audio_id_kind": audio_id_kind,
            "duration_s": round(duration, 2),
            "labels_source": args.labels_source,
        }
        if not classified:
            try:
                results = classify_transcript_segments(
                    segments,
                    client,
                    llm_model,
                    args.batch_size,
                    args.request_interval,
                )
            except Exception as exc:  # noqa: BLE001
                # 单集失败不终止整批：转写立即落盘，配额恢复后 --resume 只重跑 LLM。
                failures.append(
                    {"feed": feed, "episode": episode, "error": f"{type(exc).__name__}: {exc}"[:300]}
                )
                transcript_handle.write(
                    json.dumps(
                        {
                            **identity,
                            "segments": segments,
                            "results": {},
                            "ads": [],
                            "llm_error": f"{type(exc).__name__}: {exc}"[:300],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                transcript_handle.flush()
                print(f"[fail] {feed}/{episode}: LLM 分类失败，已保留转写", flush=True)
                continue
            ads = merge_ad_segments(
                segments, results, max_gap_s=args.max_gap, min_confidence=args.min_confidence
            )
            transcript_handle.write(
                json.dumps(
                    {
                        **identity,
                        "segments": segments,
                        "results": results,
                        "ads": ads,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            transcript_handle.flush()
        if (
            not ads
            and not args.allow_empty_ads
            and not (blind_handle and args.ground_truth_dir)
        ):
            print(f"[skip] {feed}/{episode}: LLM 未找到第三方插入广告", flush=True)
            continue
        ads = [
            {**ad, "type": "third_party_inserted"}
            for ad in ads
        ]
        annotation = {**identity, "ads": ads}
        annotation_handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
        annotation_handle.flush()
        if blind_handle and args.ground_truth_dir:
            safe = "".join(
                character if character.isalnum() or character in "-_." else "_"
                for character in f"{feed}_{episode}"
            )
            gt_path = args.ground_truth_dir / f"{safe}.json"
            gt_path.write_text(
                json.dumps(
                    {
                        "duration_s": round(duration, 2),
                        "ads": ads,
                        "source_sha256": identity["source_sha256"],
                        "audio_id": identity["audio_id"],
                        "audio_id_kind": identity["audio_id_kind"],
                        "labels_source": args.labels_source,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if key not in existing_blind:
                blind_handle.write(
                    json.dumps(
                        {
                            "audio": audio_path,
                            "feed": feed,
                            "episode": episode,
                            "ground_truth": str(gt_path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                blind_handle.flush()
        total_ads += len(ads)
        total_episodes += 1
        print(f"[annotate] {feed}/{episode}: ads={len(ads)} segments={len(segments)}", flush=True)

    annotation_handle.close()
    transcript_handle.close()
    if blind_handle:
        blind_handle.close()
    print(
        json.dumps(
            {
                "episodes": total_episodes,
                "ads": total_ads,
                "labels_source": args.labels_source,
                "annotations": str(args.annotations_out),
                "transcripts": str(args.transcripts_out),
                "failed": failures,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
