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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.feed_identity import canonical_feed_key
from dai_detect.llm import classify_transcript_segments, merge_ad_segments
from dai_detect.llm_keys import AllKeysExhausted, RotatingChatClient, load_keys
from scripts.snapshot_audio import snapshot


def _client_and_model(model: str):
    gemini_keys = load_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
    openai_keys = load_keys("OPENAI_API_KEYS", "OPENAI_API_KEY")
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    use_gemini = bool(gemini_keys) and provider in ("", "auto", "gemini")
    use_openai = bool(openai_keys) and (
        provider in ("", "auto", "openai") or not use_gemini
    )
    from openai import OpenAI

    if use_gemini:
        base_url = os.environ.get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        model_name = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        keys = gemini_keys
    elif use_openai:
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        model_name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        keys = openai_keys
    else:
        return None, ""

    def factory(key: str):
        return (
            OpenAI(api_key=key, base_url=base_url, timeout=120.0, max_retries=0)
            if base_url
            else OpenAI(api_key=key, timeout=120.0, max_retries=0)
        )

    client = RotatingChatClient(factory, keys, start=_shard_offset())
    print(f"[llm] key 池: {len(keys)} 个可用 key，起始 key #{client.active_key_number}", flush=True)
    return client, model_name


def _shard_offset() -> int:
    """1-based shard index from CI, so parallel shards start on different keys."""
    try:
        return max(0, int(os.environ.get("SHARD_INDEX", "1")) - 1)
    except ValueError:
        return 0


GEMINI_MODEL_CANDIDATES = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
)
OPENAI_MODEL_CANDIDATES = (
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5-mini",
)


def _candidate_models(llm_model: str) -> list[str]:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "openai" or (provider in ("", "auto") and not os.environ.get("GEMINI_API_KEY", "").strip()):
        pool = OPENAI_MODEL_CANDIDATES
    else:
        pool = GEMINI_MODEL_CANDIDATES
    return [llm_model] + [name for name in pool if name != llm_model]


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


def _identity_key(row: dict) -> tuple[str, str, str]:
    """Cache identity: feed + episode + the exact audio bytes.

    DAI can serve a different ad variant for the same episode and URL, so a
    feed/episode key alone would reuse the old variant's ASR and labels for a
    new recording. That produces a dataset that is 100% "covered" and entirely
    wrong.
    """
    return (*_episode_key(row), str(row.get("source_sha256") or ""))


def _clamp_ads(ads: list[dict], duration: float, min_span_s: float = 1.0) -> list[dict]:
    """Trim ad spans to the decoded duration.

    Whisper segments can end a few seconds past the audio length; those spans
    fail downstream validation, so clamp instead of rejecting the episode.
    """
    if duration <= 0:
        return ads
    clamped = []
    for ad in ads:
        start = max(0.0, float(ad.get("start_s") or 0.0))
        end = min(duration, float(ad.get("end_s") or 0.0))
        if end - start < min_span_s:
            continue
        clamped.append({**ad, "start_s": round(start, 3), "end_s": round(end, 3)})
    return clamped


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
        "--provider",
        default="",
        help="强制 provider：gemini / openai（留空按 LLM_PROVIDER 或自动优先 Gemini）",
    )
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
    parser.add_argument("--shard-index", type=int, default=0, help="分片序号（从 0 开始）")
    parser.add_argument("--shard-count", type=int, default=1, help="分片总数")
    parser.add_argument("--max-gap", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--allow-empty-ads", action="store_true")
    parser.add_argument(
        "--max-consecutive-llm-failures",
        type=int,
        default=2,
        help="连续多少次 LLM 失败后中止分片（配额耗尽时避免空转数小时），0 表示不限制",
    )
    args = parser.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    client, llm_model = _client_and_model(args.model)
    if client is None:
        print("缺少 GEMINI_API_KEY/OPENAI_API_KEY，无法生成 LLM 标注", file=sys.stderr)
        return 2
    if args.pick_model:
        candidates = _candidate_models(llm_model)
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
    if args.shard_count > 1:
        rows = [
            row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index
        ]

    from faster_whisper import WhisperModel

    print(
        f"[config] model={llm_model} batch={args.batch_size} "
        f"interval={args.request_interval}s labels={args.labels_source} "
        f"shard={args.shard_index + 1}/{args.shard_count} resume={args.resume}",
        flush=True,
    )
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type=args.compute_type)
    args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
    args.transcripts_out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    done_annotations = (
        {_identity_key(row) for row in _load_jsonl(args.annotations_out)}
        if args.resume
        else set()
    )
    cached_transcripts = (
        {_identity_key(row): row for row in _load_jsonl(args.transcripts_out)}
        if args.resume
        else {}
    )
    if args.resume:
        # LLM 失败的集在 annotations 里可能是空 ads（复核脚本重建过），
        # 必须重新进入 LLM 分类，而不是当成已完成跳过。
        done_annotations -= {
            key for key, row in cached_transcripts.items() if row.get("llm_error")
        }
    annotation_handle = args.annotations_out.open(mode, encoding="utf-8")
    transcript_handle = args.transcripts_out.open(mode, encoding="utf-8")
    if args.blind_manifest_out:
        args.blind_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        existing_blind = (
            {
                _identity_key(row)
                for row in _load_jsonl(args.blind_manifest_out)
                # 缺 ground_truth 的行不是有效索引（旧版本混进来的转写行），
                # 必须重写成正式索引行。
                if row.get("ground_truth")
            }
            if args.resume
            else set()
        )
        blind_handle = args.blind_manifest_out.open(mode, encoding="utf-8")
    else:
        existing_blind = set()
        blind_handle = None
    if args.ground_truth_dir:
        args.ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # 已完成的集会被跳过，但训练/盲测仍要解码音频：先补齐缺失快照。
    prefetch_failures: dict[tuple[str, str, str], str] = {}
    for row in rows:
        source = str(row.get("audio_url") or row.get("audio") or "")
        digest_hint = str(row.get("source_sha256") or "")
        if not source or not digest_hint:
            continue
        if _existing_snapshot(args.snapshot_dir, digest_hint):
            continue
        sibling = args.snapshot_dir.parent / (
            "snapshots" if args.snapshot_dir.name == "blind_snapshots" else "blind_snapshots"
        )
        if _existing_snapshot(sibling, digest_hint):
            continue
        try:
            target, _ = snapshot(source, args.snapshot_dir, None, False)
            print(f"[snapshot] 补齐: {target.name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 单集下载失败不应终止整批
            detail = f"{type(exc).__name__}: {exc}"
            prefetch_failures[_identity_key(row)] = detail
            print(f"[snapshot] 补齐失败，记录为不可用: {detail}", flush=True)

    total_ads = 0
    total_episodes = 0
    failures: list[dict] = []
    unavailable: list[dict] = []
    consecutive_llm_failures = 0
    aborted_reason = ""
    for row in rows:
        source = str(row.get("audio_url") or row.get("audio") or "")
        if not source:
            continue
        feed = str(row.get("feed") or "unknown")
        episode = str(row.get("episode") or Path(source).stem)
        key = _identity_key(row)
        # 盲测的真值文件不在缓存里（artifact 里也没有），所以盲测行不能因为
        # "标注已完成"就整集跳过，否则本次 artifact 会一代代丢失 blind_gt。
        if key in done_annotations and not (blind_handle and args.ground_truth_dir):
            print(f"[resume] 跳过已完成（同一音频哈希）: {feed}/{episode}", flush=True)
            continue
        if key in prefetch_failures:
            unavailable.append(
                {
                    "feed": feed,
                    "episode": episode,
                    "source_sha256": key[2],
                    "audio": source,
                    "reason": prefetch_failures[key],
                }
            )
            print(f"[unavailable] {feed}/{episode}: 音频快照不可用，跳过", flush=True)
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
            # 转写可能来自旧 artifact，但音频快照未必在本次目录里；
            # 盲测/训练都要解码这份音频，缺失时必须补下载。
            if digest:
                existing = _existing_snapshot(args.snapshot_dir, digest)
                if existing:
                    audio_path = str(existing)
                else:
                    try:
                        target, digest = snapshot(source, args.snapshot_dir, None, False)
                    except Exception as exc:  # noqa: BLE001
                        detail = f"{type(exc).__name__}: {exc}"
                        unavailable.append(
                            {
                                "feed": feed,
                                "episode": episode,
                                "audio": source,
                                "reason": detail,
                            }
                        )
                        print(f"[unavailable] {feed}/{episode}: 快照补下载失败 {detail}", flush=True)
                        continue
                    audio_path = str(target)
                    print(f"[resume] 补齐音频快照: {target.name}", flush=True)
            classified = not cached.get("llm_error")
            if classified:
                results = dict(cached.get("results") or {})
                ads = _clamp_ads(list(cached.get("ads") or []), duration)
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
                try:
                    target, digest = snapshot(source, args.snapshot_dir, None, False)
                except Exception as exc:  # noqa: BLE001
                    detail = f"{type(exc).__name__}: {exc}"
                    unavailable.append(
                        {
                            "feed": feed,
                            "episode": episode,
                            "audio": source,
                            "reason": detail,
                        }
                    )
                    print(f"[unavailable] {feed}/{episode}: 音频下载失败 {detail}", flush=True)
                    continue
            # 清单里的 source_sha256 是抽样时那份音频；重新下载若拿到另一个
            # DAI 变体，就不能把它的标注冒充成清单音频的真值。
            if digest_hint and digest != digest_hint:
                detail = (
                    f"audio_sha256_mismatch: manifest={digest_hint[:16]}… "
                    f"downloaded={digest[:16]}…"
                )
                unavailable.append(
                    {
                        "feed": feed,
                        "episode": episode,
                        "source_sha256": digest,
                        "audio": source,
                        "reason": detail,
                    }
                )
                print(f"[unavailable] {feed}/{episode}: {detail}", flush=True)
                continue
            # Reuse snapshot_audio's identity extraction without a second download.
            from dai_detect.single_episode import extract_features

            try:
                mono, forensic = extract_features(str(target), proxy=None)
                duration = min(
                    float(mono.get("duration_s") or 0.0),
                    float(forensic.get("duration_s") or 0.0),
                )
                asr_began = time.time()
                segments = _transcribe(target, whisper, args.language)
            except Exception as exc:  # noqa: BLE001 - 单集损坏不应拖垮整个分片
                detail = f"{type(exc).__name__}: {exc}"
                unavailable.append(
                    {
                        "feed": feed,
                        "episode": episode,
                        "audio": source,
                        "reason": detail,
                    }
                )
                print(f"[unavailable] {feed}/{episode}: 解码/转写失败 {detail}", flush=True)
                continue
            audio_path = str(target)
            audio_id = str(mono.get("audio_id") or "")
            audio_id_kind = str(mono.get("audio_id_kind") or "")
            print(
                f"[asr] {feed}/{episode}: {len(segments)} 段, "
                f"{time.time() - asr_began:.0f}s",
                flush=True,
            )
            classified = False
        identity = {
            "audio": audio_path,
            "feed": feed,
            "feed_url": str(row.get("feed_url") or ""),
            "episode": episode,
            "source_sha256": digest,
            "audio_id": audio_id,
            "audio_id_kind": audio_id_kind,
            "duration_s": round(duration, 2),
            "labels_source": args.labels_source,
        }
        if not classified:
            began = time.time()
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
                detail = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}"
                failures.append({"feed": feed, "episode": episode, "error": detail})
                consecutive_llm_failures += 1
                print(
                    f"[fail] {feed}/{episode}: LLM 分类失败，已保留转写; "
                    f"耗时 {time.time() - began:.0f}s; 错误 {detail}",
                    flush=True,
                )
                transcript_handle.write(
                    json.dumps(
                        {
                            **identity,
                            "segments": segments,
                            "results": {},
                            "ads": [],
                            "llm_error": detail[:500],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                transcript_handle.flush()
                if isinstance(exc, AllKeysExhausted):
                    # 整个 key 池都被拒绝：继续重试没有意义，立即停下，配额
                    # 恢复后用 --resume 只补 LLM 部分。
                    aborted_reason = f"key 池已耗尽：{detail}"
                    print(f"[abort] {aborted_reason}", flush=True)
                    break
                if (
                    args.max_consecutive_llm_failures > 0
                    and consecutive_llm_failures >= args.max_consecutive_llm_failures
                ):
                    aborted_reason = (
                        f"连续 {consecutive_llm_failures} 集 LLM 调用失败，"
                        f"疑似配额耗尽：{detail}"
                    )
                    print(f"[abort] {aborted_reason}", flush=True)
                    break
                continue
            consecutive_llm_failures = 0
            ads = merge_ad_segments(
                segments, results, max_gap_s=args.max_gap, min_confidence=args.min_confidence
            )
            ads = _clamp_ads(ads, duration)
            print(
                f"[llm] {feed}/{episode}: 分类完成 {time.time() - began:.0f}s, "
                f"广告段 {len(ads)}",
                flush=True,
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
            failures.append(
                {
                    "feed": feed,
                    "episode": episode,
                    "error": "no_ads_and_allow_empty_ads_disabled",
                }
            )
            continue
        ads = [
            {
                **ad,
                "type": "third_party_ad",
                "ad_content_type": "third_party_ad",
                # 纯文本判定无法证明广告是 DAI 动态插入还是原生剪辑进母版；
                # 也不该把"第三方广告"直接说成"已确认的动态插入"。
                "ad_production_style": "unknown",
                "insertion_mechanism": "unknown",
            }
            for ad in ads
        ]
        annotation = {**identity, "ads": ads}
        if key not in done_annotations:
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
                            "feed_url": str(row.get("feed_url") or ""),
                            "feed_key": canonical_feed_key(
                                feed, str(row.get("feed_url") or "")
                            ),
                            "episode": episode,
                            "source_sha256": digest,
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

    # 覆盖率是发布门禁的输入：任何未产出标注、未产出真值或快照不可用的行
    # 都必须显式记账，不能靠"退出码 0"掩盖。身份键包含音频哈希，
    # 同一集的另一个 DAI 变体不算"已覆盖"。
    produced = {_identity_key(row) for row in _load_jsonl(args.annotations_out)}
    expected = {_identity_key(row) for row in rows}
    covered: list[dict] = []
    for row in rows:
        key = _identity_key(row)
        status = "annotated" if key in produced else "missing"
        covered.append(
            {
                "feed": key[0],
                "episode": key[1],
                "source_sha256": key[2],
                "status": status,
            }
        )
    unavailable_keys = {_identity_key(row) for row in unavailable}
    incomplete = [
        item
        for item in covered
        if item["status"] != "annotated"
        and (item["feed"], item["episode"], item["source_sha256"]) not in unavailable_keys
    ]
    coverage = {
        "annotations": str(args.annotations_out),
        "manifest_rows": len(expected),
        "annotated_rows": len(expected & produced),
        "unavailable_rows": len(unavailable),
        "failed_rows": len(failures),
        "missing_rows": len(incomplete),
        "complete": not incomplete and not unavailable,
        "aborted": aborted_reason,
        "unavailable": unavailable,
        "failed": failures,
    }
    coverage_path = args.annotations_out.with_suffix(".coverage.json")
    coverage_path.write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "episodes": total_episodes,
                "ads": total_ads,
                "labels_source": args.labels_source,
                "annotations": str(args.annotations_out),
                "transcripts": str(args.transcripts_out),
                "coverage": str(coverage_path),
                "manifest_rows": coverage["manifest_rows"],
                "annotated_rows": coverage["annotated_rows"],
                "unavailable_rows": coverage["unavailable_rows"],
                "failed_rows": coverage["failed_rows"],
                "missing_rows": coverage["missing_rows"],
                "failed": failures,
                "unavailable": unavailable,
            },
            ensure_ascii=False,
        )
    )
    if incomplete or unavailable:
        print(
            "[coverage] 本分片没有被完整处理：存在失败或不可用行。"
            "发布流程应拒绝发布；请修复后重跑，或从清单中显式移除该集。",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
