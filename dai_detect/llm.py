from __future__ import annotations

import json
import math
import os
import time
from typing import Any

AD_TYPES = {"third_party_ad", "self_promo", "content", "uncertain"}


def _log(message: str) -> None:
    """Emit progress to stdout so CI logs show where LLM calls stall."""
    if os.environ.get("LLM_QUIET", "").strip():
        return
    print(f"[llm] {message}", flush=True)


SYSTEM_PROMPT = """You classify transcript snippets from podcast segments that were acoustically flagged as possible ad insertions.
Return strict JSON only. For every item decide:
- is_ad: true only for a paid advertisement or sponsorship. Ordinary episode content and unpaid self-promotion are false.
- ad_type: "third_party_ad" for a paid sponsor/ad, "self_promo" for the show's own products/other episodes, "content" for normal content, "uncertain" when unclear.
- host_read: true when the ad is read by the show host in their own voice; false for produced or separately inserted creatives.
- confidence: 0.0 to 1.0.
- brand: sponsor/brand name if identifiable, else "".
- reason: at most 120 characters.
Input JSON looks like {"items":[{"id":"0:1","text":"..."}]}.
Output JSON must look like {"results":[{"id":"0:1","is_ad":false,"ad_type":"content","host_read":false,"confidence":0.9,"brand":"","reason":"..."}]}.
Include exactly one result per input id."""


SEGMENT_SYSTEM_PROMPT = """You classify timestamped transcript segments from a podcast episode.
Return strict JSON only. For every segment decide:
- is_ad: true only for a paid advertisement or sponsorship. Ordinary episode content and unpaid self-promotion are false.
- ad_type: "third_party_ad" for a paid sponsor/ad, "self_promo" for the show's own products/other episodes, "content" for normal content, "uncertain" when unclear.
- host_read: true when the ad is read by the show host in their own voice; false for produced or separately inserted creatives.
- confidence: 0.0 to 1.0.
- brand: sponsor/brand name if identifiable, else "".
- reason: at most 120 characters.
Input JSON looks like {"items":[{"id":"0","start_s":10.0,"end_s":20.0,"text":"..."}]}.
Output JSON must look like {"results":[{"id":"0","is_ad":false,"ad_type":"content","host_read":false,"confidence":0.9,"brand":"","reason":"..."}]}.
Include exactly one result per input id."""


def classify_transcript_segments(
    segments: list[dict],
    client,
    model: str,
    batch_size: int = 12,
    request_interval_s: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Classify full-episode transcript segments as ads or content.

    ``request_interval_s`` paces batches so long episodes do not exceed
    provider rate limits (Gemini free tier is 15 requests/minute).
    """
    results: dict[str, dict[str, Any]] = {}
    batches = math.ceil(len(segments) / batch_size) if segments else 0
    for start in range(0, len(segments), batch_size):
        batch_index = start // batch_size + 1
        if start and request_interval_s > 0:
            time.sleep(request_interval_s)
        batch = []
        for index in range(start, min(start + batch_size, len(segments))):
            segment = segments[index]
            batch.append(
                {
                    "id": str(index),
                    "start_s": round(float(segment.get("start_s", 0.0)), 2),
                    "end_s": round(float(segment.get("end_s", 0.0)), 2),
                    "text": str(segment.get("text") or "")[:800],
                }
            )
        expected = {item["id"] for item in batch}
        began = time.time()
        _log(f"batch {batch_index}/{batches} 开始: segments={len(batch)} model={model}")
        for parse_attempt in range(2):
            response = _create_with_retry(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps({"items": batch}, ensure_ascii=False),
                    },
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            try:
                parsed = parse_response(text, expected)
                missing = expected - set(parsed)
                results.update(parsed)
                _log(
                    f"batch {batch_index}/{batches} 完成: "
                    f"{time.time() - began:.1f}s, 解析 {len(parsed)}/{len(expected)} 条"
                    + (f", 缺少 {len(missing)} 条" if missing else "")
                )
                break
            except ValueError as exc:
                _log(
                    f"batch {batch_index}/{batches} 解析失败"
                    f"（第 {parse_attempt + 1}/2 次）: {exc}; "
                    f"响应前 200 字符: {text[:200]!r}"
                )
                if parse_attempt == 1:
                    raise
    return results


def merge_ad_segments(
    segments: list[dict],
    results: dict[str, dict[str, Any]],
    max_gap_s: float = 1.0,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Merge consecutive third-party ad segments into start/end intervals."""
    spans: list[list[float]] = []
    for index, segment in enumerate(segments):
        result = results.get(str(index))
        if not result:
            continue
        if (
            result.get("is_ad") is not True
            or result.get("ad_type") != "third_party_ad"
            or result.get("host_read")
            or float(result.get("confidence") or 0.0) < min_confidence
        ):
            continue
        start = float(segment.get("start_s", 0.0))
        end = float(segment.get("end_s", 0.0))
        if spans and start - spans[-1][1] <= max_gap_s:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [{"start_s": round(start, 3), "end_s": round(end, 3)} for start, end in spans]


def build_items(report: dict, max_text: int = 600) -> list[dict[str, str]]:
    items = []
    for episode_index, episode in enumerate(report.get("episodes", [])):
        for candidate_index, candidate in enumerate(episode.get("candidates", [])):
            text = str(candidate.get("evidence", {}).get("asr_text") or "").strip()
            if not text:
                continue
            items.append({"id": f"{episode_index}:{candidate_index}", "text": text[:max_text]})
    return items


def build_prompt(items: list[dict[str, str]]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


def parse_response(text: str, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response has no JSON object")
    payload = json.loads(cleaned[start:end + 1])
    raw_results = payload.get("results")
    if raw_results is None and isinstance(payload, list):
        raw_results = payload
    if not isinstance(raw_results, list):
        raise ValueError("LLM response has no results array")
    results: dict[str, dict[str, Any]] = {}
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id", ""))
        if item_id not in expected_ids:
            continue
        is_ad = raw.get("is_ad") is True
        ad_type = raw.get("ad_type")
        if ad_type not in AD_TYPES:
            ad_type = "third_party_ad" if is_ad else "content"
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        results[item_id] = {
            "is_ad": is_ad,
            "ad_type": ad_type,
            "host_read": raw.get("host_read") is True,
            "confidence": max(0.0, min(1.0, confidence)),
            "brand": str(raw.get("brand") or "")[:80],
            "reason": str(raw.get("reason") or "")[:160],
        }
    return results


def _retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "quota",
            "temporarily",
            "timeout",
            "timed out",
            "500",
            "502",
            "503",
            "internal server error",
        )
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _describe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc).replace("\n", " ")[:400]
    return f"{type(exc).__name__}(status={status}): {text}"


def _daily_quota_exhausted(exc: Exception) -> bool:
    """A 429 caused by a per-day quota will not recover within a retry window.

    Retrying it with exponential backoff only burns hours (each episode took
    ~28 minutes in practice). Rate-per-minute limits still deserve retries.
    """
    text = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if status != 429 and "429" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "perday",
            "per day",
            "daily",
            "quota exceeded",
            "quotaexceeded",
            "resource_exhausted",
        )
    )


def is_key_failure(exc: Exception) -> bool:
    """True when another API key might succeed where this one failed.

    Daily quota, exhausted rate limits and auth/permission rejections are
    properties of the key, not of the request. Anything else (malformed
    prompt, model not found, transient network error) must not silently burn
    through the whole key pool.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if _daily_quota_exhausted(exc):
        return True
    return status in (401, 403, 429)


def _create_with_retry(client, attempts: int | None = None, **kwargs):
    if attempts is None:
        attempts = int(os.environ.get("LLM_RETRY_ATTEMPTS", "9"))
    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if _daily_quota_exhausted(exc):
                _log(f"每日配额耗尽，立即放弃本轮: {_describe_error(exc)}")
                raise
            retryable = _retryable(exc)
            _log(
                f"请求失败（第 {attempt + 1}/{attempts} 次，"
                f"{'可重试' if retryable else '不可重试'}）: {_describe_error(exc)}"
            )
            if attempt == attempts - 1 or not retryable:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(120.0, 10.0 * (2 ** attempt))
            _log(f"等待 {delay:.0f}s 后重试")
            time.sleep(delay)


def classify_items(
    report: dict,
    client,
    model: str,
    batch_size: int = 8,
) -> dict[str, dict[str, Any]]:
    items = build_items(report)
    results: dict[str, dict[str, Any]] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        expected = {item["id"] for item in batch}
        for parse_attempt in range(2):
            response = _create_with_retry(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(batch)},
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            try:
                results.update(parse_response(text, expected))
                break
            except ValueError:
                if parse_attempt == 1:
                    raise
    return results


def apply_classifications(
    report: dict,
    results: dict[str, dict[str, Any]],
    drop_content: bool = True,
    drop_host_read: bool = True,
    keep_repeated_content: bool = False,
) -> dict:
    verified = 0
    non_ad = 0
    host_read = 0
    dropped_repeated_content = 0
    for episode_index, episode in enumerate(report.get("episodes", [])):
        kept = []
        for candidate_index, candidate in enumerate(episode.get("candidates", [])):
            result = results.get(f"{episode_index}:{candidate_index}")
            if result is None:
                kept.append(candidate)
                continue
            evidence = candidate.setdefault("evidence", {})
            evidence["llm"] = result
            is_target = (
                bool(result.get("is_ad"))
                and not result.get("host_read")
                and result.get("ad_type") == "third_party_ad"
            )
            report_labels = report.setdefault("llm_labels", [])
            report_labels.append(
                {
                    "episode": episode.get("title"),
                    "episode_id": episode.get("id"),
                    "episode_duration_s": episode.get("duration_s"),
                    "id": f"{episode_index}:{candidate_index}",
                    "tier": candidate.get("tier"),
                    "start_s": candidate.get("start_s"),
                    "end_s": candidate.get("end_s"),
                    "score_before_llm": candidate.get("score"),
                    "heuristic_score": evidence.get("heuristic_score"),
                    "features": {
                        "musicness": float(evidence.get("musicness") or 0.0),
                        "loudness_dev_db": float(evidence.get("loudness_dev_db") or 0.0),
                        "boundary_strength": float(evidence.get("boundary_strength") or 0.0),
                        "timbre_min": float(evidence.get("timbre_min") or 0.0),
                        "position": float(evidence.get("position") or 0.0),
                        "duration_s": float(candidate.get("end_s", 0.0)) - float(candidate.get("start_s", 0.0)),
                        "slot_matches": int(evidence.get("slot_matches") or 0),
                        "start_spread_s": float(evidence.get("start_spread_s") or 0.0),
                        "fixed_slot": bool(evidence.get("fixed_slot")),
                        "matched_episodes": int(evidence.get("matched_episodes") or 0),
                        "votes": int(evidence.get("votes") or 0),
                        "ad_language": bool(evidence.get("ad_language")),
                    },
                    "asr_text": str(evidence.get("asr_text") or "")[:400],
                    "ad_terms": evidence.get("ad_terms") or [],
                    "evidence": dict(evidence),
                    "llm": result,
                    "is_ad": is_target,
                }
            )
            evidence["llm_verified"] = is_target
            if result.get("host_read"):
                host_read += 1
            if is_target:
                verified += 1
            else:
                non_ad += 1
            score = float(candidate.get("score", 0.0))
            confidence = float(result.get("confidence", 0.0))
            if is_target:
                score = max(score + 0.08, 0.82 + 0.15 * confidence)
            else:
                score -= 0.08
            candidate["score"] = round(min(0.99, max(0.0, score)), 3)
            if drop_host_read and result.get("host_read"):
                continue
            if drop_content and not is_target and candidate.get("tier") == "inserted":
                continue
            if (
                drop_content
                and not is_target
                and candidate.get("tier") == "repeated"
                and not keep_repeated_content
                and confidence >= 0.8
                and result.get("ad_type") in {"content", "self_promo"}
                and not evidence.get("ad_language")
            ):
                dropped_repeated_content += 1
                continue
            kept.append(candidate)
        episode["candidates"] = sorted(
            kept,
            key=lambda item: (-float(item.get("score", 0.0)), float(item.get("start_s", 0.0))),
        )
    report["llm"] = {
        "verified_third_party_ads": verified,
        "non_ads": non_ad,
        "host_read": host_read,
        "dropped_repeated_content": dropped_repeated_content,
    }
    return report
