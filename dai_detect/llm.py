from __future__ import annotations

import json
import time
from typing import Any

AD_TYPES = {"third_party_ad", "self_promo", "content", "uncertain"}

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
    return any(marker in text for marker in ("429", "rate limit", "quota", "temporarily"))


def _create_with_retry(client, **kwargs):
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == 3 or not _retryable(exc):
                raise
            time.sleep(2 * (attempt + 1))


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
) -> dict:
    verified = 0
    non_ad = 0
    host_read = 0
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
            evidence["llm_verified"] = is_target
            if result.get("host_read"):
                host_read += 1
            if is_target:
                verified += 1
            else:
                non_ad += 1
            score = float(candidate.get("score", 0.0)) + (0.08 if is_target else -0.08)
            candidate["score"] = round(min(0.99, max(0.0, score)), 3)
            if drop_host_read and result.get("host_read"):
                continue
            if (
                drop_content
                and not is_target
                and candidate.get("tier") == "inserted"
                and not evidence.get("slot_matches")
            ):
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
    }
    return report
