#!/usr/bin/env python3
"""Check which LLM models the configured key can use, without printing the key."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _native_models(key: str) -> tuple[int | None, list[str], str]:
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        url = base + "?" + urllib.parse.urlencode({"key": key})
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        models = [
            item["name"].split("/")[-1]
            for item in payload.get("models", [])
            if "generateContent" in (item.get("supportedGenerationMethods") or [])
        ]
        return resp.status, models, "api_key"
    except urllib.error.HTTPError as exc:
        first_status = exc.code
    except Exception as exc:  # noqa: BLE001
        return None, [], f"{type(exc).__name__}: {exc}"[:200]
    try:
        req = urllib.request.Request(base, headers={"Accept": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        models = [
            item["name"].split("/")[-1]
            for item in payload.get("models", [])
            if "generateContent" in (item.get("supportedGenerationMethods") or [])
        ]
        return resp.status, models, "bearer"
    except Exception as exc:  # noqa: BLE001
        return first_status, [], f"{type(exc).__name__}: {exc}"[:200]


def _probe_openai(key: str) -> dict:
    """Probe the OpenAI key independently of the Gemini key.

    Both keys can be configured at once; Gemini wins in the annotation script,
    so a Gemini daily-quota exhaustion hides whether OpenAI still has capacity.
    """
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=60.0, max_retries=0)
    candidates = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini"]
    results = []
    for model in candidates:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Reply with JSON: {\"ok\":true}."},
                    {"role": "user", "content": "ping"},
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            results.append({"model": model, "ok": True, "text": text[:120]})
            print("openai model ok:", model, text[:80])
        except Exception as exc:  # noqa: BLE001
            detail = f"{type(exc).__name__}: {exc}"[:220]
            results.append({"model": model, "ok": False, "error": detail})
            print("openai model failed:", model, detail[:160])
    return {"key_present": True, "models": results}


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    result: dict = {}
    if openai_key:
        result["openai"] = _probe_openai(openai_key)
    else:
        result["openai"] = {"key_present": False}
        print("OPENAI_API_KEY is not configured")
    if not key:
        print("GEMINI_API_KEY is not configured")
        out = Path("out")
        out.mkdir(exist_ok=True)
        (out / "llm_check.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0 if openai_key else 1

    result["key_present"] = True
    status, models, mode = _native_models(key)
    result.update({"native_status": status, "auth_mode": mode, "models": models[:80]})
    print(f"native models status={status} auth={mode} count={len(models)}")
    for name in models[:40]:
        print("model:", name)

    candidates = [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    available = [name for name in candidates if name in models]
    ordered = available + [name for name in candidates if name not in models]

    results = []
    if models or mode == "api_key":
        from openai import OpenAI

        client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        prompt = [
            {"role": "system", "content": "Classify the text. Reply with JSON: {\"is_ad\":true|false}."},
            {"role": "user", "content": "This episode is brought to you by Acme. Use code SAVE20 at acme.com."},
        ]
        for model in ordered:
            try:
                response = client.chat.completions.create(model=model, messages=prompt, temperature=0)
                text = response.choices[0].message.content or ""
                results.append({"model": model, "ok": True, "text": text[:300]})
                print("model ok:", model, text[:160])
                break
            except Exception as exc:  # noqa: BLE001
                results.append({"model": model, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
                print("model failed:", model, type(exc).__name__, str(exc)[:160])

    result["classifications"] = results
    out = Path("out")
    out.mkdir(exist_ok=True)
    (out / "llm_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
