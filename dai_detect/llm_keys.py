"""Rotate across several API keys behind one OpenAI-compatible client.

A single Gemini key can exhaust its daily quota in the middle of a batch run.
Rather than failing the shard, the client transparently advances to the next
key and retries the same request. Keys are never logged or echoed: only the
1-based index appears in progress output.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Callable

from dai_detect.llm import _log, is_key_failure

KEY_SEPARATORS = (",", "\n", ";", " ")


class AllKeysExhausted(RuntimeError):
    """Every key in the pool failed; the caller should stop the run."""


def load_keys(env_name: str, fallback: str = "") -> list[str]:
    """Read a key pool from ``<env_name>`` plus an optional single-key env var.

    Accepts comma, semicolon, whitespace or newline separated values so the
    same secret works whether it was pasted as a list or one key per line.
    """
    raw = os.environ.get(env_name, "")
    for separator in KEY_SEPARATORS[1:]:
        raw = raw.replace(separator, KEY_SEPARATORS[0])
    keys = [item.strip() for item in raw.split(KEY_SEPARATORS[0])]
    if fallback:
        keys.append(os.environ.get(fallback, ""))
    unique: list[str] = []
    for key in keys:
        if key and key not in unique:
            unique.append(key)
    return unique


class RotatingChatClient:
    """Drop-in for ``OpenAI`` that rotates keys on key-level failures.

    ``start`` rotates the starting point so concurrent shards do not all hit
    key #1 at the same moment (they would exhaust its per-minute limit and
    then march through the pool in lockstep).
    """

    def __init__(
        self,
        factory: Callable[[str], object],
        keys: list[str],
        start: int = 0,
    ) -> None:
        self._factory = factory
        keys = list(keys)
        if keys and start:
            offset = start % len(keys)
            keys = keys[offset:] + keys[:offset]
        self._keys = keys
        self._index = 0
        self._clients: dict[int, object] = {}
        self._exhausted: dict[int, str] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def active_key_number(self) -> int:
        return min(self._index + 1, len(self._keys))

    def _client(self) -> object:
        if self._index not in self._clients:
            self._clients[self._index] = self._factory(self._keys[self._index])
        return self._clients[self._index]

    def _create(self, **kwargs):
        if self._index >= len(self._keys):
            raise AllKeysExhausted(
                f"全部 {len(self._keys)} 个 key 已耗尽或被拒绝，停止调用"
            )
        while self._index < len(self._keys):
            attempt = self._index
            try:
                response = self._client().chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if not is_key_failure(exc):
                    raise
                self._exhausted[attempt] = type(exc).__name__
                self._index += 1
                if self._index >= len(self._keys):
                    raise AllKeysExhausted(
                        f"全部 {len(self._keys)} 个 key 均已不可用"
                        f"（最后一个错误 {type(exc).__name__}）"
                    ) from exc
                _log(
                    f"key #{attempt + 1} 不可用（{type(exc).__name__}），"
                    f"切换到 key #{self._index + 1}/{len(self._keys)}"
                )
                continue
            return response
        raise AllKeysExhausted("key 池已空")

    def status(self) -> dict:
        return {
            "keys": len(self._keys),
            "active": self.active_key_number,
            "exhausted": len(self._exhausted),
            "complete": len(self._exhausted) >= len(self._keys),
        }
