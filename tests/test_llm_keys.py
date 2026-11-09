from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from dai_detect.llm_keys import (
    AllKeysExhausted,
    RotatingChatClient,
    load_keys,
)


class _ApiError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _daily_quota_error() -> _ApiError:
    return _ApiError("429 RESOURCE_EXHAUSTED: quota exceeded per day", 429)


def _minute_rate_error() -> _ApiError:
    return _ApiError(
        "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric "
        "'Generate requests per minute'",
        429,
    )


class _FakeCompletions:
    def __init__(self, key: str, behaviour: dict[str, object]) -> None:
        self.key = key
        self.behaviour = behaviour

    def create(self, **kwargs):
        action = self.behaviour.get(self.key, "ok")
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(choices=action)


def _factory(behaviour: dict[str, object]):
    def build(key: str):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(key, behaviour))
        )

    return build


def _response(text: str) -> list:
    return [SimpleNamespace(message=SimpleNamespace(content=text))]


class ShardOffsetTest(unittest.TestCase):
    def test_start_rotates_the_pool_without_dropping_keys(self) -> None:
        behaviour = {
            "k2": _daily_quota_error(),
            "k3": _daily_quota_error(),
            "k1": _response("ok"),
        }
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2", "k3"], start=1)
        with redirect_stdout(io.StringIO()):
            response = client.chat.completions.create(model="m", messages=[])
        # start=1 -> k2, k3 exhausted first, then the untouched k1.
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(client.status()["keys"], 3)
        self.assertEqual(client.active_key_number, 3)

    def test_offset_wraps_past_pool_length(self) -> None:
        client = RotatingChatClient(_factory({}), ["a", "b", "c"], start=4)
        self.assertEqual(client.active_key_number, 1)


class LoadKeysTest(unittest.TestCase):
    def test_parses_all_separators_and_dedupes(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"POOL": "a,b\nc;d e\na", "SINGLE": "z"},
            clear=False,
        ):
            keys = load_keys("POOL", "SINGLE")
        self.assertEqual(keys, ["a", "b", "c", "d", "e", "z"])

    def test_missing_env_yields_empty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_keys("POOL", "SINGLE"), [])


class RotatingClientTest(unittest.TestCase):
    def test_rotates_to_next_key_after_daily_quota(self) -> None:
        behaviour = {"k1": _daily_quota_error(), "k2": _response("ok")}
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2"])
        out = io.StringIO()
        with redirect_stdout(out):
            response = client.chat.completions.create(model="m", messages=[])
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertEqual(client.active_key_number, 2)
        self.assertNotIn("k1", out.getvalue())
        self.assertNotIn("k2", out.getvalue())
        self.assertIn("#1", out.getvalue())

    def test_all_keys_exhausted_raises(self) -> None:
        behaviour = {"k1": _daily_quota_error(), "k2": _daily_quota_error()}
        client = RotatingChatClient(
            _factory(behaviour), ["k1", "k2"]
        )
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(AllKeysExhausted):
                client.chat.completions.create(model="m", messages=[])
        self.assertTrue(client.status()["complete"])

    def test_non_key_failure_is_not_rotated(self) -> None:
        behaviour = {
            "k1": _ApiError("400 invalid prompt", 400),
            "k2": _response("ok"),
        }
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2"])
        with self.assertRaises(_ApiError):
            client.chat.completions.create(model="m", messages=[])
        self.assertEqual(client.active_key_number, 1)

    def test_per_minute_limit_does_not_burn_a_key(self) -> None:
        behaviour = {"k1": _minute_rate_error(), "k2": _response("ok")}
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2"])
        with self.assertRaises(_ApiError):
            client.chat.completions.create(model="m", messages=[])
        self.assertEqual(client.active_key_number, 1)
        self.assertEqual(client.status()["exhausted"], 0)

    def test_auth_rejection_rotates_to_next_key(self) -> None:
        behaviour = {
            "k1": _ApiError("401 invalid API key", 401),
            "k2": _response("ok"),
        }
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2"])
        with redirect_stdout(io.StringIO()):
            response = client.chat.completions.create(model="m", messages=[])
        self.assertEqual(response.choices[0].message.content, "ok")

    def test_client_reuses_rotated_key_for_later_calls(self) -> None:
        behaviour = {"k1": _daily_quota_error(), "k2": _response("ok")}
        client = RotatingChatClient(_factory(behaviour), ["k1", "k2"])
        with redirect_stdout(io.StringIO()):
            first = client.chat.completions.create(model="m", messages=[])
        second = client.chat.completions.create(model="m", messages=[])
        self.assertEqual(len(first.choices), 1)
        self.assertEqual(len(second.choices), 1)
        self.assertEqual(client.status()["exhausted"], 1)


if __name__ == "__main__":
    unittest.main()
