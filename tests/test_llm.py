from __future__ import annotations

import json
import unittest

from dai_detect.llm import apply_classifications, build_items, parse_response


def _candidate(tier: str, text: str, score: float = 0.6, slot: int = 0) -> dict:
    return {
        "start_s": 10.0,
        "end_s": 20.0,
        "score": score,
        "tier": tier,
        "evidence": {"asr_text": text, "slot_matches": slot},
        "matches": [],
    }


class LlmTest(unittest.TestCase):
    def test_parse_response_accepts_fences_and_skips_unknown_ids(self) -> None:
        text = """```json
        {"results":[
          {"id":"0:0","is_ad":true,"ad_type":"third_party_ad","host_read":false,"confidence":1.4},
          {"id":"9:9","is_ad":true}
        ]}
        ```"""
        parsed = parse_response(text, {"0:0"})
        self.assertEqual(set(parsed), {"0:0"})
        self.assertTrue(parsed["0:0"]["is_ad"])
        self.assertEqual(parsed["0:0"]["confidence"], 1.0)

    def test_apply_keeps_ads_drops_content_and_host_reads(self) -> None:
        report = {
            "episodes": [
                {
                    "candidates": [
                        _candidate("inserted", "This episode is brought to you by Acme.", 0.6),
                        _candidate("inserted", "They sailed on the ship in 1857.", 0.6),
                        _candidate("inserted", "Host read sponsor message.", 0.6),
                        _candidate("inserted", "Pre-order my new book.", 0.6),
                        _candidate("inserted", "Unclear but slot matched.", 0.6, slot=1),
                    ]
                }
            ]
        }
        results = {
            "0:0": {"is_ad": True, "ad_type": "third_party_ad", "host_read": False, "confidence": 0.95, "brand": "Acme", "reason": "sponsor"},
            "0:1": {"is_ad": False, "ad_type": "content", "host_read": False, "confidence": 0.9, "brand": "", "reason": "story"},
            "0:2": {"is_ad": True, "ad_type": "third_party_ad", "host_read": True, "confidence": 0.9, "brand": "Acme", "reason": "host read"},
            "0:3": {"is_ad": False, "ad_type": "self_promo", "host_read": False, "confidence": 0.9, "brand": "", "reason": "own book"},
            "0:4": {"is_ad": False, "ad_type": "content", "host_read": False, "confidence": 0.5, "brand": "", "reason": "unclear"},
        }
        apply_classifications(report, results)
        candidates = report["episodes"][0]["candidates"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["evidence"]["llm"]["brand"], "Acme")
        self.assertTrue(candidates[0]["evidence"]["llm_verified"])
        self.assertGreater(candidates[0]["score"], 0.6)
        self.assertEqual(report["llm"]["verified_third_party_ads"], 1)
        self.assertEqual(report["llm"]["host_read"], 1)

    def test_build_items_only_uses_asr_text(self) -> None:
        report = {"episodes": [{"candidates": [_candidate("inserted", "hello"), {"evidence": {}}]}]}
        items = build_items(report)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "0:0")

    def test_keep_weak_flag_keeps_unverified_inserted(self) -> None:
        report = {"episodes": [{"candidates": [_candidate("inserted", "story")]}]}
        results = {"0:0": {"is_ad": False, "ad_type": "content", "host_read": False, "confidence": 0.9, "brand": "", "reason": "story"}}
        apply_classifications(report, results, drop_content=False)
        self.assertEqual(len(report["episodes"][0]["candidates"]), 1)
        self.assertFalse(report["episodes"][0]["candidates"][0]["evidence"]["llm_verified"])


if __name__ == "__main__":
    unittest.main()
