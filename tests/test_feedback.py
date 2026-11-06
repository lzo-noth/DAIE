from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.collect_labels import collect
from scripts.fit_calibration import _load_rows
from scripts.mine_keywords import mine
from scripts.update_loop_state import main as update_state


def _report(ad_is_ad: bool = True) -> dict:
    return {
        "feed": {"title": "TestCast"},
        "episodes": [
            {
                "id": "ep1",
                "title": "Episode One",
                "duration_s": 1800.0,
                "candidates": [
                    {"evidence": {"asr_text": "This episode is sponsored by Coca-Cola."}},
                    {"evidence": {"asr_text": "Ordinary interview chatter."}},
                ],
            }
        ],
        "llm_labels": [
            {
                "episode": "Episode One",
                "id": "0:0",
                "tier": "inserted",
                "start_s": 10.0,
                "end_s": 30.0,
                "heuristic_score": 0.8,
                "features": {
                    "musicness": 0.6,
                    "loudness_dev_db": 8.0,
                    "boundary_strength": 0.7,
                    "timbre_min": 0.2,
                    "position": 1.0,
                    "duration_s": 20.0,
                    "slot_matches": 0,
                    "fixed_slot": False,
                    "matched_episodes": 0,
                    "votes": 0,
                },
                "llm": {"is_ad": ad_is_ad, "ad_type": "third_party_ad" if ad_is_ad else "content", "brand": "Coca-Cola"},
                "is_ad": ad_is_ad,
            },
            {
                "episode": "Episode One",
                "id": "0:1",
                "tier": "inserted",
                "start_s": 100.0,
                "end_s": 120.0,
                "heuristic_score": 0.5,
                "features": {
                    "musicness": 0.1,
                    "loudness_dev_db": 2.0,
                    "boundary_strength": 0.6,
                    "timbre_min": 0.1,
                    "position": 0.35,
                    "duration_s": 20.0,
                    "slot_matches": 0,
                },
                "llm": {"is_ad": False, "ad_type": "content", "brand": ""},
                "is_ad": False,
            },
        ],
    }


class FeedbackLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-loop-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_collect_labels_dedupes_and_keeps_asr(self) -> None:
        report_dir = self.directory / "artifacts" / "dai-test"
        report_dir.mkdir(parents=True)
        (report_dir / "report.json").write_text(json.dumps(_report()), encoding="utf-8")
        store = self.directory / "feedback" / "labels.jsonl"
        first = collect(self.directory / "artifacts", store, "run-1")
        self.assertEqual(first["store_rows"], 2)
        self.assertEqual(first["ads"], 1)
        rows = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
        ad = next(row for row in rows if row["is_ad"])
        self.assertIn("Coca-Cola", ad["asr_text"])
        self.assertEqual(ad["episode_duration_s"], 1800.0)
        second = collect(self.directory / "artifacts", store, "run-2")
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["store_rows"], first["store_rows"])

    def test_mine_keywords_extracts_brand_and_sponsor_phrases(self) -> None:
        rows = [
            {
                "is_ad": True,
                "asr_text": "This episode is sponsored by Coca-Cola. Use code SAVE now.",
                "llm": {"brand": "Coca-Cola"},
                "ad_terms": [],
            },
            {
                "is_ad": True,
                "asr_text": "Support comes from Betterment and Disney Plus.",
                "llm": {"brand": "Betterment"},
                "ad_terms": [],
            },
            {"is_ad": False, "asr_text": "We talked about the ship and the ocean.", "llm": {}, "ad_terms": []},
        ]
        payload = mine(rows, {})
        terms = set(payload["terms"])
        self.assertIn("coca-cola", terms)
        self.assertIn("betterment", terms)
        self.assertIn("sponsored by", terms)
        self.assertIn("support comes from", terms)
        self.assertNotIn("plus", terms)

    def test_fit_dedupes_jsonl_rows(self) -> None:
        store = self.directory / "labels.jsonl"
        row = {
            "feed": "F",
            "episode": "E",
            "tier": "inserted",
            "start_s": 1.0,
            "end_s": 20.0,
            "episode_duration_s": 100.0,
            "features": {"musicness": 0.5, "duration_s": 19.0},
            "is_ad": True,
            "heuristic_score": 0.7,
        }
        store.write_text(
            json.dumps(row) + "\n" + json.dumps({**row, "is_ad": False}) + "\n",
            encoding="utf-8",
        )
        (
            rows,
            labels,
            baselines,
            groups,
            episodes,
            durations,
            texts,
            run_ids,
            timestamps,
        ) = _load_rows([store])
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(labels.sum()), 0)
        self.assertEqual(groups, ["F"])
        self.assertEqual(durations, [100.0])
        self.assertEqual(texts, [""])
        self.assertEqual(run_ids, [""])
        self.assertEqual(timestamps, [""])

    def test_update_loop_state_writes_complete_marker(self) -> None:
        metrics = self.directory / "metrics.json"
        state = self.directory / "state.json"
        metrics.write_text(
            json.dumps(
                {
                    "learned": {"precision": 0.9, "recall": 0.9, "auc": 0.95},
                    "recall_at": {"40": 0.99},
                    "fp_per_hour": 0.5,
                    "candidates_per_hour": 10.0,
                }
            ),
            encoding="utf-8",
        )
        argv = [
            "update_loop_state.py",
            "--metrics",
            str(metrics),
            "--state",
            str(state),
            "--round",
            "1",
            "--max-rounds",
            "5",
        ]
        with mock.patch.object(sys, "argv", argv):
            update_state()
        payload = json.loads(state.read_text(encoding="utf-8"))
        self.assertTrue(payload["complete"])
        self.assertTrue((self.directory / "COMPLETE").exists())


if __name__ == "__main__":
    unittest.main()
