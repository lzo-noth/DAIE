from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import blind_test_batch
from scripts.apply_review_decisions import derive_ads
from scripts.blind_test import _identity_checks, _independence_check, _scope_ground_truth, _verify_human_labels
from dai_detect.feed_identity import canonical_feed_key
from scripts.merge_release_shards import _merge_jsonl
from scripts.review_ad_boundaries import apply_review, build_worklist
from scripts.verify_release_coverage import verify
from scripts.publish_annotation_data import main as publish_main

SEGMENTS = [
    {"start_s": 10.0, "end_s": 14.0},
    {"start_s": 14.0, "end_s": 18.0},
    {"start_s": 26.0, "end_s": 30.0},
    {"start_s": 30.0, "end_s": 34.0},
]


def _results(**overrides: dict) -> dict[str, dict]:
    base = {
        "0": {"is_ad": True, "ad_type": "third_party_ad", "confidence": 0.9, "host_read": True},
        "1": {"is_ad": True, "ad_type": "third_party_ad", "confidence": 0.9, "host_read": True},
        "2": {"is_ad": True, "ad_type": "third_party_ad", "confidence": 0.9, "host_read": False},
        "3": {"is_ad": True, "ad_type": "third_party_ad", "confidence": 0.9, "host_read": False},
    }
    base.update(overrides)
    return base


class HostReadPolicyTest(unittest.TestCase):
    def test_rebuild_drops_host_read_ads(self) -> None:
        ads = derive_ads(SEGMENTS, _results(), {"ignore_host_read": True}, [])
        self.assertEqual(len(ads), 1)
        self.assertEqual((ads[0]["start_s"], ads[0]["end_s"]), (26.0, 34.0))

    def test_rebuild_keeps_host_read_when_policy_allows(self) -> None:
        ads = derive_ads(SEGMENTS, _results(), {"ignore_host_read": False}, [])
        self.assertEqual(len(ads), 2)

    def test_self_promo_and_brand_mentions_are_not_ads(self) -> None:
        results = _results(
            **{
                "2": {"is_ad": True, "ad_type": "self_promo", "confidence": 0.9},
                "3": {"is_ad": False, "ad_type": "content", "confidence": 0.1},
            }
        )
        ads = derive_ads(SEGMENTS, results, {"ignore_host_read": True}, [])
        self.assertEqual(ads, [])


class ShortAdScopeTest(unittest.TestCase):
    def test_short_ground_truth_ads_are_excluded(self) -> None:
        ads = [
            {"start_s": 10.0, "end_s": 20.0},
            {"start_s": 30.0, "end_s": 32.7},
        ]
        kept, excluded = _scope_ground_truth(ads, 10.0)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(excluded), 1)


class HumanLabelGateTest(unittest.TestCase):
    def test_unverified_human_claim_is_downgraded(self) -> None:
        ads = [{"start_s": 10.0, "end_s": 20.0, "label_method": "", "start_verified": None}]
        source, problems = _verify_human_labels(ads, "human")
        self.assertEqual(source, "unverified")
        self.assertTrue(problems)

    def test_verified_human_claim_is_kept(self) -> None:
        ads = [
            {
                "start_s": 10.0,
                "end_s": 20.0,
                "label_method": "human_audio_review",
                "start_verified": True,
                "end_verified": True,
            }
        ]
        source, problems = _verify_human_labels(ads, "human")
        self.assertEqual(source, "human")
        self.assertEqual(problems, [])


class MergePriorityTest(unittest.TestCase):
    def _write(self, directory: Path, rows: list[dict]) -> Path:
        path = directory / "annotations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        return path

    def test_success_outranks_stale_error_regardless_of_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = {"feed": "A", "episode": "1", "ads": [], "llm_error": "429"}
            success = {"feed": "A", "episode": "1", "ads": [{"start_s": 1.0, "end_s": 12.0}]}
            first = self._write(root / "one", [failed])
            second = self._write(root / "two", [success])
            third = self._write(root / "three", [failed])
            target = root / "out" / "annotations.jsonl"
            _merge_jsonl([first, second, third], target)
            rows = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertNotIn("llm_error", rows[0])
            self.assertEqual(len(rows[0]["ads"]), 1)

    def test_newer_timestamp_wins_between_successes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = {
                "feed": "A",
                "episode": "1",
                "ads": [{"start_s": 1.0, "end_s": 12.0}],
                "generated_at": "2026-01-01T00:00:00",
            }
            newer = {
                "feed": "A",
                "episode": "1",
                "ads": [{"start_s": 2.0, "end_s": 14.0}],
                "generated_at": "2026-02-01T00:00:00",
            }
            first = self._write(root / "one", [older])
            second = self._write(root / "two", [newer])
            target = root / "out" / "annotations.jsonl"
            _merge_jsonl([second, first], target)
            row = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["generated_at"], "2026-02-01T00:00:00")


class ClusterBootstrapTest(unittest.TestCase):
    def test_bootstrap_returns_clustered_intervals(self) -> None:
        reports = [
            {
                "feed": "A",
                "duration_s": 3600.0,
                "metrics": {"matched": 1, "false_positives": 20, "false_negatives": 3},
                "matches": [{"start_error": 1.0, "end_error": 2.0}],
            },
            {
                "feed": "A",
                "duration_s": 3600.0,
                "metrics": {"matched": 1, "false_positives": 25, "false_negatives": 4},
                "matches": [{"start_error": 3.0, "end_error": 4.0}],
            },
            {
                "feed": "B",
                "duration_s": 3600.0,
                "metrics": {"matched": 4, "false_positives": 3, "false_negatives": 0},
                "matches": [{"start_error": 0.5, "end_error": 0.6}],
            },
        ]
        interval = blind_test_batch._cluster_bootstrap(reports, iterations=200, seed=1)
        for key in ("event_precision", "event_recall", "boundary_p95_s"):
            self.assertIn(key, interval)
            self.assertIsNotNone(interval[key]["lower"])
            self.assertIsNotNone(interval[key]["upper"])
            self.assertLessEqual(
                interval[key]["lower"], interval[key]["upper"], key
            )


class CoverageGateTest(unittest.TestCase):
    def test_missing_manifest_row_blocks_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text(
                json.dumps({"feed": "A", "episode": "1"})
                + "\n"
                + json.dumps({"feed": "A", "episode": "2"})
                + "\n",
                encoding="utf-8",
            )
            blind = root / "blind.jsonl"
            blind.write_text(json.dumps({"feed": "B", "episode": "9"}) + "\n", encoding="utf-8")
            data = root / "data"
            data.mkdir()
            (data / "annotations.jsonl").write_text(
                json.dumps({"feed": "A", "episode": "1", "ads": []}) + "\n",
                encoding="utf-8",
            )
            (data / "blind_annotations.jsonl").write_text("", encoding="utf-8")
            (data / "blind_generated.jsonl").write_text("", encoding="utf-8")
            report = verify(train, blind, data, [], allow_unavailable=False)
            self.assertFalse(report["complete"])
            self.assertIn(("A", "2"), report["train_missing"])
            self.assertIn(("B", "9"), report["blind_missing"])

    def test_explicit_unavailable_rows_are_accounted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            train.write_text(
                "\n".join(
                    json.dumps({"feed": "A", "episode": episode}) for episode in ("1", "2")
                )
                + "\n",
                encoding="utf-8",
            )
            blind = root / "blind.jsonl"
            blind.write_text("", encoding="utf-8")
            data = root / "data"
            data.mkdir()
            (data / "annotations.jsonl").write_text(
                json.dumps({"feed": "A", "episode": "1", "ads": []}) + "\n",
                encoding="utf-8",
            )
            (data / "blind_annotations.jsonl").write_text("", encoding="utf-8")
            (data / "blind_generated.jsonl").write_text("", encoding="utf-8")
            shard = root / "shard"
            shard.mkdir()
            (shard / "annotations.coverage.json").write_text(
                json.dumps(
                    {
                        "unavailable": [
                            {"feed": "A", "episode": "2", "reason": "HTTP 404"}
                        ],
                        "failed": [],
                    }
                ),
                encoding="utf-8",
            )
            report = verify(train, blind, data, [shard], allow_unavailable=True)
            self.assertTrue(report["complete"])
            self.assertEqual(report["unavailable_rows"], [("A", "2")])


class PublishHoldoutTest(unittest.TestCase):
    def test_blind_labels_are_not_published_and_old_copies_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            annotation = {
                "feed": "A",
                "episode": "1",
                "audio": "snapshots/a.mp3",
                "ads": [{"start_s": 10.0, "end_s": 25.0}],
                "labels_source": "ai_reviewed",
            }
            (source / "annotations.jsonl").write_text(
                json.dumps(annotation) + "\n", encoding="utf-8"
            )
            (source / "transcripts.jsonl").write_text("{}\n", encoding="utf-8")
            (source / "blind_annotations.jsonl").write_text("{}\n", encoding="utf-8")
            (source / "blind_generated.jsonl").write_text("{}\n", encoding="utf-8")
            (source / "blind_gt").mkdir()
            (source / "blind_gt" / "b.json").write_text("{}", encoding="utf-8")

            dest = root / "dest"
            dest.mkdir()
            (dest / "blind_annotations.jsonl").write_text("{}", encoding="utf-8")
            (dest / "blind_gt").mkdir()
            (dest / "blind_gt" / "old.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(
                sys,
                "argv",
                [
                    "publish_annotation_data.py",
                    "--source",
                    str(source),
                    "--dest",
                    str(dest),
                    "--run-id",
                    "1",
                ],
            ):
                self.assertEqual(publish_main(), 0)

            self.assertTrue((dest / "annotations.jsonl").exists())
            self.assertFalse((dest / "blind_annotations.jsonl").exists())
            self.assertFalse((dest / "blind_generated.jsonl").exists())
            self.assertFalse((dest / "blind_gt").exists())
            index = json.loads((dest / "index.json").read_text(encoding="utf-8"))
            self.assertFalse(index["holdout"]["published"])


class FeedIdentityTest(unittest.TestCase):
    def test_same_url_different_names_share_a_key(self) -> None:
        url = "https://feeds.megaphone.fm/GLT4787413333/"
        self.assertEqual(
            canonical_feed_key("The Rest Is History", url),
            canonical_feed_key("Rest Is History", url),
        )

    def test_feed_key_blocks_renamed_training_feed(self) -> None:
        url = "https://feeds.twit.tv/twit.xml"
        key = canonical_feed_key("This Week in Tech", url)
        model = {
            "training_identity": {
                "audio_ids": [],
                "source_sha256": [],
                "feeds": ["This Week in Tech"],
                "feed_keys": [key],
            }
        }
        independence, problems = _independence_check(
            {"id": "x", "source_sha256": "y"}, model, "TWiT", feed_url=url
        )
        self.assertTrue(independence["feed_seen_in_training"])
        self.assertFalse(independence["independent"])
        self.assertTrue(problems)


class EmptyGroundTruthIdentityTest(unittest.TestCase):
    def test_audio_identity_still_applies_to_no_ad_sets(self) -> None:
        timeline_audio = {"id": "a" * 64, "source_sha256": "s" * 64}
        missing, mismatches, warnings = _identity_checks(
            timeline_audio,
            {"source_sha256": "s" * 64, "audio_id": "b" * 64, "audio_id_kind": "x"},
            True,
        )
        # source 相同，仅 PCM 指纹漂移时只告警。
        self.assertEqual(mismatches, [])
        self.assertTrue(warnings)


class ManualReviewTest(unittest.TestCase):
    def test_worklist_keeps_candidate_boundaries(self) -> None:
        rows = [
            {
                "feed": "A",
                "episode": "1",
                "audio": "snapshots/a.mp3",
                "ads": [{"start_s": 10.0, "end_s": 25.0, "type": "third_party_inserted"}],
                "labels_source": "ai_reviewed",
            }
        ]
        worklist = build_worklist(rows, "ai_reviewed")
        self.assertEqual(len(worklist), 1)
        ad = worklist[0]["ads"][0]
        self.assertEqual(ad["candidate_start_s"], 10.0)
        self.assertIsNone(ad["start_s"])

    def test_apply_requires_listening_and_edge_verification(self) -> None:
        rows = [
            {"feed": "A", "episode": "1", "ads": [], "labels_source": "ai_reviewed"}
        ]
        review = {
            "feed": "A",
            "episode": "1",
            "reviewer": "human",
            "reviewed_at": "2026-11-09T00:00:00",
            "listened": True,
            "ads": [
                {
                    "start_s": 120.0,
                    "end_s": 151.42,
                    "start_verified": True,
                    "end_verified": False,
                }
            ],
        }
        updated, rejected = apply_review(rows, [review])
        self.assertEqual(updated[0]["labels_source"], "ai_reviewed")
        self.assertEqual(len(rejected), 1)

    def test_verified_review_marks_episode_human(self) -> None:
        rows = [
            {"feed": "A", "episode": "1", "ads": [], "labels_source": "ai_reviewed"}
        ]
        review = {
            "feed": "A",
            "episode": "1",
            "reviewer": "human",
            "reviewed_at": "2026-11-09T00:00:00",
            "listened": True,
            "ads": [
                {
                    "start_s": 120.0,
                    "end_s": 151.42,
                    "start_verified": True,
                    "end_verified": True,
                    "candidate_start_s": 121.15,
                    "candidate_end_s": 150.9,
                }
            ],
        }
        updated, rejected = apply_review(rows, [review])
        self.assertEqual(rejected, [])
        self.assertEqual(updated[0]["labels_source"], "human")
        self.assertEqual(updated[0]["ads"][0]["label_method"], "human_audio_review")
        self.assertEqual(updated[0]["ads"][0]["candidate_start_s"], 121.15)


if __name__ == "__main__":
    unittest.main()
