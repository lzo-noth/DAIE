from __future__ import annotations

import unittest

import numpy as np

from dai_detect.boundary_model import BOUNDARY_FEATURES
from dai_detect.single_features import WINDOW_FEATURES
from scripts.blind_test import _identity_checks
from scripts.train_single_episode import (
    EpisodeSample,
    _event_metrics,
    _group_folds,
    _identity_problems,
)


class GroupedCrossValidationTest(unittest.TestCase):
    def test_folds_hold_out_one_feed(self) -> None:
        groups = ["A", "A", "B", "B", "C", "C"]
        folds = _group_folds(groups)
        self.assertEqual(len(folds), 3)
        held_feeds = []
        for held in folds:
            feeds = {groups[index] for index in held}
            self.assertEqual(len(feeds), 1)
            held_feeds.append(next(iter(feeds)))
        self.assertEqual(sorted(held_feeds), ["A", "B", "C"])
        self.assertEqual(sum(len(held) for held in folds), len(groups))

    def test_single_feed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _group_folds(["A", "A", "A"])


class EventEvaluationAlignmentTest(unittest.TestCase):
    def _sample(self) -> EpisodeSample:
        count = 20
        return EpisodeSample(
            feed="f",
            episode="e",
            audio="e.wav",
            windows={
                "times": np.arange(count, dtype=np.float64) * 0.5 + 0.5,
                "hop_s": 0.5,
                "span_s": 1.0,
                "features": np.zeros((count, len(WINDOW_FEATURES)), dtype=np.float32),
            },
            indices=np.arange(count),
            labels=np.ones(count),
            ads=[{"start_s": 2.0, "end_s": 7.0}],
            duration_s=10.0,
            change_scores=np.zeros(count),
            change_grid_s=0.01,
            boundary_features=np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float16),
        )

    def test_rejects_probabilities_shorter_than_full_timeline(self) -> None:
        with self.assertRaises(ValueError):
            _event_metrics([self._sample()], [0], {0: np.zeros(5)}, thresholds=0.5)

    def test_accepts_full_length_probabilities(self) -> None:
        metrics = _event_metrics(
            [self._sample()], [0], {0: np.zeros(20)}, thresholds=0.5, refine=False
        )
        self.assertIn("event_precision", metrics)

    def test_empty_ground_truth_feed_still_reports_false_positives(self) -> None:
        sample = self._sample()
        sample.ads = []
        probabilities = np.full(20, 0.95)
        metrics = _event_metrics(
            [sample], [0], {0: probabilities}, thresholds=0.5, refine=False
        )
        self.assertEqual(metrics["event_precision"], 0.0)
        self.assertIsNone(metrics["event_recall"])
        self.assertGreaterEqual(metrics["false_positives"], 1)
        self.assertIsNotNone(metrics["false_positives_per_hour"])


class AudioIdentityTest(unittest.TestCase):
    def test_training_requires_identity_by_default(self) -> None:
        problems = _identity_problems({}, "episode.wav", {"audio_id": "a" * 64}, allow_unverified=False)
        self.assertTrue(problems)

    def test_training_allows_unverified_only_explicitly(self) -> None:
        problems = _identity_problems({}, "episode.wav", {"audio_id": "a" * 64}, allow_unverified=True)
        self.assertEqual(problems, [])

    def test_training_rejects_mismatched_pcm_id(self) -> None:
        entry = {"audio_id": "a" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        mono = {"audio_id": "b" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        problems = _identity_problems(entry, "episode.wav", mono, allow_unverified=False)
        self.assertTrue(any("audio_id" in problem for problem in problems))

    def test_ground_truth_requires_identity(self) -> None:
        missing, mismatches = _identity_checks({}, {}, input_is_local=True)
        self.assertTrue(missing)
        self.assertEqual(mismatches, [])

    def test_ground_truth_local_requires_byte_hash_and_pcm_id(self) -> None:
        missing, _ = _identity_checks(
            {"source_sha256": "s" * 64, "id": "a" * 64, "id_kind": "k"},
            {"source_sha256": "s" * 64},
            input_is_local=True,
        )
        self.assertTrue(any("audio_id" in problem for problem in missing))
        self.assertTrue(any("audio_id_kind" in problem for problem in missing))

    def test_ground_truth_mismatch_is_reported(self) -> None:
        missing, mismatches = _identity_checks(
            {"source_sha256": "s" * 64, "id": "a" * 64, "id_kind": "k"},
            {"source_sha256": "x" * 64, "audio_id": "b" * 64, "audio_id_kind": "k"},
            input_is_local=True,
        )
        self.assertEqual(missing, [])
        self.assertEqual(len(mismatches), 2)


if __name__ == "__main__":
    unittest.main()
