from __future__ import annotations

import unittest

import numpy as np

from dai_detect.boundary_model import BOUNDARY_FEATURES
from dai_detect.single_features import WINDOW_FEATURES
from scripts.blind_test import _identity_checks
from scripts.train_single_episode import (
    EpisodeSample,
    _aggregate_fold_window_metrics,
    _choose_classifier_threshold,
    _choose_event_threshold,
    _event_metrics,
    _group_folds,
    _identity_problems,
    _inner_folds,
    _training_artifacts_complete,
)


def _sample(feed: str, labels: list[int]) -> EpisodeSample:
    values = np.asarray(labels, dtype=np.float64)
    features = np.zeros((len(values), len(WINDOW_FEATURES)), dtype=np.float32)
    features[:, 0] = values * 2.0 - 0.5
    return EpisodeSample(
        feed=feed,
        episode=f"{feed}-{len(values)}",
        audio="",
        windows={
            "times": np.arange(len(values), dtype=np.float64) * 0.5 + 0.5,
            "hop_s": 0.5,
            "span_s": 1.0,
            "features": features,
        },
        indices=np.arange(len(values)),
        labels=values,
        ads=[],
        duration_s=len(values) * 0.5,
        change_scores=np.zeros(len(values)),
        change_grid_s=0.01,
        boundary_features=np.zeros((len(values), len(BOUNDARY_FEATURES)), dtype=np.float16),
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

    def _threshold_setup(self, first_labels: list[int]):
        samples = [
            _sample("A", first_labels),
            _sample("B", [0, 1, 1]),
            _sample("C", [0, 1, 1]),
        ]
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {}
        cursor = 0
        for sample in samples:
            offsets[id(sample)] = (cursor, cursor + len(sample.labels))
            cursor += len(sample.labels)
        return samples, X, y, offsets

    def test_negative_only_inner_test_fold_is_kept(self) -> None:
        samples, X, y, offsets = self._threshold_setup([0, 0, 0])
        _, metrics = _choose_classifier_threshold([0, 1, 2], samples, X, y, offsets, seed=1)
        self.assertEqual(metrics["source"], "inner_grouped_cv")
        self.assertEqual(metrics["folds"], 3)

    def test_positive_only_inner_test_fold_is_kept(self) -> None:
        samples, X, y, offsets = self._threshold_setup([1, 1, 1])
        _, metrics = _choose_classifier_threshold([0, 1, 2], samples, X, y, offsets, seed=1)
        self.assertEqual(metrics["source"], "inner_grouped_cv")
        self.assertEqual(metrics["folds"], 3)

    def test_within_feed_inner_split_is_labelled(self) -> None:
        samples = [_sample("A", [0, 1]), _sample("A", [0, 1])]
        folds, kind = _inner_folds([0, 1], samples)
        self.assertEqual(len(folds), 2)
        self.assertEqual(kind, "within_feed_fallback")
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {id(samples[0]): (0, 2), id(samples[1]): (2, 4)}
        _, metrics = _choose_classifier_threshold([0, 1], samples, X, y, offsets, seed=1)
        self.assertEqual(metrics["source"], "within_feed_fallback")

    def test_window_metrics_use_each_fold_threshold(self) -> None:
        samples = [_sample("A", [1, 0]), _sample("B", [1, 0])]
        y = np.asarray([1, 0, 1, 0], dtype=np.float64)
        oof = np.asarray([0.9, 0.1, 0.9, 0.1])
        offsets = {id(samples[0]): (0, 2), id(samples[1]): (2, 4)}
        metrics = _aggregate_fold_window_metrics(
            y, oof, [[0], [1]], {0: 0.5, 1: 0.95}, samples, offsets
        )
        self.assertEqual(metrics["threshold_source"], "per_outer_fold")
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)

    def test_fold_wise_auc_separates_score_scales(self) -> None:
        # Each fold alone is perfectly ranked, but their scores overlap across
        # folds; macro AUC is 1.0 while pooled AUC is only 0.75.
        samples = [_sample("A", [1, 0]), _sample("B", [1, 0])]
        y = np.asarray([1, 0, 1, 0], dtype=np.float64)
        oof = np.asarray([0.6, 0.5, 0.2, 0.1])
        offsets = {id(samples[0]): (0, 2), id(samples[1]): (2, 4)}
        metrics = _aggregate_fold_window_metrics(
            y, oof, [[0], [1]], {0: 0.5, 1: 0.5}, samples, offsets
        )
        self.assertEqual(metrics["auc_fold"], [1.0, 1.0])
        self.assertEqual(metrics["auc_macro"], 1.0)
        from dai_detect.metrics import roc_auc

        self.assertEqual(roc_auc(y, oof), 0.75)

    def test_outer_held_labels_do_not_change_threshold(self) -> None:
        samples = [_sample("A", [0, 1]), _sample("B", [0, 1]), _sample("C", [0, 1])]
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {id(samples[0]): (0, 2), id(samples[1]): (2, 4), id(samples[2]): (4, 6)}
        first, _ = _choose_event_threshold([0, 1], samples, X, y, offsets, seed=1)
        mutated = y.copy()
        mutated[4:6] = 1.0 - mutated[4:6]
        second, _ = _choose_event_threshold([0, 1], samples, X, mutated, offsets, seed=1)
        self.assertEqual(first, second)

    def test_event_threshold_objective_is_used_with_events(self) -> None:
        samples = [_sample("A", [0, 1, 1, 0]), _sample("B", [0, 1, 1, 0])]
        for sample in samples:
            sample.ads = [{"start_s": 1.0, "end_s": 2.0}]
            sample.duration_s = 2.0
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {id(samples[0]): (0, 4), id(samples[1]): (4, 8)}
        _, metrics = _choose_event_threshold([0, 1], samples, X, y, offsets, seed=1)
        self.assertEqual(metrics["objective"], "event_f1")

    def test_boundary_artifact_is_required_unless_skipped(self) -> None:
        self.assertFalse(_training_artifacts_complete(False, weak_mode=False, skip_boundary=False))
        self.assertTrue(_training_artifacts_complete(False, weak_mode=True, skip_boundary=False))
        self.assertTrue(_training_artifacts_complete(False, weak_mode=False, skip_boundary=True))
        self.assertTrue(_training_artifacts_complete(True, weak_mode=False, skip_boundary=False))


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
