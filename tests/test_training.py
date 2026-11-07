from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from dai_detect.audio import file_sha256
from dai_detect.boundary_model import BOUNDARY_FEATURES
from dai_detect.single_features import WINDOW_FEATURES
from scripts.blind_test import _identity_checks
from scripts.train_single_episode import (
    EpisodeSample,
    _aggregate_fold_window_metrics,
    _aggregate_head_groups,
    _annotation_problems,
    _boundary_candidates,
    _calibrate_thresholds,
    _calibration_validity,
    _choose_boundary_thresholds,
    _choose_classifier_threshold,
    _choose_event_threshold,
    _cv_quality,
    _duplicate_identity_problems,
    _event_metrics,
    _group_folds,
    _identity_problems,
    _inner_folds,
    _select_boundary_candidate,
    _training_artifacts_complete,
)


def _boundary_episode(feed: str) -> EpisodeSample:
    duration = 40.0
    times = np.arange(0.5, duration, 0.5)
    labels = ((times >= 11.0) & (times <= 19.0)).astype(np.float64)
    features = np.zeros((len(times), len(WINDOW_FEATURES)), dtype=np.float32)
    features[:, 0] = labels * 2.0 - 1.0
    grid = 0.02
    count = int(duration / grid)
    boundary = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float16)
    positions = np.arange(count) * grid
    boundary[:, 0] = np.where(np.abs(positions - 10.0) <= 0.1, 2.0, 0.0)
    boundary[:, 1] = np.where(np.abs(positions - 20.0) <= 0.1, 2.0, 0.0)
    return EpisodeSample(
        feed=feed,
        episode=feed,
        audio="",
        windows={
            "times": times,
            "hop_s": 0.5,
            "span_s": 1.0,
            "features": features,
        },
        indices=np.arange(len(times)),
        labels=labels,
        ads=[{"start_s": 10.0, "end_s": 20.0}],
        duration_s=duration,
        change_scores=np.zeros(len(times)),
        change_grid_s=grid,
        boundary_features=boundary,
    )


def _negative_episode(feed: str) -> EpisodeSample:
    """Confirmed no-ad episode with consistent window/boundary labels."""
    duration = 40.0
    times = np.arange(0.5, duration, 0.5)
    features = np.zeros((len(times), len(WINDOW_FEATURES)), dtype=np.float32)
    features[:, 0] = -1.0
    grid = 0.02
    count = int(duration / grid)
    return EpisodeSample(
        feed=feed,
        episode=feed,
        audio="",
        windows={
            "times": times,
            "hop_s": 0.5,
            "span_s": 1.0,
            "features": features,
        },
        indices=np.arange(len(times)),
        labels=np.zeros(len(times), dtype=np.float64),
        ads=[],
        duration_s=duration,
        change_scores=np.zeros(len(times)),
        change_grid_s=grid,
        boundary_features=np.zeros(
            (count, len(BOUNDARY_FEATURES)), dtype=np.float16
        ),
    )


def _boundary_rows(sample: EpisodeSample) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Onset/offset matrices with positives at the real boundaries."""
    features = sample.boundary_features.astype(np.float32)
    positions = np.arange(len(features)) * sample.change_grid_s
    negatives = [
        index
        for index in range(0, len(features), 37)
        if min(abs(positions[index] - 10.0), abs(positions[index] - 20.0)) >= 5.0
    ]

    def build(edge: float) -> tuple[np.ndarray, np.ndarray]:
        positives = [index for index in range(len(features)) if abs(positions[index] - edge) <= 0.1]
        X = np.vstack([features[positives], features[negatives]]) if negatives else features[positives]
        y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
        return X.astype(np.float32), y

    onset_X, onset_y = build(10.0)
    offset_X, offset_y = build(20.0)
    return onset_X, onset_y, offset_X, offset_y


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

    def test_boundary_auc_is_aggregated_per_fold(self) -> None:
        groups = [
            (np.asarray([0.6, 0.5]), np.asarray([1, 0]), 0.5, 0),
            (np.asarray([0.2, 0.1]), np.asarray([1, 0]), 0.5, 1),
        ]
        metrics = _aggregate_head_groups(groups)
        self.assertEqual(metrics["auc_fold"], [1.0, 1.0])
        self.assertEqual(metrics["auc_macro"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["auc_pooled"], 0.75)

    def test_auc_is_null_when_no_fold_has_both_classes(self) -> None:
        groups = [
            (np.asarray([0.7, 0.6]), np.asarray([1, 1]), 0.5, 0),
            (np.asarray([0.3, 0.2]), np.asarray([0, 0]), 0.5, 1),
        ]
        metrics = _aggregate_head_groups(groups)
        self.assertIsNone(metrics["auc_macro"])
        self.assertIsNone(metrics["auc"])

    def test_window_auc_is_null_without_mixed_folds(self) -> None:
        samples = [_sample("A", [1, 1]), _sample("B", [0, 0])]
        y = np.asarray([1, 1, 0, 0], dtype=np.float64)
        oof = np.asarray([0.9, 0.8, 0.2, 0.1])
        offsets = {id(samples[0]): (0, 2), id(samples[1]): (2, 4)}
        metrics = _aggregate_fold_window_metrics(
            y, oof, [[0], [1]], {0: 0.5, 1: 0.5}, samples, offsets
        )
        self.assertIsNone(metrics["auc_macro"])

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

    def test_outer_held_event_annotations_do_not_change_threshold(self) -> None:
        samples = [_sample("A", [0, 1, 1, 0]), _sample("B", [0, 1, 1, 0]), _sample("C", [0, 1, 1, 0])]
        for sample in samples:
            sample.ads = [{"start_s": 1.0, "end_s": 2.0}]
            sample.duration_s = 2.0
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {
            id(samples[0]): (0, 4),
            id(samples[1]): (4, 8),
            id(samples[2]): (8, 12),
        }
        first, metrics = _choose_event_threshold([0, 1], samples, X, y, offsets, seed=1)
        self.assertEqual(metrics["objective"], "event_f1")
        mutated = y.copy()
        mutated[8:12] = 1.0 - mutated[8:12]
        samples[2].ads = []
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


class BoundaryThresholdCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [_boundary_episode(feed) for feed in ("A", "B", "C")]
        self.X = np.vstack([sample.windows["features"] for sample in self.samples])
        self.y = np.concatenate([sample.labels for sample in self.samples])
        self.offsets = {
            id(sample): (index * 80, (index + 1) * 80)
            for index, sample in enumerate(self.samples)
        }
        rows = [_boundary_rows(sample) for sample in self.samples]
        self.onset_X = [row[0] for row in rows]
        self.onset_y = [row[1] for row in rows]
        self.offset_X = [row[2] for row in rows]
        self.offset_y = [row[3] for row in rows]

    def test_boundary_thresholds_use_decoded_events_and_report_coverage(self) -> None:
        onset, offset, onset_metrics, offset_metrics = _choose_boundary_thresholds(
            [0, 1, 2],
            self.samples,
            self.onset_X,
            self.onset_y,
            self.offset_X,
            self.offset_y,
            [0, 1, 2],
            self.X,
            self.y,
            self.offsets,
            seed=3,
            provisional_threshold=0.5,
        )
        self.assertGreaterEqual(onset, 0.2)
        self.assertLessEqual(onset, 0.8)
        self.assertGreaterEqual(offset, 0.2)
        self.assertLessEqual(offset, 0.8)
        self.assertEqual(onset_metrics["objective"], "boundary_p95")
        coverage = onset_metrics["coverage"]
        self.assertEqual(coverage["folds_attempted"], 3)
        self.assertEqual(coverage["classifier_folds_completed"], 3)
        self.assertEqual(coverage["onset_folds_completed"], 3)
        self.assertEqual(coverage["offset_folds_completed"], 3)
        self.assertEqual(coverage["pipeline_folds_completed"], 3)
        self.assertEqual(onset_metrics["ground_truth_events"], 3)
        self.assertEqual(onset_metrics["matched_events"], 3)
        self.assertEqual(onset_metrics["event_recall"], 1.0)
        self.assertEqual(offset_metrics["matched_events"], 3)

    def test_missing_boundary_heads_are_reported_not_pretended(self) -> None:
        onset_y = [np.zeros_like(values) for values in self.onset_y]
        onset, _, onset_metrics, _ = _choose_boundary_thresholds(
            [0, 1, 2],
            self.samples,
            self.onset_X,
            onset_y,
            self.offset_X,
            self.offset_y,
            [0, 1, 2],
            self.X,
            self.y,
            self.offsets,
            seed=3,
        )
        coverage = onset_metrics["coverage"]
        self.assertEqual(coverage["classifier_folds_completed"], 3)
        self.assertEqual(coverage["onset_folds_completed"], 0)
        self.assertEqual(coverage["pipeline_folds_completed"], 0)
        self.assertEqual(onset_metrics["matched_events"], 0)
        self.assertEqual(onset_metrics["event_recall"], 0.0)
        self.assertEqual(onset, 0.5)

    def test_p95_does_not_beat_event_coverage(self) -> None:
        results = [
            {"threshold": 0.4, "matched": 2, "p95": 0.12},
            {"threshold": 0.6, "matched": 15, "p95": 0.25},
        ]
        best = _select_boundary_candidate(results, min_recall_ratio=0.8)
        self.assertEqual(best["threshold"], 0.6)

    def test_boundary_search_has_fine_candidates(self) -> None:
        coarse, _ = _boundary_candidates(center=0.5)
        self.assertEqual(coarse[0], 0.2)
        self.assertEqual(coarse[-1], 0.8)
        _, fine = _boundary_candidates(center=0.45)
        self.assertTrue(fine)
        self.assertAlmostEqual(min(fine), 0.35)
        self.assertAlmostEqual(max(fine), 0.55)
        self.assertTrue(all(abs(value - round(value, 2)) < 1e-9 for value in fine))

    def test_calibration_alternates_classifier_and_boundary(self) -> None:
        event, onset, offset, event_metrics, onset_metrics, _ = _calibrate_thresholds(
            [0, 1, 2],
            self.samples,
            self.X,
            self.y,
            self.offsets,
            self.onset_X,
            self.onset_y,
            self.offset_X,
            self.offset_y,
            [0, 1, 2],
            event_seed=1,
            boundary_seed=2,
        )
        calibration = event_metrics["calibration"]
        self.assertLessEqual(calibration["rounds"], 5)
        self.assertGreaterEqual(calibration["rounds"], 1)
        self.assertEqual(calibration["final_classifier_threshold"], round(event, 4))
        self.assertEqual(
            onset_metrics["provisional_classifier_threshold"],
            calibration["history"][-1]["classifier_threshold"],
        )
        self.assertEqual(len(calibration["history"]), calibration["rounds"])
        self.assertIn("converged", calibration)
        self.assertGreaterEqual(onset, 0.2)
        self.assertLessEqual(onset, 0.8)
        self.assertGreaterEqual(offset, 0.2)
        self.assertLessEqual(offset, 0.8)

    def test_event_metrics_rejects_heuristic_fallback_when_boundary_required(self) -> None:
        sample = self.samples[0]
        probabilities = np.full(len(sample.windows["times"]), 0.9)
        with self.assertRaises(ValueError):
            _event_metrics(
                [sample],
                [0],
                {0: probabilities},
                thresholds=0.5,
                refine=True,
                boundary_heads={},
                require_boundary=True,
            )

    def test_negative_control_holdout_still_completes_pipeline(self) -> None:
        samples = [
            _boundary_episode("A"),
            _boundary_episode("B"),
            _negative_episode("C"),
        ]
        X = np.vstack([sample.windows["features"] for sample in samples])
        y = np.concatenate([sample.labels for sample in samples])
        offsets = {
            id(sample): (index * 80, (index + 1) * 80)
            for index, sample in enumerate(samples)
        }
        rows = [_boundary_rows(sample) for sample in samples[:2]]
        onset_X = [row[0] for row in rows]
        onset_y = [row[1] for row in rows]
        offset_X = [row[2] for row in rows]
        offset_y = [row[3] for row in rows]
        _, _, onset_metrics, _ = _choose_boundary_thresholds(
            [0, 1, 2],
            samples,
            onset_X,
            onset_y,
            offset_X,
            offset_y,
            [0, 1],
            X,
            y,
            offsets,
            seed=3,
        )
        coverage = onset_metrics["coverage"]
        self.assertEqual(coverage["pipeline_folds_completed"], 3)
        self.assertEqual(coverage["classifier_folds_completed"], 3)
        self.assertEqual(coverage["onset_folds_completed"], 3)
        self.assertEqual(coverage["offset_folds_completed"], 3)
        self.assertEqual(coverage["positive_boundary_episodes"], 2)
        self.assertEqual(coverage["negative_control_episodes"], 1)
        self.assertEqual(onset_metrics["ground_truth_events"], 2)

    def test_nonconverged_run_recalibrates_with_final_boundaries(self) -> None:
        event_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.65]
        boundary_values = [
            (0.2, 0.6),
            (0.3, 0.7),
            (0.4, 0.8),
            (0.5, 0.2),
            (0.6, 0.3),
        ]
        event_calls: list[dict] = []
        boundary_calls: list[float] = []

        def fake_event(*args, **kwargs):
            event_calls.append(kwargs)
            value = event_values[len(event_calls) - 1]
            metrics = {
                "source": "inner_grouped_cv_event_f1",
                "objective": "event_f1",
                "folds_attempted": 2,
                "pipeline_folds_completed": 2,
                "event": {
                    "event_precision": 1.0,
                    "event_recall": 1.0,
                    "event_f1": 1.0,
                    "boundary_p95_s": 0.1,
                },
            }
            return value, metrics

        def fake_boundary(*args, **kwargs):
            boundary_calls.append(kwargs.get("provisional_threshold"))
            onset, offset = boundary_values[len(boundary_calls) - 1]
            selection = {
                "boundary_calibration_valid": True,
                "provisional_classifier_threshold": kwargs.get("provisional_threshold"),
                "coverage": {"folds_attempted": 2, "pipeline_folds_completed": 2},
            }
            return onset, offset, dict(selection), dict(selection)

        with mock.patch(
            "scripts.train_single_episode._choose_event_threshold",
            side_effect=fake_event,
        ), mock.patch(
            "scripts.train_single_episode._choose_boundary_thresholds",
            side_effect=fake_boundary,
        ), mock.patch(
            "scripts.train_single_episode._final_joint_metrics",
            return_value={
                "event_f1": 1.0,
                "boundary_p95_s": 0.1,
                "folds_attempted": 2,
                "pipeline_folds_completed": 2,
                "false_positives_per_hour": 0.0,
            },
        ) as final_joint:
            event, onset, offset, event_metrics, _, _ = _calibrate_thresholds(
                [0, 1, 2],
                self.samples,
                self.X,
                self.y,
                self.offsets,
                self.onset_X,
                self.onset_y,
                self.offset_X,
                self.offset_y,
                [0, 1, 2],
                event_seed=1,
                boundary_seed=2,
            )

        self.assertEqual(len(boundary_calls), 5)
        self.assertEqual(len(event_calls), 6)
        self.assertFalse(event_metrics["calibration"]["converged"])
        self.assertTrue(event_metrics["calibration"]["final_recalibrated"])
        self.assertEqual(event, event_values[-1])
        self.assertEqual(onset, boundary_values[-1][0])
        self.assertEqual(offset, boundary_values[-1][1])
        self.assertEqual(
            event_calls[-1]["boundary_thresholds"],
            boundary_values[-1],
        )
        final_call = final_joint.call_args.kwargs
        self.assertEqual(final_call["classifier_threshold"], event_values[-1])
        self.assertEqual(final_call["onset_threshold"], boundary_values[-1][0])
        self.assertEqual(final_call["offset_threshold"], boundary_values[-1][1])
        final_metrics = event_metrics["final_calibration_metrics"]
        self.assertEqual(final_metrics["event_f1"], 1.0)
        self.assertEqual(final_metrics["boundary_p95_s"], 0.1)

    def test_calibration_validity_requires_convergence_and_joint_coverage(self) -> None:
        event = {
            "objective": "event_f1",
            "folds_attempted": 3,
            "pipeline_folds_completed": 3,
        }
        final = {
            "folds_attempted": 3,
            "pipeline_folds_completed": 3,
            "false_positives_per_hour": 0.0,
        }
        onset = {"boundary_calibration_valid": True}
        offset = {"boundary_calibration_valid": True}
        good = _calibration_validity(
            event, final, onset, offset, True, True, "strict_nested_grouped_cv"
        )
        self.assertTrue(good["valid"])
        self.assertTrue(good["final_pipeline_valid"])
        self.assertTrue(good["deployment_eligible"])

        not_converged = _calibration_validity(
            event, final, onset, offset, True, False, "strict_nested_grouped_cv"
        )
        self.assertFalse(not_converged["valid"])
        self.assertFalse(not_converged["final_pipeline_valid"])
        self.assertFalse(not_converged["converged"])

        fallback = _calibration_validity(
            {"objective": "window_f1_fallback", "folds_attempted": 3,
             "pipeline_folds_completed": 3},
            final,
            onset,
            offset,
            True,
            True,
            "strict_nested_grouped_cv",
        )
        self.assertFalse(fallback["event_threshold_valid"])

        bad_offset = _calibration_validity(
            event,
            final,
            onset,
            {"boundary_calibration_valid": False},
            True,
            True,
            "strict_nested_grouped_cv",
        )
        self.assertFalse(bad_offset["valid"])
        self.assertFalse(bad_offset["offset_threshold_valid"])

        no_boundary = _calibration_validity(
            event, final, {}, {}, False, True, "strict_nested_grouped_cv"
        )
        self.assertTrue(no_boundary["valid"])
        self.assertIsNone(no_boundary["onset_threshold_valid"])

        limited_cv = _calibration_validity(
            event, final, onset, offset, True, True, "limited_nested_cv"
        )
        self.assertFalse(limited_cv["valid"])
        self.assertFalse(limited_cv["deployment_eligible"])

        too_many_fp = _calibration_validity(
            event,
            {**final, "false_positives_per_hour": 30.0},
            onset,
            offset,
            True,
            True,
            "strict_nested_grouped_cv",
            max_false_positives_per_hour=15.0,
        )
        self.assertFalse(too_many_fp["valid"])
        self.assertFalse(too_many_fp["event_false_positive_rate_valid"])

        final_coverage_gap = _calibration_validity(
            event,
            {**final, "pipeline_folds_completed": 2},
            onset,
            offset,
            True,
            True,
            "strict_nested_grouped_cv",
        )
        self.assertFalse(final_coverage_gap["valid"])
        self.assertFalse(final_coverage_gap["final_coverage_valid"])

    def test_no_ad_false_positive_requires_learned_boundary_head(self) -> None:
        sample = _boundary_episode("A")
        sample.ads = []
        probabilities = np.full(len(sample.windows["times"]), 0.9)
        with self.assertRaises(ValueError):
            _event_metrics(
                [sample],
                [0],
                {0: probabilities},
                thresholds=0.5,
                refine=True,
                boundary_heads={},
                require_boundary=True,
            )

    def test_no_ad_episode_without_events_skips_refinement(self) -> None:
        sample = _boundary_episode("A")
        sample.ads = []
        probabilities = np.full(len(sample.windows["times"]), 0.05)
        metrics = _event_metrics(
            [sample],
            [0],
            {0: probabilities},
            thresholds=0.5,
            refine=True,
            boundary_heads={},
            require_boundary=True,
        )
        self.assertIsNone(metrics["event_precision"])
        self.assertIsNone(metrics["event_recall"])
        self.assertEqual(metrics["false_positives"], 0)

    def test_absolute_boundary_recall_is_enforced(self) -> None:
        results = [
            {"threshold": 0.3, "matched": 10, "ground_truth_events": 100, "p95": 0.05},
            {"threshold": 0.5, "matched": 8, "ground_truth_events": 100, "p95": 0.02},
        ]
        best = _select_boundary_candidate(
            results, min_recall_ratio=0.8, min_event_recall=0.8
        )
        self.assertFalse(best["recall_valid"])
        self.assertEqual(best["threshold"], 0.3)

        valid_results = [
            {"threshold": 0.3, "matched": 9, "ground_truth_events": 10, "p95": 0.2},
            {"threshold": 0.5, "matched": 8, "ground_truth_events": 10, "p95": 0.1},
        ]
        valid = _select_boundary_candidate(
            valid_results, min_recall_ratio=0.8, min_event_recall=0.8
        )
        self.assertTrue(valid["recall_valid"])
        self.assertEqual(valid["threshold"], 0.5)

    def test_false_positive_limit_is_a_hard_constraint(self) -> None:
        results = [
            {
                "threshold": 0.4,
                "matched": 5,
                "ground_truth_events": 5,
                "p95": 0.10,
                "false_positives": 8,
                "false_positives_per_hour": 80.0,
            },
            {
                "threshold": 0.6,
                "matched": 5,
                "ground_truth_events": 5,
                "p95": 0.15,
                "false_positives": 1,
                "false_positives_per_hour": 10.0,
            },
        ]
        best = _select_boundary_candidate(
            results,
            min_recall_ratio=0.8,
            min_event_recall=0.8,
            max_false_positives_per_hour=20.0,
        )
        self.assertEqual(best["threshold"], 0.6)
        self.assertTrue(best["false_positive_rate_valid"])

        all_over_limit = [
            {**results[0]},
            {
                **results[1],
                "threshold": 0.7,
                "false_positives": 4,
                "false_positives_per_hour": 40.0,
            },
        ]
        fallback = _select_boundary_candidate(
            all_over_limit,
            min_recall_ratio=0.8,
            min_event_recall=0.8,
            max_false_positives_per_hour=20.0,
        )
        self.assertFalse(fallback["false_positive_rate_valid"])
        self.assertEqual(fallback["threshold"], 0.7)

    def test_cv_quality_requires_complete_pipeline(self) -> None:
        report = {
            "fold": 0,
            "threshold_metrics": {
                "source": "inner_grouped_cv_event_f1",
                "folds_attempted": 2,
                "pipeline_folds_completed": 2,
            },
            "boundary_threshold_selection": {
                "onset": {
                    "coverage": {"folds_attempted": 2, "pipeline_folds_completed": 2},
                    "boundary_calibration_valid": True,
                },
                "offset": {
                    "coverage": {"folds_attempted": 2, "pipeline_folds_completed": 2},
                    "boundary_calibration_valid": True,
                },
            },
        }
        strict = _cv_quality(["A", "B", "C"], [report], boundary_expected=True)
        self.assertEqual(strict["level"], "strict_nested_grouped_cv")
        report["boundary_threshold_selection"]["onset"]["coverage"][
            "pipeline_folds_completed"
        ] = 1
        limited = _cv_quality(["A", "B", "C"], [report], boundary_expected=True)
        self.assertEqual(limited["level"], "limited_nested_cv")
        self.assertFalse(limited["folds"][0]["boundary_pipeline_complete"])

        report["boundary_threshold_selection"]["onset"]["coverage"][
            "pipeline_folds_completed"
        ] = 2
        report["boundary_threshold_selection"]["onset"][
            "boundary_calibration_valid"
        ] = False
        invalid_recall = _cv_quality(["A", "B", "C"], [report], boundary_expected=True)
        self.assertEqual(invalid_recall["level"], "limited_nested_cv")

        two_feeds = _cv_quality(["A", "B"], [report], boundary_expected=True)
        self.assertEqual(two_feeds["level"], "limited_nested_cv")


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

    def test_event_matching_is_scoped_to_each_episode(self) -> None:
        predicted_only = _boundary_episode("A")
        predicted_only.ads = []
        truth_only = _boundary_episode("B")
        times = predicted_only.windows["times"]
        high = np.where((times >= 10.5) & (times <= 19.5), 0.95, 0.05)
        low = np.full(len(times), 0.05)
        metrics = _event_metrics(
            [predicted_only, truth_only],
            [0, 1],
            {0: high, 1: low},
            thresholds=0.5,
            refine=False,
        )
        self.assertEqual(metrics["predicted_events"], 1)
        self.assertEqual(metrics["ground_truth_events"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["event_precision"], 0.0)
        self.assertEqual(metrics["event_recall"], 0.0)
        self.assertEqual(metrics["event_f1"], 0.0)


class AudioIdentityTest(unittest.TestCase):
    def test_training_requires_identity_by_default(self) -> None:
        problems, _ = _identity_problems(
            {}, "episode.wav", {"audio_id": "a" * 64}, allow_unverified=False
        )
        self.assertTrue(problems)

    def test_training_allows_unverified_only_explicitly(self) -> None:
        problems, _ = _identity_problems(
            {}, "episode.wav", {"audio_id": "a" * 64}, allow_unverified=True
        )
        self.assertEqual(problems, [])

    def test_training_rejects_mismatched_pcm_id(self) -> None:
        entry = {"audio_id": "a" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        mono = {"audio_id": "b" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        problems, _ = _identity_problems(entry, "episode.wav", mono, allow_unverified=False)
        self.assertTrue(any("audio_id" in problem for problem in problems))

    def test_training_downgrades_pcm_id_drift_when_bytes_match(self) -> None:
        entry = {
            "audio_id": "a" * 64,
            "audio_id_kind": "downmix_48k_to_16k",
            "source_sha256": file_sha256(__file__),
        }
        mono = {"audio_id": "b" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        problems, warnings = _identity_problems(
            entry, __file__, mono, allow_unverified=False
        )
        self.assertEqual(problems, [])
        self.assertTrue(any("audio_id" in warning for warning in warnings))

    def test_training_strict_mode_keeps_pcm_drift_fatal(self) -> None:
        entry = {
            "audio_id": "a" * 64,
            "audio_id_kind": "downmix_48k_to_16k",
            "source_sha256": file_sha256(__file__),
        }
        mono = {"audio_id": "b" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        problems, _ = _identity_problems(
            entry, __file__, mono, allow_unverified=False, strict_audio_id=True
        )
        self.assertTrue(any("audio_id" in problem for problem in problems))

    def test_training_rejects_local_audio_without_source_hash(self) -> None:
        entry = {"audio_id": "a" * 64, "audio_id_kind": "k"}
        mono = {"audio_id": "a" * 64, "audio_id_kind": "k"}
        problems, _ = _identity_problems(entry, "episode.wav", mono, allow_unverified=False)
        self.assertTrue(any("source_sha256" in problem for problem in problems))

    def test_training_rejects_missing_audio_kind(self) -> None:
        entry = {"audio_id": "a" * 64, "source_sha256": "s" * 64}
        mono = {"audio_id": "a" * 64, "audio_id_kind": "downmix_48k_to_16k"}
        problems, _ = _identity_problems(entry, "episode.wav", mono, allow_unverified=False)
        self.assertTrue(any("audio_id_kind" in problem for problem in problems))

    def test_annotation_structure_validation(self) -> None:
        problems = _annotation_problems(
            [
                {"start_s": 10.0, "end_s": 12.0},
                {"start_s": 11.0, "end_s": 15.0},
                {"start_s": 50.0, "end_s": 60.0},
            ],
            40.0,
        )
        self.assertTrue(any("重叠" in problem for problem in problems))
        self.assertTrue(any("超出" in problem for problem in problems))
        self.assertEqual(
            _annotation_problems([{"start_s": 10.0, "end_s": 12.0}], 60.0),
            [],
        )
        self.assertTrue(
            _annotation_problems(
                [{"start_s": 10.0, "end_s": 12.0}],
                60.0,
                require_ad_type=True,
            )
        )

    def test_duplicate_identity_across_feeds_is_rejected(self) -> None:
        entries = [
            {"feed": "A", "episode": "1", "audio": "a.wav", "audio_id": "x" * 64},
            {"feed": "B", "episode": "2", "audio": "b.wav", "audio_id": "x" * 64},
        ]
        problems = _duplicate_identity_problems(entries)
        self.assertTrue(any("audio_id" in problem for problem in problems))
        self.assertEqual(
            _duplicate_identity_problems([entries[0]]),
            [],
        )

    def test_ground_truth_requires_identity(self) -> None:
        missing, mismatches, warnings = _identity_checks({}, {}, input_is_local=True)
        self.assertTrue(missing)
        self.assertEqual(mismatches, [])
        self.assertEqual(warnings, [])

    def test_ground_truth_local_requires_byte_hash_and_pcm_id(self) -> None:
        missing, _, _ = _identity_checks(
            {"source_sha256": "s" * 64, "id": "a" * 64, "id_kind": "k"},
            {"source_sha256": "s" * 64},
            input_is_local=True,
        )
        self.assertTrue(any("audio_id" in problem for problem in missing))
        self.assertTrue(any("audio_id_kind" in problem for problem in missing))

    def test_ground_truth_mismatch_is_reported(self) -> None:
        missing, mismatches, _ = _identity_checks(
            {"source_sha256": "s" * 64, "id": "a" * 64, "id_kind": "k"},
            {"source_sha256": "x" * 64, "audio_id": "b" * 64, "audio_id_kind": "k"},
            input_is_local=True,
        )
        self.assertEqual(missing, [])
        self.assertEqual(len(mismatches), 2)

    def test_ground_truth_pcm_drift_with_same_bytes_is_a_warning(self) -> None:
        missing, mismatches, warnings = _identity_checks(
            {"source_sha256": "s" * 64, "id": "a" * 64, "id_kind": "k"},
            {"source_sha256": "s" * 64, "audio_id": "b" * 64, "audio_id_kind": "k"},
            input_is_local=True,
        )
        self.assertEqual(missing, [])
        self.assertEqual(mismatches, [])
        self.assertTrue(any("audio_id" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
