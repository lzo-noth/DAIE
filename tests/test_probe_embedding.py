from __future__ import annotations

import unittest

import numpy as np

from dai_detect.acoustic_embedding import EMBEDDING_DIM, window_embeddings
from scripts.probe_embedding import (
    _fit_model,
    _pool_at_thresholds,
    _score,
    _train_thresholds,
    _view_rows,
    _window_context,
    _window_folds,
)


class WindowEmbeddingTest(unittest.TestCase):
    def test_patches_inside_the_window_are_averaged(self) -> None:
        centres = np.asarray([0.48, 0.96, 1.44, 1.92])
        embeddings = np.asarray(
            [[1.0] * EMBEDDING_DIM, [2.0] * EMBEDDING_DIM, [4.0] * EMBEDDING_DIM, [8.0] * EMBEDDING_DIM],
            dtype=np.float32,
        )
        out = window_embeddings(np.asarray([1.0, 2.0]), centres, embeddings)
        self.assertEqual(out.shape, (2, EMBEDDING_DIM))
        self.assertAlmostEqual(float(out[0, 0]), 3.0, places=5)
        self.assertAlmostEqual(float(out[1, 0]), 8.0, places=5)

    def test_window_without_patches_uses_the_nearest_patch(self) -> None:
        centres = np.asarray([0.48, 1.92])
        embeddings = np.asarray([[1.0] * EMBEDDING_DIM, [5.0] * EMBEDDING_DIM], dtype=np.float32)
        out = window_embeddings(np.asarray([10.0]), centres, embeddings)
        self.assertAlmostEqual(float(out[0, 0]), 5.0, places=5)

    def test_missing_patches_produce_zeros(self) -> None:
        out = window_embeddings(
            np.asarray([1.0]), np.zeros(0), np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        )
        self.assertEqual(out.shape, (1, EMBEDDING_DIM))
        self.assertFalse(out.any())


class WindowContextTest(unittest.TestCase):
    def test_rolling_mean_uses_only_the_requested_span(self) -> None:
        times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        matrix = np.asarray([[0.0], [0.0], [9.0], [0.0], [0.0]], dtype=np.float32)
        out = _window_context(matrix, times, spans=(1.0,))
        self.assertEqual(out.shape, (5, 2))
        self.assertAlmostEqual(float(out[2, 0]), 9.0, places=5)
        self.assertAlmostEqual(float(out[2, 1]), 3.0, places=5)

    def test_context_columns_are_appended_for_every_span(self) -> None:
        times = np.arange(0.0, 20.0, 0.5)
        matrix = np.ones((len(times), 3), dtype=np.float32)
        out = _window_context(matrix, times, spans=(4.0, 12.0, 30.0))
        self.assertEqual(out.shape, (len(times), 12))
        self.assertTrue(np.allclose(out, 1.0))


class ClassifierChoiceTest(unittest.TestCase):
    def test_hist_and_logistic_both_score_separable_data(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn 未安装；CI 会安装")
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.0, 1.0, (200, 4)), rng.normal(3.0, 1.0, (200, 4))])
        y = np.concatenate([np.zeros(200), np.ones(200)])
        for name in ("logistic", "hist"):
            model = _fit_model(name, X, y)
            scores = _score(model, X)
            self.assertEqual(scores.shape, (400,))
            self.assertGreater(float(scores[200:].mean()), float(scores[:200].mean()))


class ViewScopeTest(unittest.TestCase):
    def _windows(self) -> dict:
        return {"times": np.arange(0.5, 30.0, 0.5), "span_s": 1.0}

    def _ads(self) -> list[dict]:
        return [
            {"start_s": 5.0, "end_s": 15.0, "ad_production_style": "produced"},
            {"start_s": 20.0, "end_s": 25.0, "ad_production_style": "host_read"},
        ]

    def _labelled(self, view: str) -> dict:
        windows = self._windows()
        indices, labels = _view_rows(windows, self._ads(), view)
        return {float(windows["times"][index]): int(label) for index, label in zip(indices, labels)}

    def test_produced_view_counts_host_read_as_content(self) -> None:
        labelled = self._labelled("produced")
        self.assertEqual(labelled[7.0], 1)
        self.assertEqual(labelled[22.0], 0)

    def test_host_read_view_counts_produced_as_content(self) -> None:
        labelled = self._labelled("host_read")
        self.assertEqual(labelled[7.0], 0)
        self.assertEqual(labelled[22.0], 1)

    def test_all_view_keeps_both_styles_positive(self) -> None:
        labelled = self._labelled("all")
        self.assertEqual(labelled[7.0], 1)
        self.assertEqual(labelled[22.0], 1)

    def test_unknown_style_ads_are_masked_in_every_view(self) -> None:
        windows = self._windows()
        ads = self._ads() + [
            {"start_s": 27.0, "end_s": 29.5, "ad_production_style": "unknown"}
        ]
        for view in ("all", "produced", "host_read"):
            indices, _ = _view_rows(windows, ads, view)
            kept = windows["times"][indices]
            self.assertFalse(
                any(abs(float(time) - 28.25) <= 1.5 for time in kept),
                f"{view} 把风格未知的区间当成监督信号",
            )


class WindowFoldsTest(unittest.TestCase):
    def test_episode_folds_expand_to_window_rows(self) -> None:
        """折里装的是集编号时必须展开成窗口行，否则会训在全量、测在前几行。"""
        bounds = [(0, 3), (3, 5), (5, 9)]
        folds = _window_folds([[0, 2], [1]], bounds)
        self.assertEqual(folds, [[0, 1, 2, 5, 6, 7, 8], [3, 4]])
        self.assertEqual(sum(len(fold) for fold in folds), 9)


class PoolAtThresholdsTest(unittest.TestCase):
    def test_thresholds_are_applied_per_fold(self) -> None:
        labels = np.concatenate([np.ones(10), np.zeros(100)])
        scores = np.concatenate(
            [np.arange(100, 110, dtype=np.float64), np.arange(100, dtype=np.float64)]
        )
        fold = list(range(len(labels)))
        report = _pool_at_thresholds(
            scores, labels, [(fold, {"0.99": 99.0})], "0.99", hours=1.0
        )
        self.assertEqual(report["recall"], 1.0)
        self.assertAlmostEqual(report["specificity"], 0.99, places=4)
        self.assertAlmostEqual(report["false_positives_per_hour"], 1.0, places=4)

    def test_fold_without_threshold_is_not_scored(self) -> None:
        labels = np.asarray([1.0, 0.0])
        scores = np.asarray([0.9, 0.1])
        report = _pool_at_thresholds(
            scores, labels, [([0, 1], None)], "0.99", hours=1.0
        )
        self.assertIsNone(report["recall"])
        self.assertIsNone(report["specificity"])


class TrainThresholdTest(unittest.TestCase):
    def _model(self) -> dict:
        return {"mean": [0.0], "scale": [1.0], "weights": [1.0], "intercept": 0.0}

    def test_thresholds_are_quantiles_of_training_negatives(self) -> None:
        values = np.linspace(-5.0, 5.0, 100).reshape(-1, 1)
        labels = np.zeros(100)
        labels[-10:] = 1.0
        train = np.arange(90)
        thresholds = _train_thresholds(self._model(), values, labels, train)
        assert thresholds is not None
        negatives = values[train][labels[train] == 0].ravel()
        expected = 1.0 / (1.0 + np.exp(-np.quantile(negatives, 0.99)))
        self.assertAlmostEqual(thresholds["0.99"], expected, places=4)
        self.assertGreaterEqual(thresholds["0.99"], thresholds["0.90"])

    def test_no_negative_rows_means_no_threshold(self) -> None:
        values = np.ones((4, 1))
        labels = np.ones(4)
        self.assertIsNone(_train_thresholds(self._model(), values, labels, np.arange(4)))


if __name__ == "__main__":
    unittest.main()
