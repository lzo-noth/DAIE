from __future__ import annotations

import unittest

import numpy as np

from dai_detect.acoustic_embedding import EMBEDDING_DIM, window_embeddings
from scripts.probe_embedding import (
    _inner_thresholds,
    _pool_at_thresholds,
    _view_rows,
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


class InnerThresholdTest(unittest.TestCase):
    def _fixture(self):
        # 4 集、两个 feed；每集 20 个窗口行。
        rng = np.random.default_rng(3)
        X = rng.standard_normal((80, 3))
        labels = np.zeros(80)
        labels[[2, 3, 22, 23, 42, 43, 62, 63]] = 1.0
        rows = [np.arange(start, start + 20) for start in range(0, 80, 20)]
        items = [
            {"feed": "Feed A" if index < 2 else "Feed B"} for index in range(4)
        ]
        return X, labels, rows, items

    def test_thresholds_come_from_inner_folds_only(self) -> None:
        X, labels, rows, items = self._fixture()
        thresholds = _inner_thresholds(
            X, labels, rows, items, [0, 1, 2], ratio=4, inner_folds=2
        )
        self.assertIsNotNone(thresholds)
        self.assertGreaterEqual(thresholds["0.99"], thresholds["0.90"])

    def test_single_feed_training_set_has_no_threshold(self) -> None:
        X, labels, rows, items = self._fixture()
        items = [{"feed": "Feed A"} for _ in items]
        self.assertIsNone(
            _inner_thresholds(X, labels, rows, items, [0, 1, 2], ratio=4, inner_folds=2)
        )


if __name__ == "__main__":
    unittest.main()
