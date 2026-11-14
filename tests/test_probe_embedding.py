from __future__ import annotations

import unittest

import numpy as np

from dai_detect.acoustic_embedding import EMBEDDING_DIM, window_embeddings
from scripts.probe_embedding import _recall_at_specificity, _window_folds


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


class WindowFoldsTest(unittest.TestCase):
    def test_episode_folds_expand_to_window_rows(self) -> None:
        """折里装的是集编号时必须展开成窗口行，否则会训在全量、测在前几行。"""
        bounds = [(0, 3), (3, 5), (5, 9)]
        folds = _window_folds([[0, 2], [1]], bounds)
        self.assertEqual(folds, [[0, 1, 2, 5, 6, 7, 8], [3, 4]])
        self.assertEqual(sum(len(fold) for fold in folds), 9)


class RecallAtSpecificityTest(unittest.TestCase):
    def test_threshold_is_chosen_per_fold_on_negatives(self) -> None:
        labels = np.concatenate([np.ones(10), np.zeros(100)])
        oof = np.concatenate([np.arange(100, 110, dtype=np.float64), np.arange(100, dtype=np.float64)])
        folds = [list(range(len(labels)))]
        report = _recall_at_specificity(oof, labels, folds, 0.99, hours=1.0)
        self.assertEqual(report["recall"], 1.0)
        self.assertAlmostEqual(report["specificity"], 0.99, places=4)
        self.assertAlmostEqual(report["false_positives_per_hour"], 1.0, places=4)

    def test_folds_keep_their_own_score_scale(self) -> None:
        # 两折分数尺度不同；逐折定阈值后仍然不应产生误报。
        fold_a = [0, 1, 2, 3]
        fold_b = [4, 5, 6, 7]
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        oof = np.asarray([0.9, 0.1, 0.8, 0.2, 0.09, 0.01, 0.08, 0.02])
        report = _recall_at_specificity(oof, labels, [fold_a, fold_b], 0.99, hours=1.0)
        self.assertEqual(report["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
