from __future__ import annotations

import unittest

import numpy as np

from dai_detect.metrics import best_f1_threshold, roc_auc


class RankingMetricsTest(unittest.TestCase):
    def test_perfect_ranking_auc_is_one(self) -> None:
        self.assertAlmostEqual(roc_auc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])), 1.0)

    def test_reversed_ranking_auc_is_zero(self) -> None:
        self.assertAlmostEqual(roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])), 0.0)

    def test_all_tied_scores_auc_is_half(self) -> None:
        self.assertAlmostEqual(roc_auc(np.array([1, 0, 1, 0]), np.zeros(4)), 0.5)

    def test_partial_ties_use_average_ranks(self) -> None:
        # 0.5 forms a tie group at ranks 3-5 (average 4).
        # Positive ranks sum to 2 + 4 + 4 = 10; U = 10 - 6 = 4; AUC = 4/6.
        labels = np.array([1, 0, 1, 1, 0])
        scores = np.array([0.2, 0.1, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(roc_auc(labels, scores), 4.0 / 6.0, places=6)

    def test_threshold_prefers_half_when_f1_ties(self) -> None:
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.1, 0.0])
        threshold, metrics = best_f1_threshold(labels, scores)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertLessEqual(abs(threshold - 0.5), 0.05)


if __name__ == "__main__":
    unittest.main()
