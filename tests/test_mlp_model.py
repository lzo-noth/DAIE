from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dai_detect.ad_classifier import AdModel
from dai_detect.mlp_model import fit_mlp, mlp_probabilities
from dai_detect.single_features import WINDOW_FEATURE_SCHEMA, WINDOW_FEATURES
from dai_detect.window_context import (
    CONTEXT_SCHEMA,
    CONTEXT_SPANS,
    context_feature_names,
    window_context,
)


def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


class WindowContextTest(unittest.TestCase):
    def test_means_stay_inside_the_requested_span(self) -> None:
        times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        features = np.asarray([[0.0], [0.0], [9.0], [0.0], [0.0]], dtype=np.float32)
        out = window_context(features, times, spans=(1.0,))
        self.assertEqual(out.shape, (5, 2))
        self.assertAlmostEqual(float(out[2, 1]), 3.0, places=5)

    def test_names_match_the_matrix_width(self) -> None:
        names = context_feature_names(["a", "b"])
        self.assertEqual(len(names), 2 + 2 * len(CONTEXT_SPANS))
        self.assertIn("a__ctx4s", names)


@unittest.skipUnless(_has_sklearn(), "scikit-learn 未安装")
class MlpRoundTripTest(unittest.TestCase):
    def _data(self, dim: int = 6) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        X = np.vstack(
            [rng.normal(0.0, 1.0, (300, dim)), rng.normal(2.5, 1.0, (300, dim))]
        )
        y = np.concatenate([np.zeros(300), np.ones(300)]).astype(int)
        return X, y

    def test_exported_layers_reproduce_sklearn_probabilities(self) -> None:
        from sklearn.neural_network import MLPClassifier

        X, y = self._data()
        model = fit_mlp(X, y, seed=7)
        mean = np.asarray(model["mean"])
        scale = np.asarray(model["scale"])
        reference = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=7,
        ).fit((X - mean) / scale, y)
        expected = reference.predict_proba((X - mean) / scale)[:, 1]
        actual = mlp_probabilities(model, X)
        self.assertTrue(np.allclose(actual, expected, atol=1e-6))

    def test_model_round_trips_through_json(self) -> None:
        # 训练矩阵宽度必须与模型声明的特征一致：带上下文时 predict 会把
        # 84 维窗口特征扩成 336 维再按名字取列，用小维度训练会让归一化参数
        # 与输入宽度对不上。
        context_names = context_feature_names(list(WINDOW_FEATURES))
        X, y = self._data(len(context_names))
        model = fit_mlp(X, y)
        times = np.arange(0.0, 3.0, 0.5)
        base = np.zeros((len(times), len(WINDOW_FEATURES)), dtype=np.float32)
        payload = {
            "feature_schema": WINDOW_FEATURE_SCHEMA,
            "input_schema": CONTEXT_SCHEMA,
            "context_spans": list(CONTEXT_SPANS),
            "features": context_names,
            "threshold": 0.5,
            "metrics": {},
            **model,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ad_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = AdModel.load(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.kind, "mlp")
            scores = loaded.predict(base, list(WINDOW_FEATURES), times)
            self.assertEqual(scores.shape, (len(times),))

    def test_mlp_without_context_schema_is_rejected(self) -> None:
        X, y = self._data()
        payload = {
            "feature_schema": WINDOW_FEATURE_SCHEMA,
            "features": list(WINDOW_FEATURES),
            "metrics": {},
            **fit_mlp(X, y),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ad_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(AdModel.load(path))


if __name__ == "__main__":
    unittest.main()
