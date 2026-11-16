"""Small MLP window classifier with a NumPy-only runtime.

The feasibility probe showed the linear head was the binding constraint: on the
same 84 hand-crafted features, gradient boosting moved the produced-view AUC
from 0.70 to 0.90. Shipping a tree ensemble would drag a heavy runtime
dependency into the detector, so training fits a small MLP with scikit-learn
(offline only) and exports plain weight matrices; inference is a handful of
NumPy matrix products.
"""

from __future__ import annotations

import numpy as np

MLP_KIND = "mlp"


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: tuple[int, ...] = (64, 32),
    seed: int = 0,
    max_iter: int = 200,
) -> dict:
    """Fit an MLP offline and return a JSON-serialisable payload."""
    try:
        from sklearn.neural_network import MLPClassifier
    except ImportError as error:  # pragma: no cover - CI installs sklearn
        raise RuntimeError(
            "mlp 分类器需要 scikit-learn；请先 pip install scikit-learn"
        ) from error

    matrix = np.asarray(X, dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    estimator = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        random_state=seed,
    )
    estimator.fit((matrix - mean) / scale, np.asarray(y, dtype=int))
    layers = []
    for index, (weights, bias) in enumerate(
        zip(estimator.coefs_, estimator.intercepts_)
    ):
        last = index == len(estimator.coefs_) - 1
        layers.append(
            {
                "weights": np.asarray(weights, dtype=np.float64).tolist(),
                "bias": np.asarray(bias, dtype=np.float64).tolist(),
                "activation": "logistic" if last else "relu",
            }
        )
    return {
        "kind": MLP_KIND,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "layers": layers,
    }


def mlp_probabilities(model: dict, X: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    hidden = (np.asarray(X, dtype=np.float64) - mean) / np.where(
        np.abs(scale) < 1e-9, 1.0, scale
    )
    for layer in model["layers"]:
        hidden = hidden @ np.asarray(layer["weights"], dtype=np.float64)
        hidden = hidden + np.asarray(layer["bias"], dtype=np.float64)
        if layer["activation"] == "relu":
            np.maximum(hidden, 0.0, out=hidden)
        else:
            hidden = 1.0 / (1.0 + np.exp(-np.clip(hidden, -30.0, 30.0)))
    return hidden.ravel()
