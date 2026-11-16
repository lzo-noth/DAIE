"""Window-level ad probability model.

Phase 1 uses a learned logistic regression stored as JSON, because it runs on
plain NumPy and is small enough to ship with the repo. When no model file is
present, a transparent heuristic fallback keeps the pipeline runnable so blind
tests still produce a timeline; its output must not be presented as trained
performance.

The runtime contract is strict: this module only scores features derived from
the single input recording. It never reads labels, transcripts or other
episodes.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .single_features import WINDOW_FEATURE_SCHEMA, WINDOW_FEATURES, select_features
from .window_context import CONTEXT_SCHEMA, context_feature_names, window_context
from .mlp_model import mlp_probabilities

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ad_segment_model.json"


@dataclass
class AdModel:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    threshold: float = 0.5
    kind: str = "logistic"
    layers: list = field(default_factory=list)
    context_spans: list = field(default_factory=list)
    source: str = ""
    model_version: str = ""
    data_version: str = ""
    git_commit: str = ""
    deployment_calibration: dict = field(default_factory=dict)
    training_identity: dict = field(default_factory=dict)
    target_scope: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Optional["AdModel"]:
        target = Path(path or os.environ.get("DAI_AD_MODEL", DEFAULT_MODEL_PATH))
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            features = [str(name) for name in payload["features"]]
            if payload.get("feature_schema") != WINDOW_FEATURE_SCHEMA:
                # 旧模型按名字取列，新增特征不会让它报错，只会让它少看证据
                # 并悄悄退化；这里直接拒绝，避免把旧模型当成当前模型使用。
                print(
                    "[ad] 模型缺少当前特征 schema，已忽略；请重新训练",
                    file=sys.stderr,
                )
                return None
            metrics = payload.get("metrics") or {}
            calibration = metrics.get("deployment_calibration") or {}
            kind = str(payload.get("kind", "logistic"))
            layers = payload.get("layers") or []
            context_spans = [float(value) for value in payload.get("context_spans") or []]
            if kind == "mlp":
                if payload.get("input_schema") != CONTEXT_SCHEMA:
                    print(
                        "[ad] MLP 模型缺少当前上下文 schema，已忽略；请重新训练",
                        file=sys.stderr,
                    )
                    return None
                if not layers:
                    return None
                mean = np.asarray(payload["mean"], dtype=np.float64)
                scale = np.asarray(payload["scale"], dtype=np.float64)
                # 归一化参数必须与模型声明的特征一一对应；否则 predict 里
                # 会在广播阶段报错，错误信息完全看不出是模型产物的问题。
                first = (
                    np.asarray(layers[0].get("weights", []), dtype=np.float64)
                    if isinstance(layers[0], dict)
                    else np.zeros((0, 0))
                )
                if (
                    mean.shape != (len(features),)
                    or scale.shape != (len(features),)
                    or first.ndim != 2
                    or first.shape[0] != len(features)
                ):
                    print(
                        "[ad] MLP 模型输入维度与特征列表不一致，已忽略；请重新训练",
                        file=sys.stderr,
                    )
                    return None
                return cls(
                    features=features,
                    mean=mean,
                    scale=scale,
                    weights=np.zeros(0, dtype=np.float64),
                    intercept=0.0,
                    threshold=float(payload.get("threshold", 0.5)),
                    kind=kind,
                    layers=layers,
                    context_spans=context_spans,
                    source=str(target),
                    model_version=str(payload.get("model_version", "")),
                    data_version=str(payload.get("data_version", "")),
                    git_commit=str(payload.get("git_commit", "")),
                    deployment_calibration=calibration if isinstance(calibration, dict) else {},
                    training_identity=(
                        payload.get("training_identity")
                        if isinstance(payload.get("training_identity"), dict)
                        else {}
                    ),
                    target_scope=(
                        payload.get("target_scope")
                        if isinstance(payload.get("target_scope"), dict)
                        else {}
                    ),
                )
            return cls(
                features=features,
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                intercept=float(payload["intercept"]),
                threshold=float(payload.get("threshold", 0.5)),
                kind=str(payload.get("kind", "logistic")),
                source=str(target),
                model_version=str(payload.get("model_version", "")),
                data_version=str(payload.get("data_version", "")),
                git_commit=str(payload.get("git_commit", "")),
                deployment_calibration=calibration if isinstance(calibration, dict) else {},
                training_identity=(
                    payload.get("training_identity")
                    if isinstance(payload.get("training_identity"), dict)
                    else {}
                ),
                target_scope=(
                    payload.get("target_scope")
                    if isinstance(payload.get("target_scope"), dict)
                    else {}
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def predict(
        self, features: np.ndarray, names: list[str], times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        matrix = np.asarray(features)
        columns = list(names)
        if self.context_spans and times is not None:
            matrix = window_context(matrix, np.asarray(times), tuple(self.context_spans))
            columns = context_feature_names(columns, tuple(self.context_spans))
        matrix = select_features(matrix, columns, self.features)
        if self.kind == "mlp":
            return mlp_probabilities(
                {"mean": self.mean, "scale": self.scale, "layers": self.layers}, matrix
            )
        normalized = (matrix - self.mean) / np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        logits = normalized @ self.weights + self.intercept
        logits = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


class HeuristicAdModel:
    """Untrained fallback: relative tonal content + level + spectral change.

    It intentionally ignores absolute loudness and any "music == ad" rule.
    Every cue is relative to this episode's own baseline.
    """

    kind = "heuristic"
    source = ""
    threshold = 0.5
    model_version = "heuristic"
    data_version = ""
    deployment_calibration: dict = {}
    features = list(WINDOW_FEATURES)

    def predict(
        self, features: np.ndarray, names: list[str], times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        matrix = select_features(features, names, self.features)
        positions = {name: index for index, name in enumerate(self.features)}
        loud_dev = matrix[:, positions["loud_dev"]]
        musicness = matrix[:, positions["musicness"]]
        band_dist = matrix[:, positions["band_dist"]]
        hf_dev = matrix[:, positions["hf_dev"]]
        corr_change = np.abs(matrix[:, positions["corr_dev"]])
        clip = matrix[:, positions["clip_mean"]]
        floor_dev = matrix[:, positions["floor_dev"]]
        flux = matrix[:, positions["flux_p95"]]
        logits = (
            -1.8
            + 0.95 * np.clip(loud_dev / 8.0, -1.5, 2.0)
            + 1.0 * musicness
            + 0.6 * np.tanh(band_dist / 4.0)
            + 0.4 * np.tanh(hf_dev / 0.05)
            + 0.25 * np.tanh(corr_change / 0.2)
            + 0.3 * np.clip(clip / 0.02, 0.0, 1.0)
            + 0.25 * np.tanh(floor_dev / 6.0)
            + 0.15 * np.tanh(flux / 0.1)
        )
        logits = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


def load_ad_model(path: Optional[str | Path] = None) -> AdModel | HeuristicAdModel:
    if path is not None:
        model = AdModel.load(path)
        if model is None:
            raise ValueError(f"无法加载广告分类模型: {path}（文件缺失、损坏或特征不兼容）")
        return model
    model = AdModel.load(None)
    return model if model is not None else HeuristicAdModel()
