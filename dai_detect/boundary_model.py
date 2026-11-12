"""Frame-level boundary evidence with separate onset and offset heads.

Advertising enters and exits with different production changes: an onset often
raises level, opens the stereo image or changes the spectral profile, while an
offset reverses those changes. A single "big change" score cannot tell them
apart, so the model has two heads over the same frame features.

The features are still decoded-audio features, not bitstream-level forensic
analysis.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_BOUNDARY_MODEL = Path(__file__).resolve().parents[1] / "models" / "boundary_model.json"
BOUNDARY_GRID_S = 0.01
HALF_WINDOW_S = 0.35
LONG_WINDOW_S = 2.0
FLUX_WINDOW_S = 0.15

# 短窗（±0.35s）看到的是"变化瞬间"，长窗（±2s）看到的是"持续状态"。
# 广告插入通常是一段数秒到数十秒的电平/频谱台阶，只有长窗能把它和
# 节目内的普通剪辑区分开，所以两组尺度并存。
BOUNDARY_FEATURES: list[str] = [
    "d_loud",
    "d_flat",
    "d_mfcc",
    "d_flux",
    "d_hf",
    "d_entropy",
    "d_corr",
    "d_width",
    "d_band",
    "s_loud",
    "s_flat",
    "s_hf",
    "s_entropy",
    "s_corr",
    "s_width",
    "d_loud_long",
    "s_loud_long",
    "d_flat_long",
    "d_flux_long",
    "d_hf_long",
    "d_corr_long",
    "d_width_long",
    "d_rolloff",
    "s_rolloff",
    "d_phase",
    "d_centroid",
    "s_centroid",
    "d_mfcc_max",
    "d_loud_var",
]

_SIGNED_INDEX: dict[str, int] = {
    name: BOUNDARY_FEATURES.index(name)
    for name in (
        "s_loud",
        "s_flat",
        "s_hf",
        "s_entropy",
        "s_corr",
        "s_width",
        "s_loud_long",
        "s_rolloff",
        "s_centroid",
    )
}


def _moving_mean(values: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Prev/next window means around each frame.

    Both windows include the current frame. Excluding it makes the "previous"
    window empty at frame 0, so the mean collapses to 0.0 and the first frame of
    every episode looks like a maximally strong boundary.
    """
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(count)
    low = np.maximum(0, index - half)
    high = np.minimum(count, index + half)
    pre = (cumulative[index + 1] - cumulative[low]) / np.maximum(1, index + 1 - low)
    post = (cumulative[high] - cumulative[index]) / np.maximum(1, high - index)
    return pre, post


def _moving_std(values: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    cumulative2 = np.concatenate(([0.0], np.cumsum(values * values)))
    index = np.arange(count)
    low = np.maximum(0, index - half)
    high = np.minimum(count, index + half)

    def variance(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        span = np.maximum(1, end - start)
        mean = (cumulative[end] - cumulative[start]) / span
        mean2 = (cumulative2[end] - cumulative2[start]) / span
        return np.maximum(0.0, mean2 - mean * mean)

    return np.sqrt(variance(low, index + 1)), np.sqrt(variance(index, high))



def _moving_max(values: np.ndarray, half: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if half <= 0 or len(values) == 0:
        return values
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return windows.max(axis=1)


def boundary_change_matrix(mono: dict, forensic: dict) -> dict:
    """Compute boundary change features (absolute and signed) on the 10 ms grid."""
    grid = float(mono.get("fine_grid_s") or BOUNDARY_GRID_S)
    half = max(1, int(round(HALF_WINDOW_S / grid)))
    long_half = max(half, int(round(LONG_WINDOW_S / grid)))
    flux_half = max(1, int(round(FLUX_WINDOW_S / grid)))

    loud = np.asarray(mono["fine_loudness_db"], dtype=np.float64)
    flat = np.asarray(mono["fine_flatness"], dtype=np.float64)
    mfcc = np.asarray(mono["fine_mfcc"], dtype=np.float64)
    flux = np.asarray(mono["fine_flux"], dtype=np.float64)

    f_loud = np.asarray(forensic["loudness_db"], dtype=np.float64)
    hf = np.asarray(forensic["hf_ratio"], dtype=np.float64)
    entropy = np.asarray(forensic["entropy"], dtype=np.float64)
    corr = np.asarray(forensic["stereo_corr"], dtype=np.float64)
    width = np.asarray(forensic["side_mid_db"], dtype=np.float64)
    bands = np.asarray(forensic["bands"], dtype=np.float64)
    rolloff = np.asarray(forensic["rolloff_hz"], dtype=np.float64)
    phase = np.asarray(forensic["phase_jump"], dtype=np.float64)

    count = min(len(loud), len(f_loud))
    if count == 0:
        return {
            "times": np.zeros(0),
            "features": np.zeros((0, len(BOUNDARY_FEATURES)), dtype=np.float32),
            "names": list(BOUNDARY_FEATURES),
            "kind": "empty",
        }
    loud, flat, mfcc, flux = loud[:count], flat[:count], mfcc[:count], flux[:count]
    f_loud, hf, entropy, corr = f_loud[:count], hf[:count], entropy[:count], corr[:count]
    width, bands = width[:count], bands[:count]
    rolloff, phase = rolloff[:count], phase[:count]

    def change(
        series: np.ndarray, window: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        pre, post = _moving_mean(series, half if window is None else window)
        return np.abs(post - pre), post - pre

    def column(name: str) -> int:
        return BOUNDARY_FEATURES.index(name)

    features = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float64)
    features[:, column("d_loud")], features[:, column("s_loud")] = change(loud)
    features[:, column("d_flat")], features[:, column("s_flat")] = change(flat)

    mfcc_delta = np.zeros_like(mfcc)
    for dimension in range(mfcc.shape[1]):
        pre, post = _moving_mean(mfcc[:, dimension], half)
        mfcc_delta[:, dimension] = post - pre
    features[:, column("d_mfcc")] = np.sqrt(np.mean(mfcc_delta ** 2, axis=1))
    features[:, column("d_mfcc_max")] = np.max(np.abs(mfcc_delta), axis=1)
    flux_abs, _ = change(flux)
    features[:, column("d_flux")] = np.maximum(_moving_max(flux, flux_half), flux_abs)
    features[:, column("d_hf")], features[:, column("s_hf")] = change(hf)
    features[:, column("d_entropy")], features[:, column("s_entropy")] = change(entropy)
    features[:, column("d_corr")], features[:, column("s_corr")] = change(corr)
    features[:, column("d_width")], features[:, column("s_width")] = change(width)
    # 削波比例在正常母带的播客里恒为 0，作为特征只会给标准化带来除零噪声；
    # 实测该维在真值边界上的 AUC 恰好 0.000（全常数）。

    band_delta = np.zeros_like(bands)
    for dimension in range(bands.shape[1]):
        pre, post = _moving_mean(bands[:, dimension], half)
        band_delta[:, dimension] = post - pre
    features[:, column("d_band")] = np.sqrt(np.mean(band_delta ** 2, axis=1))

    # 长窗：整段广告的电平/频谱台阶，短窗只会看到台阶边缘的尖峰。
    features[:, column("d_loud_long")], features[:, column("s_loud_long")] = change(
        loud, long_half
    )
    features[:, column("d_flat_long")], _ = change(flat, long_half)
    features[:, column("d_flux_long")], _ = change(flux, long_half)
    features[:, column("d_hf_long")], _ = change(hf, long_half)
    features[:, column("d_corr_long")], _ = change(corr, long_half)
    features[:, column("d_width_long")], _ = change(width, long_half)

    # 频谱重心与相位跳变：拼接式插入的广告常伴随滚降频率与相位不连续。
    weights = np.power(10.0, bands / 10.0)
    positions = np.arange(bands.shape[1], dtype=np.float64)
    centroid = (weights @ positions) / np.maximum(
        weights.sum(axis=1), 1e-9
    )
    features[:, column("d_centroid")], features[:, column("s_centroid")] = change(
        centroid
    )
    features[:, column("d_rolloff")], features[:, column("s_rolloff")] = change(
        rolloff
    )
    features[:, column("d_phase")], _ = change(phase)

    # 电平起伏的变化率：广告段通常比节目本体电平更平稳（配音+固定音量）。
    pre_std, post_std = _moving_std(loud, long_half)
    features[:, column("d_loud_var")] = np.log(
        (post_std + 1e-3) / (pre_std + 1e-3)
    )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    times = np.arange(count, dtype=np.float64) * grid
    return {
        "times": times,
        "features": features.astype(np.float32),
        "names": list(BOUNDARY_FEATURES),
        "kind": "change",
    }


@dataclass
class _Head:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    @classmethod
    def from_payload(cls, payload: dict) -> "_Head":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            intercept=float(payload["intercept"]),
        )

    def predict(self, features: np.ndarray, names: list[str], wanted: list[str]) -> np.ndarray:
        positions = {name: index for index, name in enumerate(names)}
        missing = [name for name in wanted if name not in positions]
        if missing:
            raise ValueError(f"边界特征缺失: {', '.join(missing[:5])}")
        matrix = np.asarray(features)[:, [positions[name] for name in wanted]]
        normalized = (matrix - self.mean) / np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        logits = np.clip(normalized @ self.weights + self.intercept, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class BoundaryModel:
    features: list[str]
    onset: _Head
    offset: _Head
    threshold: float = 0.5
    onset_threshold: float = 0.5
    offset_threshold: float = 0.5
    kind: str = "onset_offset"
    version: str = ""
    data_version: str = ""
    git_commit: str = ""
    source: str = ""
    deployment_calibration: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Optional["BoundaryModel"]:
        target = Path(path or os.environ.get("DAI_BOUNDARY_MODEL", DEFAULT_BOUNDARY_MODEL))
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        try:
            features = [str(name) for name in payload["features"]]
            if features != BOUNDARY_FEATURES:
                # 旧模型按名字取特征；特征集变化后它只能静默退化成启发式，
                # 而项目里禁止边界头静默回退，所以这里直接拒绝加载。
                print(
                    "[boundary] 模型特征集与当前代码不一致，已忽略；请重新训练",
                    file=sys.stderr,
                )
                return None
            if payload.get("kind") == "onset_offset":
                onset = _Head.from_payload(payload["onset"])
                offset = _Head.from_payload(payload["offset"])
                kind = "onset_offset"
            else:
                # Legacy single-head model: apply the same head to both directions.
                head = _Head.from_payload(payload)
                onset = offset = head
                kind = "legacy_single"
            threshold = float(payload.get("threshold", 0.5))
            metrics = payload.get("metrics") or {}
            calibration = metrics.get("deployment_calibration") or {}
            return cls(
                features=features,
                onset=onset,
                offset=offset,
                threshold=threshold,
                onset_threshold=float(payload.get("onset_threshold", threshold)),
                offset_threshold=float(payload.get("offset_threshold", threshold)),
                kind=kind,
                version=str(payload.get("model_version", "")),
                data_version=str(payload.get("data_version", "")),
                git_commit=str(payload.get("git_commit", "")),
                source=str(target),
                deployment_calibration=calibration if isinstance(calibration, dict) else {},
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def predict_onset(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        return self.onset.predict(features, names, self.features)

    def predict_offset(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        return self.offset.predict(features, names, self.features)


_BASE_WEIGHTS = {
    "d_loud": 0.30,
    "d_flat": 0.02,
    "d_mfcc": 0.18,
    "d_flux": 0.02,
    "d_hf": 0.12,
    "d_entropy": 0.04,
    "d_corr": 0.10,
    "d_width": 0.06,
    "d_band": 0.16,
    "d_loud_long": 0.20,
    "d_flat_long": 0.04,
    "d_flux_long": 0.05,
    "d_hf_long": 0.08,
    "d_corr_long": 0.06,
    "d_width_long": 0.04,
    "d_rolloff": 0.05,
    "d_phase": 0.05,
    "d_centroid": 0.06,
    "d_mfcc_max": 0.04,
    "d_loud_var": 0.04,
}
BASE_WEIGHTS = np.asarray(
    [_BASE_WEIGHTS.get(name, 0.0) for name in BOUNDARY_FEATURES], dtype=np.float64
)
DIRECTION_WEIGHTS = {
    "s_loud": 0.25,
    "s_flat": 0.03,
    "s_hf": 0.12,
    "s_entropy": 0.05,
    "s_corr": 0.08,
    "s_width": 0.05,
    "s_loud_long": 0.20,
    "s_centroid": 0.05,
    "s_rolloff": 0.04,
}


def heuristic_head_scores(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Episode-relative onset/offset scores without a trained boundary model."""
    if len(features) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    scales = np.percentile(np.abs(features), 90, axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized_abs = np.clip(np.abs(features) / scales, 0.0, 3.0)
    base = normalized_abs @ BASE_WEIGHTS
    directional = np.zeros(len(features), dtype=np.float64)
    for name, weight in DIRECTION_WEIGHTS.items():
        column = _SIGNED_INDEX[name]
        direction = np.clip(features[:, column] / scales[column], -3.0, 3.0)
        directional += weight * direction
    onset = np.clip((base + directional) / 2.0, 0.0, 1.0)
    offset = np.clip((base - directional) / 2.0, 0.0, 1.0)
    return onset.astype(np.float32), offset.astype(np.float32)


def heuristic_boundary_scores(features: np.ndarray) -> np.ndarray:
    """Compatibility helper: strongest of the two heuristic directions."""
    onset, offset = heuristic_head_scores(features)
    return np.maximum(onset, offset)


def score_boundaries(
    mono: dict,
    forensic: dict,
    model: Optional[BoundaryModel] = None,
) -> dict:
    change = boundary_change_matrix(mono, forensic)
    if len(change["times"]) == 0:
        change["scores"] = np.zeros(0, dtype=np.float32)
        change["onset_scores"] = np.zeros(0, dtype=np.float32)
        change["offset_scores"] = np.zeros(0, dtype=np.float32)
        change["kind"] = "empty"
        return change
    if model is None:
        onset, offset = heuristic_head_scores(change["features"])
        change["kind"] = "heuristic_onset_offset"
    else:
        onset = model.predict_onset(change["features"], change["names"]).astype(np.float32)
        offset = model.predict_offset(change["features"], change["names"]).astype(np.float32)
        change["kind"] = "learned_onset_offset"
        change["onset_threshold"] = model.onset_threshold
        change["offset_threshold"] = model.offset_threshold
    change["onset_scores"] = onset
    change["offset_scores"] = offset
    change["scores"] = np.maximum(onset, offset)
    return change


def load_boundary_model(path: Optional[str | Path] = None) -> Optional[BoundaryModel]:
    if path is not None:
        model = BoundaryModel.load(path)
        if model is None:
            raise ValueError(f"无法加载边界模型: {path}（文件缺失、损坏或特征不兼容）")
        return model
    return BoundaryModel.load(None)
