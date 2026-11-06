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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_BOUNDARY_MODEL = Path(__file__).resolve().parents[1] / "models" / "boundary_model.json"
BOUNDARY_GRID_S = 0.01
HALF_WINDOW_S = 0.35
FLUX_WINDOW_S = 0.15

BOUNDARY_FEATURES: list[str] = [
    "d_loud",
    "d_flat",
    "d_mfcc",
    "d_flux",
    "d_hf",
    "d_entropy",
    "d_corr",
    "d_width",
    "d_clip",
    "d_band",
    "s_loud",
    "s_flat",
    "s_hf",
    "s_entropy",
    "s_corr",
    "s_width",
    "s_clip",
]

_SIGNED_INDEX = {
    "s_loud": 10,
    "s_flat": 11,
    "s_hf": 12,
    "s_entropy": 13,
    "s_corr": 14,
    "s_width": 15,
    "s_clip": 16,
}


def _moving_mean(values: np.ndarray, half: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(count)
    low = np.maximum(0, index - half)
    high = np.minimum(count, index + half)
    pre = (cumulative[index] - cumulative[low]) / np.maximum(1, index - low)
    post = (cumulative[high] - cumulative[index]) / np.maximum(1, high - index)
    return pre, post


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
    clip = np.asarray(forensic["clip_ratio"], dtype=np.float64)
    bands = np.asarray(forensic["bands"], dtype=np.float64)

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
    width, clip, bands = width[:count], clip[:count], bands[:count]

    def change(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pre, post = _moving_mean(series, half)
        return np.abs(post - pre), post - pre

    features = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float64)
    features[:, 0], features[:, _SIGNED_INDEX["s_loud"]] = change(loud)
    features[:, 1], features[:, _SIGNED_INDEX["s_flat"]] = change(flat)

    mfcc_delta = np.zeros_like(mfcc)
    for dimension in range(mfcc.shape[1]):
        pre, post = _moving_mean(mfcc[:, dimension], half)
        mfcc_delta[:, dimension] = post - pre
    features[:, 2] = np.sqrt(np.mean(mfcc_delta ** 2, axis=1))
    flux_abs, _ = change(flux)
    features[:, 3] = np.maximum(_moving_max(flux, flux_half), flux_abs)
    features[:, 4], features[:, _SIGNED_INDEX["s_hf"]] = change(hf)
    features[:, 5], features[:, _SIGNED_INDEX["s_entropy"]] = change(entropy)
    features[:, 6], features[:, _SIGNED_INDEX["s_corr"]] = change(corr)
    features[:, 7], features[:, _SIGNED_INDEX["s_width"]] = change(width)
    features[:, 8], features[:, _SIGNED_INDEX["s_clip"]] = change(clip)

    band_delta = np.zeros_like(bands)
    for dimension in range(bands.shape[1]):
        pre, post = _moving_mean(bands[:, dimension], half)
        band_delta[:, dimension] = post - pre
    features[:, 9] = np.sqrt(np.mean(band_delta ** 2, axis=1))
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
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def predict_onset(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        return self.onset.predict(features, names, self.features)

    def predict_offset(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        return self.offset.predict(features, names, self.features)


BASE_WEIGHTS = np.asarray(
    [0.30, 0.02, 0.18, 0.02, 0.12, 0.04, 0.10, 0.06, 0.02, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
DIRECTION_WEIGHTS = {
    "s_loud": 0.25,
    "s_flat": 0.03,
    "s_hf": 0.12,
    "s_entropy": 0.05,
    "s_corr": 0.08,
    "s_width": 0.05,
    "s_clip": 0.02,
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
