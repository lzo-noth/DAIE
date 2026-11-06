"""Frame-level boundary evidence.

Advertising is inserted as an independent audio segment, but ordinary editing
also creates cuts. Boundary evidence therefore only answers "how abrupt is the
production change at this instant", and the temporal decoder decides whether
the surrounding span is an ad. A learned model can replace the heuristic, but
the features and the runtime contract stay the same.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
]


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
    """Compute boundary change features on the 10 ms grid."""
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

    def change(series: np.ndarray) -> np.ndarray:
        pre, post = _moving_mean(series, half)
        return np.abs(post - pre)

    features = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float64)
    features[:, 0] = change(loud)
    features[:, 1] = change(flat)
    mfcc_delta = np.zeros_like(mfcc)
    for dimension in range(mfcc.shape[1]):
        pre, post = _moving_mean(mfcc[:, dimension], half)
        mfcc_delta[:, dimension] = post - pre
    features[:, 2] = np.sqrt(np.mean(mfcc_delta ** 2, axis=1))
    features[:, 3] = np.maximum(_moving_max(flux, flux_half), change(flux))
    features[:, 4] = change(hf)
    features[:, 5] = change(entropy)
    features[:, 6] = change(corr)
    features[:, 7] = change(width)
    features[:, 8] = change(clip)
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
class BoundaryModel:
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    threshold: float = 0.5
    source: str = ""

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Optional["BoundaryModel"]:
        target = Path(path or os.environ.get("DAI_BOUNDARY_MODEL", DEFAULT_BOUNDARY_MODEL))
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            return cls(
                features=[str(name) for name in payload["features"]],
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                intercept=float(payload["intercept"]),
                threshold=float(payload.get("threshold", 0.5)),
                source=str(target),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def predict(self, features: np.ndarray, names: list[str]) -> np.ndarray:
        positions = {name: index for index, name in enumerate(names)}
        missing = [name for name in self.features if name not in positions]
        if missing:
            raise ValueError(f"边界特征缺失: {', '.join(missing[:5])}")
        matrix = np.asarray(features)[:, [positions[name] for name in self.features]]
        normalized = (matrix - self.mean) / np.where(np.abs(self.scale) < 1e-9, 1.0, self.scale)
        logits = np.clip(normalized @ self.weights + self.intercept, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))


HEURISTIC_WEIGHTS = np.asarray(
    [0.30, 0.02, 0.18, 0.02, 0.12, 0.04, 0.10, 0.06, 0.02, 0.16], dtype=np.float64
)


def heuristic_boundary_scores(features: np.ndarray) -> np.ndarray:
    """Scale each change type by its own 90th percentile, then mix.

    This is episode-relative by construction: a show with a noisy studio and a
    show with a dead-silent room get comparable boundary scores.
    """
    if len(features) == 0:
        return np.zeros(0, dtype=np.float32)
    scales = np.percentile(np.abs(features), 90, axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    normalized = np.clip(features / scales, 0.0, 3.0)
    combined = normalized @ HEURISTIC_WEIGHTS
    return np.clip(combined / 2.0, 0.0, 1.0).astype(np.float32)


def score_boundaries(
    mono: dict,
    forensic: dict,
    model: Optional[BoundaryModel] = None,
) -> dict:
    change = boundary_change_matrix(mono, forensic)
    if len(change["times"]) == 0:
        change["scores"] = np.zeros(0, dtype=np.float32)
        change["kind"] = "empty"
        return change
    if model is None:
        scores = heuristic_boundary_scores(change["features"])
        change["kind"] = "heuristic"
    else:
        scores = model.predict(change["features"], change["names"]).astype(np.float32)
        change["kind"] = "learned"
    change["scores"] = scores
    return change


def load_boundary_model(path: Optional[str | Path] = None) -> Optional[BoundaryModel]:
    return BoundaryModel.load(path)
