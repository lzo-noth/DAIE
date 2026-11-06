from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .models import Candidate

FEATURES = [
    "musicness",
    "loudness_dev",
    "boundary",
    "timbre",
    "position",
    "duration",
    "slot",
    "repeated",
]


def candidate_features(candidate: Candidate) -> list[float]:
    evidence = candidate.evidence
    duration = max(0.0, candidate.duration_s)
    return [
        float(evidence.get("musicness") or 0.0),
        float(evidence.get("loudness_dev_db") or 0.0) / 6.0,
        float(evidence.get("boundary_strength") or 0.0),
        float(evidence.get("timbre_min") or 0.0),
        float(evidence.get("position") or 0.0),
        math.log1p(duration),
        float(evidence.get("slot_matches") or 0),
        1.0 if candidate.tier == "repeated" else 0.0,
    ]


def label_features(features: dict, tier: str) -> list[float]:
    return [
        float(features.get("musicness") or 0.0),
        float(features.get("loudness_dev_db") or 0.0) / 6.0,
        float(features.get("boundary_strength") or 0.0),
        float(features.get("timbre_min") or 0.0),
        float(features.get("position") or 0.0),
        math.log1p(max(0.0, float(features.get("duration_s") or 0.0))),
        float(features.get("slot_matches") or 0),
        1.0 if tier == "repeated" else 0.0,
    ]


@dataclass
class Calibration:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    threshold: float
    enabled: bool = True
    mode: str = "filter"
    top_k: int = 0

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> Optional["Calibration"]:
        target = Path(path or os.environ.get("DAI_CALIBRATION", "calibration.json"))
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not payload.get("enabled", False):
            return None
        if payload.get("features") != FEATURES:
            return None
        try:
            return cls(
                mean=np.asarray(payload["mean"], dtype=np.float64),
                scale=np.asarray(payload["scale"], dtype=np.float64),
                weights=np.asarray(payload["weights"], dtype=np.float64),
                intercept=float(payload["intercept"]),
                threshold=float(payload.get("threshold", 0.5)),
                mode=str(payload.get("mode", "filter")),
                top_k=int(payload.get("top_k", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def predict(self, features: list[float]) -> float:
        x = np.asarray(features, dtype=np.float64)
        z = float(((x - self.mean) / self.scale) @ self.weights + self.intercept)
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))


def apply_calibration(candidates: list[Candidate], calibration: Calibration) -> list[Candidate]:
    repeated: list[Candidate] = []
    inserted: list[Candidate] = []
    for candidate in candidates:
        if candidate.tier != "inserted":
            repeated.append(candidate)
            continue
        probability = calibration.predict(candidate_features(candidate))
        candidate.evidence["heuristic_score"] = round(candidate.score, 3)
        candidate.evidence["learned_ad_score"] = round(probability, 3)
        candidate.score = round(min(0.99, max(0.0, 0.7 * probability + 0.3 * candidate.score)), 3)
        if calibration.mode == "filter" and probability < calibration.threshold:
            continue
        inserted.append(candidate)
    if calibration.mode == "rank" and calibration.top_k > 0:
        inserted.sort(key=lambda item: (-item.score, item.start_s))
        inserted = inserted[: calibration.top_k]
    return [*repeated, *inserted]
