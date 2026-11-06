from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dai_detect.calibration import Calibration, apply_calibration, candidate_features
from dai_detect.models import Candidate


def _candidate(tier: str, music: float, loud: float = 0.0) -> Candidate:
    return Candidate(
        start_s=10.0,
        end_s=25.0,
        score=0.6,
        tier=tier,
        evidence={
            "musicness": music,
            "loudness_dev_db": loud,
            "boundary_strength": 0.8,
            "timbre_min": 0.2,
            "position": 0.35,
            "slot_matches": 0,
        },
    )


class CalibrationTest(unittest.TestCase):
    def test_candidate_features_shape(self) -> None:
        features = candidate_features(_candidate("inserted", 0.5))
        self.assertEqual(len(features), 8)

    def test_apply_calibration_drops_low_probability_inserted_and_keeps_repeated(self) -> None:
        calibration = Calibration(
            mean=np.zeros(8),
            scale=np.ones(8),
            weights=np.array([5.0, 0, 0, 0, 0, 0, 0, 0]),
            intercept=-1.0,
            threshold=0.5,
        )
        inserted = _candidate("inserted", 0.0)
        repeated = _candidate("repeated", 0.0)
        kept = apply_calibration([inserted, repeated], calibration)
        self.assertEqual([c.tier for c in kept], ["repeated"])
        self.assertIn("learned_ad_score", inserted.evidence)
        self.assertLess(inserted.evidence["learned_ad_score"], 0.5)
        self.assertEqual(inserted.evidence["heuristic_score"], 0.6)
        self.assertNotIn("learned_ad_score", repeated.evidence)

    def test_load_requires_enabled_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps({"features": []}), encoding="utf-8")
            self.assertIsNone(Calibration.load(path))
            path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "features": [
                            "musicness",
                            "loudness_dev",
                            "boundary",
                            "timbre",
                            "position",
                            "duration",
                            "slot",
                            "repeated",
                        ],
                        "mean": [0] * 8,
                        "scale": [1] * 8,
                        "weights": [0] * 8,
                        "intercept": 0,
                        "threshold": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(Calibration.load(path))


if __name__ == "__main__":
    unittest.main()
