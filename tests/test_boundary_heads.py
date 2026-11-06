from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dai_detect.boundary_model import (
    BOUNDARY_FEATURES,
    BoundaryModel,
    _SIGNED_INDEX,
    heuristic_head_scores,
)
from dai_detect.refinement import refine_event
from dai_detect.temporal_decoder import DecodedEvent
from scripts.train_single_episode import _sample_boundary_heads


class BoundaryHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-boundary-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_signed_features_separate_onset_and_offset(self) -> None:
        count = 60
        features = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float32)
        features[:, 0] = 1.0
        features[:30, _SIGNED_INDEX["s_loud"]] = 3.0
        features[30:, _SIGNED_INDEX["s_loud"]] = -3.0
        onset, offset = heuristic_head_scores(features)
        self.assertGreater(float(onset[:30].mean()), float(offset[:30].mean()))
        self.assertGreater(float(offset[30:].mean()), float(onset[30:].mean()))

    def test_boundary_sampler_builds_separate_heads(self) -> None:
        count = 6000
        times = np.arange(count, dtype=np.float64) * 0.01
        features = np.zeros((count, len(BOUNDARY_FEATURES)), dtype=np.float32)
        result = _sample_boundary_heads(
            {"times": times, "features": features},
            [{"start_s": 20.0, "end_s": 40.0}],
            seed=3,
        )
        self.assertGreater(int(result["onset_y"].sum()), 30)
        self.assertGreater(int(result["offset_y"].sum()), 30)
        self.assertEqual(len(result["onset_X"]), len(result["onset_y"]))
        self.assertEqual(len(result["offset_X"]), len(result["offset_y"]))
        fraction = float(result["onset_y"].sum() / len(result["onset_y"]))
        self.assertGreater(fraction, 0.10)
        self.assertLess(fraction, 0.30)

    def test_refinement_uses_onset_for_start_and_offset_for_end(self) -> None:
        times = np.arange(6000, dtype=np.float64) * 0.01
        onset = np.zeros_like(times)
        offset = np.zeros_like(times)
        onset += 0.05
        offset += 0.05
        onset[1900:2300] = np.exp(-((times[1900:2300] - 20.0) ** 2) / 0.02)
        offset[3900:4300] = np.exp(-((times[3900:4300] - 40.0) ** 2) / 0.02)
        event = DecodedEvent(start=18.0, end=42.0, probability=0.9, peak=0.95, first_index=0, last_index=1)
        refined = refine_event(
            event,
            {
                "times": times,
                "onset_scores": onset,
                "offset_scores": offset,
                "scores": np.maximum(onset, offset),
                "kind": "learned_onset_offset",
                "onset_threshold": 0.5,
                "offset_threshold": 0.5,
            },
        )
        self.assertLessEqual(abs(refined["start"] - 20.0), 0.2)
        self.assertLessEqual(abs(refined["end"] - 40.0), 0.2)
        self.assertGreater(refined["evidence"]["onset_score"], 0.9)
        self.assertGreater(refined["evidence"]["offset_score"], 0.9)

    def test_onset_offset_model_round_trip(self) -> None:
        head = {
            "mean": [0.0] * len(BOUNDARY_FEATURES),
            "scale": [1.0] * len(BOUNDARY_FEATURES),
            "weights": [0.0] * len(BOUNDARY_FEATURES),
            "intercept": 5.0,
        }
        offset_head = dict(head, intercept=-5.0)
        payload = {
            "version": 1,
            "kind": "onset_offset",
            "features": list(BOUNDARY_FEATURES),
            "onset": head,
            "offset": offset_head,
            "threshold": 0.5,
            "onset_threshold": 0.4,
            "offset_threshold": 0.6,
        }
        path = self.directory / "boundary.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = BoundaryModel.load(path)
        self.assertIsNotNone(model)
        features = np.zeros((3, len(BOUNDARY_FEATURES)), dtype=np.float32)
        onset_scores = model.predict_onset(features, BOUNDARY_FEATURES)
        offset_scores = model.predict_offset(features, BOUNDARY_FEATURES)
        self.assertGreater(float(onset_scores.mean()), 0.9)
        self.assertLess(float(offset_scores.mean()), 0.1)
        self.assertEqual(model.onset_threshold, 0.4)
        self.assertEqual(model.offset_threshold, 0.6)

    def test_legacy_single_head_model_still_loads(self) -> None:
        payload = {
            "version": 1,
            "kind": "logistic",
            "features": list(BOUNDARY_FEATURES),
            "mean": [0.0] * len(BOUNDARY_FEATURES),
            "scale": [1.0] * len(BOUNDARY_FEATURES),
            "weights": [0.0] * len(BOUNDARY_FEATURES),
            "intercept": 1.0,
            "threshold": 0.5,
        }
        path = self.directory / "legacy.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = BoundaryModel.load(path)
        self.assertIsNotNone(model)
        self.assertEqual(model.kind, "legacy_single")
        features = np.zeros((2, len(BOUNDARY_FEATURES)), dtype=np.float32)
        np.testing.assert_allclose(
            model.predict_onset(features, BOUNDARY_FEATURES),
            model.predict_offset(features, BOUNDARY_FEATURES),
        )


if __name__ == "__main__":
    unittest.main()
