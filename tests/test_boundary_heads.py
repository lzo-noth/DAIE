from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dai_detect.boundary_model import (
    BOUNDARY_FEATURE_SCHEMA,
    BOUNDARY_FEATURES,
    BoundaryModel,
    _SIGNED_INDEX,
    heuristic_head_scores,
)
from dai_detect.refinement import refine_event, refine_events
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

    def test_overlong_block_splits_at_a_strong_internal_change(self) -> None:
        times = np.arange(10000, dtype=np.float64) * 0.01
        onset = np.full_like(times, 0.05)
        offset = np.full_like(times, 0.05)
        onset[0:60] = 1.0
        offset[8940:9000] = 1.0
        onset[4460:4540] = 1.0
        offset[4460:4540] = 1.0
        event = DecodedEvent(
            start=0.0, end=90.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
        change = {
            "times": times,
            "onset_scores": onset,
            "offset_scores": offset,
            "scores": np.maximum(onset, offset),
            "kind": "learned_onset_offset",
            "onset_threshold": 0.5,
            "offset_threshold": 0.5,
        }
        refined = refine_events([event], change, min_duration=10.0)
        self.assertEqual(len(refined), 2, refined)
        self.assertLess(refined[0]["end"], refined[1]["start"] + 1e-6)

    def test_overlong_block_stays_whole_without_a_strong_cut(self) -> None:
        times = np.arange(10000, dtype=np.float64) * 0.01
        onset = np.full_like(times, 0.05)
        offset = np.full_like(times, 0.05)
        onset[0:60] = 1.0
        offset[8940:9000] = 1.0
        onset[4460:4540] = 0.6
        offset[4460:4540] = 0.6
        event = DecodedEvent(
            start=0.0, end=90.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
        change = {
            "times": times,
            "onset_scores": onset,
            "offset_scores": offset,
            "scores": np.maximum(onset, offset),
            "kind": "learned_onset_offset",
            "onset_threshold": 0.5,
            "offset_threshold": 0.5,
        }
        refined = refine_events([event], change, min_duration=10.0)
        self.assertEqual(len(refined), 1, refined)

    def test_edge_only_moves_when_clearly_stronger(self) -> None:
        times = np.arange(6000, dtype=np.float64) * 0.01
        onset = np.full_like(times, 0.02)
        offset = np.full_like(times, 0.02)
        onset[1970:2030] = 0.95
        onset[2570:2630] = 0.99
        offset[3900:4300] = 1.0
        event = DecodedEvent(
            start=20.0, end=40.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
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
        # 旁边 6 s 处的峰只强 4%，不足以把端点拽走。
        self.assertLessEqual(abs(refined["start"] - 20.0), 0.2)

    def test_cue_marker_anchors_the_event_start(self) -> None:
        times = np.arange(6000, dtype=np.float64) * 0.01
        onset = np.full_like(times, 0.02)
        offset = np.full_like(times, 0.02)
        offset[3900:4300] = 1.0
        event = DecodedEvent(
            start=30.0, end=40.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
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
                "cue_starts": [28.4],
            },
        )
        self.assertLessEqual(abs(refined["start"] - 28.4), 0.05)
        self.assertEqual(refined["evidence"]["start_source"], "cue")

    def test_cue_marker_outside_the_radius_is_ignored(self) -> None:
        times = np.arange(6000, dtype=np.float64) * 0.01
        onset = np.full_like(times, 0.02)
        offset = np.full_like(times, 0.02)
        offset[3900:4300] = 1.0
        event = DecodedEvent(
            start=30.0, end=40.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
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
                "cue_starts": [5.0],
            },
        )
        self.assertLessEqual(abs(refined["start"] - 30.0), 0.2)

    def test_sponsor_phrases_split_a_merged_pod(self) -> None:
        times = np.arange(9000, dtype=np.float64) * 0.01
        flat = np.full_like(times, 0.05)
        event = DecodedEvent(
            start=720.0, end=800.0, probability=0.9, peak=0.95, first_index=0, last_index=1
        )
        refined = refine_events(
            [event],
            {
                "times": times,
                "onset_scores": flat,
                "offset_scores": flat,
                "scores": flat,
                "kind": "learned_onset_offset",
                "onset_threshold": 0.5,
                "offset_threshold": 0.5,
                "cue_starts": [731.0],
                "cue_phrases": [731.0, 760.0, 781.0],
            },
            min_duration=10.0,
        )
        starts = [round(item["start"], 1) for item in refined]
        ends = [round(item["end"], 1) for item in refined]
        self.assertEqual(starts, [731.0, 760.0, 781.0], refined)
        self.assertEqual(ends[-1], 800.0)
        self.assertEqual(len(refined), 3)

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
            "feature_schema": BOUNDARY_FEATURE_SCHEMA,
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
            "feature_schema": BOUNDARY_FEATURE_SCHEMA,
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
