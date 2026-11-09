from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from dai_detect.ad_classifier import AdModel
from dai_detect.audio import PcmHasher, file_sha256
from dai_detect.boundary_model import BOUNDARY_FEATURES
from dai_detect.cli import main
from dai_detect.decode_config import resolve_thresholds
from dai_detect.evaluation import evaluate_events, match_events
from types import SimpleNamespace

from dai_detect.llm import merge_ad_segments
from dai_detect.single_episode import SingleConfig, _calibration_problems, analyze_single
from dai_detect.single_features import WINDOW_FEATURES, window_labels
from dai_detect.temporal_decoder import decode_windows
from scripts.blind_test import _independence_check, _load_ground_truth
from tests import synth

SR48 = 48000


class SingleEpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-single-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        for name, value in (
            ("DAI_AD_MODEL", self.directory / "no-model.json"),
            ("DAI_BOUNDARY_MODEL", self.directory / "no-boundary.json"),
        ):
            previous = os.environ.get(name)
            os.environ[name] = str(value)
            if previous is None:
                self.addCleanup(os.environ.pop, name, None)
            else:
                self.addCleanup(os.environ.__setitem__, name, previous)

    def _episode(self, name: str, ad_start: float, ad_end: float, total: float) -> Path:
        speech_a = synth.speech(ad_start, 201, sr=SR48)
        music = synth.ad(ad_end - ad_start, seed=11, sr=SR48)
        speech_b = synth.speech(total - ad_end, 202, sr=SR48)
        frames = np.concatenate(
            [
                synth.stereo(speech_a, 1, sr=SR48),
                synth.stereo(music, 2, sr=SR48),
                synth.stereo(speech_b, 3, sr=SR48),
            ],
            axis=0,
        )
        path = self.directory / name
        synth.write_stereo_wav(path, frames, sr=SR48)
        return path

    def test_detects_inserted_ad_in_single_recording(self) -> None:
        path = self._episode("with_ad.wav", 20.0, 40.0, 60.0)
        timeline = analyze_single(str(path), SingleConfig())
        self.assertEqual(timeline["mode"], "single_recording_blind")
        self.assertTrue(timeline["ads"], timeline)
        best = max(
            timeline["ads"],
            key=lambda ad: max(0.0, min(ad["end"], 40.0) - max(ad["start"], 20.0)),
        )
        captured = max(0.0, min(best["end"], 40.0) - max(best["start"], 20.0))
        self.assertGreaterEqual(captured, 8.0, best)
        self.assertLessEqual(abs(best["start"] - 20.0), 2.0, best)
        self.assertLessEqual(abs(best["end"] - 40.0), 2.0, best)
        self.assertTrue(best["type"] == "third_party_inserted")
        self.assertEqual(timeline["model"]["boundary"]["kind"], "heuristic")

    def test_plain_speech_has_no_ad(self) -> None:
        frames = synth.stereo(synth.speech(60.0, 211, sr=SR48), 4, sr=SR48)
        path = self.directory / "plain.wav"
        synth.write_stereo_wav(path, frames, sr=SR48)
        timeline = analyze_single(str(path), SingleConfig())
        self.assertEqual(timeline["ads"], [], timeline["ads"])

    def test_temporal_decoder_merges_gaps_and_drops_spikes(self) -> None:
        times = np.arange(120, dtype=np.float64) * 0.5 + 0.5
        probabilities = np.full(120, 0.05)
        probabilities[40:70] = 0.9
        probabilities[60] = 0.1
        probabilities[10] = 0.95
        events = decode_windows(times, probabilities, hop_s=0.5, span_s=1.0, min_ad_s=6.0)
        self.assertEqual(len(events), 1, events)
        self.assertLessEqual(abs(events[0].start - 20.0), 1.5)
        self.assertLessEqual(abs(events[0].end - 35.0), 1.5)

    def test_window_labels_respect_coverage(self) -> None:
        windows = {
            "times": np.arange(10, dtype=np.float64) * 0.5 + 0.5,
            "span_s": 1.0,
            "features": np.zeros((10, len(WINDOW_FEATURES)), dtype=np.float32),
            "names": list(WINDOW_FEATURES),
        }
        indices, labels = window_labels(windows, [{"start_s": 2.0, "end_s": 4.5}])
        self.assertGreater(len(indices), 0)
        self.assertEqual(int(labels.sum()), 4, labels)

    def test_ad_model_round_trip_and_reorders_features(self) -> None:
        payload = {
            "features": ["musicness", "loud_dev"],
            "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "weights": [2.0, 1.0],
            "intercept": -3.0,
            "threshold": 0.5,
            "kind": "logistic",
        }
        path = self.directory / "model.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = AdModel.load(path)
        self.assertIsNotNone(model)
        matrix = np.asarray([[0.0, 0.0], [0.5, 6.0]], dtype=np.float32)
        names = ["loud_dev", "musicness"]
        probabilities = model.predict(matrix, names)
        self.assertGreater(probabilities[1], probabilities[0])
        self.assertGreater(probabilities[1], 0.5)

    def test_event_evaluation_matches_and_scores_boundaries(self) -> None:
        metrics = evaluate_events(
            [{"start_s": 20.2, "end_s": 40.3}],
            [{"start_s": 20.0, "end_s": 40.0}],
            duration_s=3600.0,
        )
        self.assertEqual(metrics["event_precision"], 1.0)
        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertAlmostEqual(metrics["boundary_p95_s"], 0.295, places=3)

    def test_max_matching_beats_greedy_iou(self) -> None:
        # Greedy picks P1-G1 (IoU 0.83) and leaves P2 unmatched.
        # Maximum matching instead uses P1-G2 (0.47) and P2-G1 (0.80).
        predictions = [
            {"start_s": 0.0, "end_s": 12.0},
            {"start_s": 0.0, "end_s": 8.0},
        ]
        truths = [
            {"start_s": 0.0, "end_s": 10.0},
            {"start_s": 5.0, "end_s": 15.0},
        ]
        result = match_events(predictions, truths)
        self.assertEqual(len(result["matches"]), 2, result)
        metrics = evaluate_events(predictions, truths)
        self.assertEqual(metrics["event_recall"], 1.0)
        self.assertEqual(metrics["event_precision"], 1.0)

    def test_decode_thresholds_are_shared_with_inference(self) -> None:
        self.assertEqual(resolve_thresholds(0.5), (0.55, 0.4))
        self.assertEqual(resolve_thresholds(0.8), (0.75, 0.6))
        enter, exit_value = resolve_thresholds(0.1)
        self.assertAlmostEqual(enter, 0.45)
        self.assertAlmostEqual(exit_value, 0.3)

    def test_require_models_rejects_heuristic_fallback(self) -> None:
        path = self._episode("require.wav", 20.0, 40.0, 60.0)
        with self.assertRaises(ValueError):
            analyze_single(str(path), SingleConfig(require_models=True))

    def test_rejects_mismatched_classifier_and_boundary_versions(self) -> None:
        head = {
            "mean": [0.0] * len(BOUNDARY_FEATURES),
            "scale": [1.0] * len(BOUNDARY_FEATURES),
            "weights": [0.0] * len(BOUNDARY_FEATURES),
            "intercept": 0.0,
        }
        segment = {
            "version": 1,
            "kind": "logistic",
            "features": list(WINDOW_FEATURES),
            "mean": [0.0] * len(WINDOW_FEATURES),
            "scale": [1.0] * len(WINDOW_FEATURES),
            "weights": [0.0] * len(WINDOW_FEATURES),
            "intercept": 0.0,
            "threshold": 0.5,
            "data_version": "a" * 64,
        }
        boundary = {
            "version": 1,
            "kind": "onset_offset",
            "features": list(BOUNDARY_FEATURES),
            "onset": head,
            "offset": head,
            "threshold": 0.5,
            "data_version": "b" * 64,
        }
        segment_path = self.directory / "segment.json"
        boundary_path = self.directory / "boundary.json"
        segment_path.write_text(json.dumps(segment), encoding="utf-8")
        boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
        audio = self._episode("versions.wav", 20.0, 40.0, 60.0)
        with self.assertRaises(ValueError):
            analyze_single(
                str(audio),
                SingleConfig(
                    model_path=str(segment_path),
                    boundary_model_path=str(boundary_path),
                    require_models=True,
                ),
            )

    def test_require_models_rejects_missing_data_version(self) -> None:
        head = {
            "mean": [0.0] * len(BOUNDARY_FEATURES),
            "scale": [1.0] * len(BOUNDARY_FEATURES),
            "weights": [0.0] * len(BOUNDARY_FEATURES),
            "intercept": 0.0,
        }
        segment = {
            "version": 1,
            "kind": "logistic",
            "features": list(WINDOW_FEATURES),
            "mean": [0.0] * len(WINDOW_FEATURES),
            "scale": [1.0] * len(WINDOW_FEATURES),
            "weights": [0.0] * len(WINDOW_FEATURES),
            "intercept": 0.0,
            "threshold": 0.5,
            "data_version": "a" * 64,
        }
        boundary = {
            "version": 1,
            "kind": "onset_offset",
            "features": list(BOUNDARY_FEATURES),
            "onset": head,
            "offset": head,
            "threshold": 0.5,
        }
        segment_path = self.directory / "segment_noversion.json"
        boundary_path = self.directory / "boundary_noversion.json"
        segment_path.write_text(json.dumps(segment), encoding="utf-8")
        boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
        audio = self._episode("versions_missing.wav", 20.0, 40.0, 60.0)
        with self.assertRaises(ValueError):
            analyze_single(
                str(audio),
                SingleConfig(
                    model_path=str(segment_path),
                    boundary_model_path=str(boundary_path),
                    require_models=True,
                ),
            )

    def test_require_calibrated_models_rejects_invalid_calibration(self) -> None:
        audio, segment_path, boundary_path = self._calibrated_model_files(
            classifier_calibration={
                "calibration_process_valid": False,
                "final_pipeline_valid": False,
                "converged": False,
                "onset_threshold": 0.5,
                "offset_threshold": 0.5,
            },
            boundary_calibration=self._valid_calibration(),
        )
        with self.assertRaises(ValueError):
            analyze_single(
                str(audio),
                SingleConfig(
                    model_path=str(segment_path),
                    boundary_model_path=str(boundary_path),
                    require_models=True,
                    require_calibrated_models=True,
                ),
            )

    def test_require_calibrated_models_accepts_valid_calibration(self) -> None:
        audio, segment_path, boundary_path = self._calibrated_model_files(
            classifier_calibration=self._valid_calibration(),
            boundary_calibration=self._valid_calibration(),
        )
        timeline = analyze_single(
            str(audio),
            SingleConfig(
                model_path=str(segment_path),
                boundary_model_path=str(boundary_path),
                require_models=True,
                require_calibrated_models=True,
            ),
        )
        self.assertEqual(timeline["model"]["classifier"], "learned")

    @staticmethod
    def _valid_calibration() -> dict:
        return {
            "calibration_process_valid": True,
            "final_pipeline_valid": True,
            "deployment_eligible": True,
            "cv_quality": "strict_nested_grouped_cv",
            "converged": True,
            "onset_threshold": 0.5,
            "offset_threshold": 0.5,
        }

    def _calibrated_model_files(
        self, classifier_calibration: dict, boundary_calibration: dict
    ) -> tuple[Path, Path, Path]:
        head = {
            "mean": [0.0] * len(BOUNDARY_FEATURES),
            "scale": [1.0] * len(BOUNDARY_FEATURES),
            "weights": [0.0] * len(BOUNDARY_FEATURES),
            "intercept": 0.0,
        }
        segment = {
            "version": 1,
            "kind": "logistic",
            "features": list(WINDOW_FEATURES),
            "mean": [0.0] * len(WINDOW_FEATURES),
            "scale": [1.0] * len(WINDOW_FEATURES),
            "weights": [0.0] * len(WINDOW_FEATURES),
            "intercept": 0.0,
            "threshold": 0.5,
            "data_version": "a" * 64,
            "metrics": {"deployment_calibration": classifier_calibration},
        }
        boundary = {
            "version": 1,
            "kind": "onset_offset",
            "features": list(BOUNDARY_FEATURES),
            "onset": head,
            "offset": head,
            "threshold": 0.5,
            "onset_threshold": 0.5,
            "offset_threshold": 0.5,
            "data_version": "a" * 64,
            "metrics": {"deployment_calibration": boundary_calibration},
        }
        segment_path = self.directory / "calibrated_segment.json"
        boundary_path = self.directory / "calibrated_boundary.json"
        segment_path.write_text(json.dumps(segment), encoding="utf-8")
        boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
        audio = self._episode("calibrated.wav", 20.0, 40.0, 60.0)
        return audio, segment_path, boundary_path

    def test_audio_identity_binds_the_exact_variant(self) -> None:
        first = self._episode("variant_a.wav", 20.0, 40.0, 60.0)
        timeline_a = analyze_single(str(first), SingleConfig())
        timeline_a2 = analyze_single(str(first), SingleConfig())
        self.assertEqual(timeline_a["audio"]["id"], timeline_a2["audio"]["id"])
        self.assertEqual(timeline_a["audio"]["source_sha256"], file_sha256(first))
        self.assertEqual(timeline_a["audio"]["id_kind"], "downmix_48k_to_16k")

        # Same episode structure, different inserted creative: the decoded PCM
        # identity must change even though both files have the same duration.
        sr = SR48
        speech_a = synth.speech(20.0, 201, sr=sr)
        speech_b = synth.speech(20.0, 202, sr=sr)
        other_ad = synth.ad(20.0, seed=99, sr=sr)
        frames = np.concatenate(
            [
                synth.stereo(speech_a, 1, sr=sr),
                synth.stereo(other_ad, 2, sr=sr),
                synth.stereo(speech_b, 3, sr=sr),
            ],
            axis=0,
        )
        second = self.directory / "variant_b.wav"
        synth.write_stereo_wav(second, frames, sr=sr)
        timeline_b = analyze_single(str(second), SingleConfig())
        self.assertNotEqual(timeline_a["audio"]["id"], timeline_b["audio"]["id"])

    def test_pcm_hasher_is_deterministic_and_content_sensitive(self) -> None:
        signal = synth.speech(2.0, 301, sr=SR48)
        hasher_a = PcmHasher()
        hasher_b = PcmHasher()
        for block in np.array_split(signal, 4):
            hasher_a.push(block)
            hasher_b.push(block)
        self.assertEqual(hasher_a.hexdigest(), hasher_b.hexdigest())
        altered = signal.copy()
        altered[:100] += 0.01
        hasher_c = PcmHasher()
        hasher_c.push(altered)
        self.assertNotEqual(hasher_a.hexdigest(), hasher_c.hexdigest())

    def test_analyze_never_reads_the_ground_truth(self) -> None:
        path = self._episode("contract.wav", 20.0, 40.0, 60.0)
        # No label, transcript or sibling episode exists in this directory.
        self.assertEqual(sorted(item.name for item in self.directory.iterdir()), ["contract.wav"])
        timeline = analyze_single(str(path), SingleConfig())
        self.assertIn("ads", timeline)

    def test_cli_single_writes_timeline(self) -> None:
        path = self._episode("cli.wav", 20.0, 40.0, 60.0)
        report = self.directory / "timeline.json"
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["single", str(path), "--report", str(report), "--quiet"])
        self.assertEqual(code, 0, stderr.getvalue())
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(payload["mode"], "single_recording_blind")
        self.assertTrue(payload["ads"], payload)


class StrictCalibrationDeploymentTest(unittest.TestCase):
    def _calibration(self, **overrides) -> dict:
        calibration = {
            "calibration_process_valid": True,
            "final_pipeline_valid": True,
            "deployment_eligible": True,
            "cv_quality": "strict_nested_grouped_cv",
            "converged": True,
            "onset_threshold": 0.5,
            "offset_threshold": 0.5,
        }
        calibration.update(overrides)
        return calibration

    def _models(self, classifier=None, boundary=None) -> tuple[SimpleNamespace, SimpleNamespace]:
        return (
            SimpleNamespace(
                deployment_calibration=classifier or self._calibration()
            ),
            SimpleNamespace(
                deployment_calibration=boundary or self._calibration()
            ),
        )

    def test_accepts_jointly_valid_calibrations(self) -> None:
        model, boundary = self._models()
        self.assertEqual(_calibration_problems(model, boundary), [])

    def test_rejects_uncalibrated_classifier(self) -> None:
        model, boundary = self._models(
            classifier=self._calibration(calibration_process_valid=False)
        )
        problems = _calibration_problems(model, boundary)
        self.assertTrue(any("calibration_process_valid" in problem for problem in problems))

    def test_rejects_nonconverged_boundary(self) -> None:
        model, boundary = self._models(boundary=self._calibration(converged=False))
        problems = _calibration_problems(model, boundary)
        self.assertTrue(any("未收敛" in problem for problem in problems))

    def test_rejects_limited_cv_even_if_calibration_valid(self) -> None:
        model, boundary = self._models(
            classifier=self._calibration(
                cv_quality="limited_nested_cv", deployment_eligible=False
            )
        )
        problems = _calibration_problems(model, boundary)
        self.assertTrue(any("cv_quality" in problem for problem in problems))
        self.assertTrue(any("deployment_eligible" in problem for problem in problems))

    def test_rejects_threshold_mismatch(self) -> None:
        model, boundary = self._models(
            boundary=self._calibration(onset_threshold=0.6)
        )
        problems = _calibration_problems(model, boundary)
        self.assertTrue(any("不一致" in problem for problem in problems))

    def test_rejects_missing_boundary_model(self) -> None:
        model = SimpleNamespace(deployment_calibration=self._calibration())
        problems = _calibration_problems(model, None)
        self.assertTrue(any("边界模型缺失" in problem for problem in problems))


class BlindIndependenceTest(unittest.TestCase):
    def _model(self) -> dict:
        return {
            "training_identity": {
                "audio_ids": ["a" * 64],
                "source_sha256": ["s" * 64],
                "feeds": ["Show A"],
            }
        }

    def test_seen_audio_is_rejected_as_independent(self) -> None:
        independence, problems = _independence_check(
            {"id": "a" * 64, "source_sha256": "x" * 64},
            self._model(),
            "Show B",
        )
        self.assertTrue(independence["audio_id_seen_in_training"])
        self.assertFalse(independence["independent"])
        self.assertTrue(problems)

    def test_seen_feed_is_not_unseen_feed_blind_test(self) -> None:
        independence, problems = _independence_check(
            {"id": "b" * 64, "source_sha256": "y" * 64},
            self._model(),
            "Show A",
        )
        self.assertEqual(independence["scope"], "same_feed_holdout")
        self.assertFalse(independence["independent"])
        self.assertTrue(any("feed" in problem for problem in problems))

    def test_unseen_audio_and_feed_is_independent(self) -> None:
        independence, problems = _independence_check(
            {"id": "b" * 64, "source_sha256": "y" * 64},
            self._model(),
            "Show B",
        )
        self.assertEqual(independence["scope"], "unseen_feed_blind_test")
        self.assertTrue(independence["independent"])
        self.assertEqual(problems, [])

    def test_missing_training_identity_cannot_be_independent(self) -> None:
        independence, problems = _independence_check(
            {"id": "b" * 64, "source_sha256": "y" * 64}, {}, "Show B"
        )
        self.assertFalse(independence["independent"])
        self.assertTrue(any("training_identity" in problem for problem in problems))


class ReleaseLabelTest(unittest.TestCase):
    def test_ground_truth_label_source_is_preserved(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="dai-labels-"))
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        for source in ("llm_weak", "human"):
            path = directory / f"{source}.json"
            path.write_text(
                json.dumps(
                    {
                        "duration_s": 60.0,
                        "ads": [{"start_s": 10.0, "end_s": 20.0}],
                        "source_sha256": "s" * 64,
                        "audio_id": "a" * 64,
                        "audio_id_kind": "downmix_48k_to_16k",
                        "labels_source": source,
                    }
                ),
                encoding="utf-8",
            )
            _, _, _, label_source, _ = _load_ground_truth(path)
            self.assertEqual(label_source, source)

    def test_ground_truth_without_label_source_is_unverified(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="dai-labels-"))
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = directory / "plain.json"
        path.write_text(json.dumps({"ads": []}), encoding="utf-8")
        _, _, _, label_source, _ = _load_ground_truth(path)
        self.assertEqual(label_source, "unverified")

    def test_merge_ad_segments_merges_adjacent_third_party_ads(self) -> None:
        segments = [
            {"start_s": 10.0, "end_s": 12.0, "text": "a"},
            {"start_s": 12.5, "end_s": 15.0, "text": "b"},
            {"start_s": 15.5, "end_s": 18.0, "text": "c"},
            {"start_s": 30.0, "end_s": 32.0, "text": "content"},
        ]
        results = {
            "0": {"is_ad": True, "ad_type": "third_party_ad", "host_read": False, "confidence": 0.9},
            "1": {"is_ad": True, "ad_type": "third_party_ad", "host_read": False, "confidence": 0.9},
            "2": {"is_ad": True, "ad_type": "third_party_ad", "host_read": True, "confidence": 0.9},
            "3": {"is_ad": False, "ad_type": "content", "host_read": False, "confidence": 0.9},
        }
        merged = merge_ad_segments(segments, results, max_gap_s=1.0, min_confidence=0.5)
        self.assertEqual(merged, [{"start_s": 10.0, "end_s": 15.0}])


if __name__ == "__main__":
    unittest.main()
