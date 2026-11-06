from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dai_detect.audio import ffmpeg_path
from dai_detect.single_episode import SingleConfig, analyze_single, extract_features
from dai_detect.single_features import WINDOW_FEATURES, build_windows, window_labels
from scripts.train_single_episode import _balance, _fit_logistic
from tests import synth

SR48 = 48000


def _fade(signal: np.ndarray, seconds: float = 1.0, sr: int = SR48) -> np.ndarray:
    count = len(signal)
    ramp = int(seconds * sr)
    window = np.ones(count, dtype=np.float32)
    if ramp > 0:
        window[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        window[-ramp:] = np.linspace(1.0, 0.0, ramp, dtype=np.float32)
    return (signal * window).astype(np.float32)


class HardCaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-hard-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        for name in ("DAI_AD_MODEL", "DAI_BOUNDARY_MODEL"):
            previous = os.environ.get(name)
            os.environ[name] = str(self.directory / "missing.json")
            if previous is None:
                self.addCleanup(os.environ.pop, name, None)
            else:
                self.addCleanup(os.environ.__setitem__, name, previous)

    def _write(self, name: str, parts: list[np.ndarray]) -> Path:
        frames = np.concatenate(parts, axis=0)
        path = self.directory / name
        synth.write_stereo_wav(path, frames, sr=SR48)
        return path

    def _quiet_ad_episode(self, name: str, seed: int) -> Path:
        # Both content and ad use the same amplitude, so loudness is not a cue.
        speech_a = synth.speech(15.0, 1100 + seed, amplitude=0.10, sr=SR48)
        music = synth.ad(12.0, seed=200 + seed, amplitude=0.10, sr=SR48)
        speech_b = synth.speech(15.0, 1200 + seed, amplitude=0.10, sr=SR48)
        return self._write(
            name,
            [
                synth.stereo(speech_a, 300 + seed, width=0.01, sr=SR48),
                synth.stereo(music, 400 + seed, width=0.01, sr=SR48),
                synth.stereo(speech_b, 500 + seed, width=0.01, sr=SR48),
            ],
        )

    def _loud_ad_episode(self, name: str, seed: int) -> Path:
        speech_a = synth.speech(15.0, 1600 + seed, amplitude=0.08, sr=SR48)
        music = synth.ad(12.0, seed=400 + seed, amplitude=0.35, sr=SR48)
        speech_b = synth.speech(15.0, 1700 + seed, amplitude=0.08, sr=SR48)
        return self._write(
            name,
            [
                synth.stereo(speech_a, 900 + seed, sr=SR48),
                synth.stereo(music, 1000 + seed, sr=SR48),
                synth.stereo(speech_b, 1100 + seed, sr=SR48),
            ],
        )

    def _train_minimal_model(self, paths: list[Path]) -> Path:
        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for path in paths:
            mono, forensic = extract_features(str(path))
            windows = build_windows(mono, forensic)
            indices, values = window_labels(
                windows, [{"start_s": 15.0, "end_s": 27.0}]
            )
            features.append(windows["features"][indices])
            labels.append(values)
        X = np.vstack(features).astype(np.float32)
        y = np.concatenate(labels).astype(np.float64)
        model = _fit_logistic(*_balance(X, y, ratio=4, seed=7))
        payload = {
            "version": 1,
            "kind": "logistic",
            "features": list(WINDOW_FEATURES),
            **model,
            "threshold": 0.5,
            "model_version": "test-model",
        }
        target = self.directory / "model.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return target

    def test_learned_model_detects_same_loudness_ad(self) -> None:
        training = [self._quiet_ad_episode(f"train_{index}.wav", index) for index in range(2)]
        model_path = self._train_minimal_model(training)
        test_path = self._quiet_ad_episode("test.wav", 4)
        timeline = analyze_single(
            str(test_path), SingleConfig(model_path=str(model_path))
        )
        self.assertTrue(timeline["ads"], timeline)
        best = max(
            timeline["ads"],
            key=lambda ad: max(0.0, min(ad["end"], 27.0) - max(ad["start"], 15.0)),
        )
        self.assertGreaterEqual(max(0.0, min(best["end"], 27.0) - max(best["start"], 15.0)), 4.0)

    def test_content_music_bed_is_not_an_ad(self) -> None:
        speech = synth.speech(60.0, 1300, amplitude=0.10, sr=SR48)
        bed = synth.ad(60.0, seed=55, amplitude=0.03, sr=SR48)
        mixed = speech + bed
        frames = synth.stereo(mixed, 600, width=0.02, sr=SR48)
        path = self.directory / "content_bed.wav"
        synth.write_stereo_wav(path, frames, sr=SR48)
        timeline = analyze_single(str(path), SingleConfig())
        self.assertEqual(timeline["ads"], [], timeline["ads"])

    def test_normal_edits_are_not_ads(self) -> None:
        parts = [
            synth.stereo(synth.speech(20.0, 1400 + index, amplitude=0.10, sr=SR48), 700 + index, sr=SR48)
            for index in range(3)
        ]
        path = self._write("edits.wav", parts)
        timeline = analyze_single(str(path), SingleConfig())
        self.assertEqual(timeline["ads"], [], timeline["ads"])

    def test_faded_ad_is_detected_with_coarse_bounds(self) -> None:
        speech_a = synth.speech(15.0, 1501, amplitude=0.08, sr=SR48)
        music = _fade(synth.ad(12.0, seed=300, amplitude=0.35, sr=SR48))
        speech_b = synth.speech(15.0, 1502, amplitude=0.08, sr=SR48)
        path = self._write(
            "fade.wav",
            [
                synth.stereo(speech_a, 801, sr=SR48),
                synth.stereo(music, 802, sr=SR48),
                synth.stereo(speech_b, 803, sr=SR48),
            ],
        )
        timeline = analyze_single(str(path), SingleConfig())
        self.assertTrue(timeline["ads"], timeline)
        best = max(
            timeline["ads"],
            key=lambda ad: max(0.0, min(ad["end"], 27.0) - max(ad["start"], 15.0)),
        )
        captured = max(0.0, min(best["end"], 27.0) - max(best["start"], 15.0))
        self.assertGreaterEqual(captured, 8.0, best)
        self.assertLessEqual(abs(best["start"] - 15.0), 2.5, best)
        self.assertLessEqual(abs(best["end"] - 27.0), 2.5, best)

    def test_reencoded_mp3_ad_is_detected(self) -> None:
        path = self._loud_ad_episode("source.wav", 6)
        encoded = self.directory / "encoded.mp3"
        code = subprocess.run(
            [
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-b:a",
                "96k",
                str(encoded),
            ],
            capture_output=True,
            timeout=120,
        ).returncode
        self.assertEqual(code, 0)
        timeline = analyze_single(str(encoded), SingleConfig())
        self.assertTrue(timeline["ads"], timeline)


if __name__ == "__main__":
    unittest.main()
