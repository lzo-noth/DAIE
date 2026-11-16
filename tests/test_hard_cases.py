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
from dai_detect.single_features import WINDOW_FEATURE_SCHEMA, WINDOW_FEATURES, build_windows, window_labels
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
            "feature_schema": WINDOW_FEATURE_SCHEMA,
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

    def test_voice_distance_flags_a_different_speaker(self) -> None:
        sr = SR48
        tone_a = (0.1 * np.sin(2 * np.pi * 440.0 * np.arange(sr) / sr)).astype(np.float32)
        tone_b = (0.1 * np.sin(2 * np.pi * 1800.0 * np.arange(sr) / sr)).astype(np.float32)
        frames = np.concatenate(
            [
                np.stack([tone_a] * 2, axis=1) for _ in range(10)
            ]
            + [np.stack([tone_b] * 2, axis=1) for _ in range(10)]
            + [np.stack([tone_a] * 2, axis=1) for _ in range(10)]
        )
        path = self.directory / "voice.wav"
        synth.write_stereo_wav(path, frames, sr=sr)
        mono, forensic = extract_features(str(path))
        windows = build_windows(mono, forensic)
        columns = {name: index for index, name in enumerate(windows["names"])}
        matrix = windows["features"]
        times = windows["times"]

        def value(name: str, at: float) -> float:
            row = int(np.argmin(np.abs(times - at)))
            return float(matrix[row, columns[name]])

        # 中间段是另一个音色：相对本集基准的距离应显著高于同音色段落。
        self.assertLess(value("mfcc_voice_dist", 5.0), 1.0)
        self.assertGreater(value("mfcc_voice_dist", 15.0), 3.0)
        self.assertLess(value("mfcc_voice_dist", 25.0), 1.0)

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

    def test_silence_bracketed_ad_is_detected(self) -> None:
        speech_a = synth.stereo(synth.speech(15.0, 2101, amplitude=0.08, sr=SR48), 2101, sr=SR48)
        music = synth.stereo(synth.ad(12.0, seed=222, amplitude=0.3, sr=SR48), 2222, sr=SR48)
        speech_b = synth.stereo(synth.speech(15.0, 2102, amplitude=0.08, sr=SR48), 2102, sr=SR48)
        silence = np.zeros((int(0.5 * SR48), 2), dtype=np.float32)
        path = self._write("digital_silence.wav", [speech_a, silence, music, silence, speech_b])
        timeline = analyze_single(str(path), SingleConfig())
        self.assertTrue(timeline["ads"], timeline)

    def test_deep_silence_bracketed_ad_is_detected(self) -> None:
        speech_a = synth.stereo(synth.speech(15.0, 2201, amplitude=0.08, sr=SR48), 2201, sr=SR48)
        music = synth.stereo(synth.ad(12.0, seed=223, amplitude=0.3, sr=SR48), 2233, sr=SR48)
        speech_b = synth.stereo(synth.speech(15.0, 2202, amplitude=0.08, sr=SR48), 2202, sr=SR48)
        # -75 dBFS 噪声底：低于深静音门限，但不是全零 PCM。
        rng = np.random.default_rng(2200)
        floor = (rng.standard_normal(int(1.4 * SR48)) * 1.8e-4).astype(np.float32)
        silence = np.stack((floor, floor[::-1]), axis=1)
        path = self._write("deep_silence.wav", [speech_a, silence, music, silence, speech_b])
        timeline = analyze_single(str(path), SingleConfig())
        self.assertTrue(timeline["ads"], timeline)

    def test_silence_between_normal_edits_is_not_an_ad(self) -> None:
        parts = [
            synth.stereo(synth.speech(15.0, 2301, amplitude=0.1, sr=SR48), 2301, sr=SR48),
            np.zeros((int(0.6 * SR48), 2), dtype=np.float32),
            synth.stereo(synth.speech(15.0, 2302, amplitude=0.1, sr=SR48), 2302, sr=SR48),
        ]
        path = self._write("edits_with_pause.wav", parts)
        timeline = analyze_single(str(path), SingleConfig())
        self.assertEqual(timeline["ads"], [], timeline["ads"])

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
