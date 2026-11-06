from __future__ import annotations

import unittest

import numpy as np

from dai_detect.boundaries import acoustic_candidates, detect_boundaries
from dai_detect.models import Config, EpisodeFeatures
from dai_detect.features import Decimator, FeatureExtractor
from dai_detect.fingerprint import FPS, FingerprintExtractor


def _tone(freq: float, duration_s: float, amplitude: float) -> np.ndarray:
    time = np.arange(int(duration_s * 16000)) / 16000.0
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _features(samples: np.ndarray) -> EpisodeFeatures:
    extractor = FeatureExtractor()
    decimator = Decimator()
    fingerprinter = FingerprintExtractor()
    for start in range(0, len(samples), 16000):
        chunk = samples[start:start + 16000]
        extractor.push(chunk)
        fingerprinter.push(decimator.push(chunk))
    data = extractor.finish()
    hashes, frames = fingerprinter.finish()
    return EpisodeFeatures(
        duration_s=data["duration_s"],
        grid_s=data["grid_s"],
        loudness_db=data["loudness_db"],
        flatness=data["flatness"],
        flux=data["flux"],
        mfcc=data["mfcc"],
        hashes=hashes,
        hash_frames=frames,
        fingerprint_fps=FPS,
    )


class BoundaryTest(unittest.TestCase):
    def test_detects_loudness_step(self) -> None:
        signal = np.concatenate([_tone(440.0, 5.0, 0.02), _tone(440.0, 5.0, 0.5)])
        features = _features(signal)
        times, strengths, _ = detect_boundaries(features)
        self.assertTrue(any(abs(float(time) - 5.0) < 0.6 for time in times), times)
        self.assertTrue(np.all((strengths >= 0) & (strengths <= 1)))

    def test_acoustic_candidates_respect_duration(self) -> None:
        signal = np.concatenate(
            [
                _tone(440.0, 30.0, 0.05),
                _tone(660.0, 30.0, 0.4),
                _tone(440.0, 30.0, 0.05),
            ]
        )
        features = _features(signal)
        candidates = acoustic_candidates(features, Config())
        for candidate in candidates:
            self.assertGreaterEqual(candidate.duration_s, Config().acoustic_min_s - 0.01)
            self.assertLessEqual(candidate.duration_s, Config().acoustic_max_s + 0.01)
            self.assertEqual(candidate.tier, "acoustic")