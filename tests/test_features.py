from __future__ import annotations

import unittest

import numpy as np

from dai_detect.features import Decimator, FeatureExtractor
from dai_detect.fingerprint import FPS, FingerprintExtractor
from dai_detect.models import EpisodeFeatures


def _sine(freq: float, duration_s: float, amplitude: float, sr: int = 16000) -> np.ndarray:
    time = np.arange(int(duration_s * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _make_features(samples: np.ndarray) -> EpisodeFeatures:
    extractor = FeatureExtractor()
    decimator = Decimator()
    fingerprinter = FingerprintExtractor()
    block = 16000
    for start in range(0, len(samples), block):
        chunk = samples[start:start + block]
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
        fine_loudness_db=data["fine_loudness_db"],
        fine_flux=data["fine_flux"],
        fine_mfcc=data["fine_mfcc"],
        fine_grid_s=data["fine_grid_s"],
        hashes=hashes,
        hash_frames=frames,
        fingerprint_fps=FPS,
    )


def _chord(duration_s: float = 1.0, amplitude: float = 0.4) -> np.ndarray:
    time = np.arange(int(duration_s * 16000)) / 16000.0
    signal = sum(np.sin(2 * np.pi * freq * time) for freq in (220.0, 277.18, 329.63, 880.0))
    return (signal / np.max(np.abs(signal)) * amplitude).astype(np.float32)


class FeatureTest(unittest.TestCase):
    def test_decimator_preserves_level_and_rejects_alias(self) -> None:
        decimator = Decimator()
        low = decimator.push(_sine(1000.0, 2.0, 0.5))
        low_rms = float(np.sqrt(np.mean(low[1000:] ** 2)))
        self.assertAlmostEqual(low_rms, 0.5 / np.sqrt(2), delta=0.02)

        decimator = Decimator()
        high = decimator.push(_sine(7000.0, 2.0, 0.5))
        high_rms = float(np.sqrt(np.mean(high[1000:] ** 2)))
        self.assertLess(high_rms, low_rms * 0.1)

    def test_feature_grid_and_loudness(self) -> None:
        features = _make_features(_sine(440.0, 3.0, 0.5))
        self.assertAlmostEqual(features.duration_s, 3.0, delta=0.01)
        self.assertEqual(features.grid_s, 0.1)
        self.assertIn(len(features.loudness_db), (29, 30))
        self.assertTrue(np.all(np.isfinite(features.loudness_db)))
        expected = 20 * np.log10(0.5 / np.sqrt(2))
        self.assertAlmostEqual(float(np.median(features.loudness_db)), expected, delta=1.5)
        self.assertEqual(features.mfcc.shape, (len(features.loudness_db), 8))

    def test_fingerprint_chunk_invariant(self) -> None:
        signal = _chord(3.0)
        whole = FingerprintExtractor()
        whole.push(signal)
        whole_hashes, whole_frames = whole.finish()

        chunked = FingerprintExtractor()
        position = 0
        sizes = (1234, 5000, 777, 9000)
        index = 0
        while position < len(signal):
            step = sizes[index % len(sizes)]
            chunked.push(signal[position:position + step])
            position += step
            index += 1
        chunk_hashes, chunk_frames = chunked.finish()

        np.testing.assert_array_equal(whole_hashes, chunk_hashes)
        np.testing.assert_array_equal(whole_frames, chunk_frames)

    def test_fingerprint_hashes_deterministic(self) -> None:
        signal = _chord(1.0)
        first = _make_features(signal).hashes
        second = _make_features(signal).hashes
        self.assertGreater(len(first), 10)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
