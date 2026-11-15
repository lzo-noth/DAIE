from __future__ import annotations

import unittest

import numpy as np

from dai_detect.forensic import (
    DEEP_SILENCE_DBFS,
    DIGITAL_SILENCE_DBFS,
    FRAME_GRID_S,
    RMS_HOP,
    StereoForensicExtractor,
    silence_run_s,
)

SR = 48000


def _sine(duration_s: float, amplitude: float, freq: float = 440.0) -> np.ndarray:
    time = np.arange(int(duration_s * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def _extract(frames: np.ndarray, block_s: float = 2.0) -> dict:
    extractor = StereoForensicExtractor()
    step = max(1, int(block_s * SR))
    for start in range(0, len(frames), step):
        extractor.push(frames[start:start + step])
    return extractor.finish()


class AntiPhaseTest(unittest.TestCase):
    def test_energy_survives_opposite_channels(self) -> None:
        """左右反相时 (L+R)/2 会互相抵消，能量必须按声道功率求和。"""
        mono = _sine(1.0, 0.5)
        frames = np.stack((mono, -mono), axis=1)
        payload = _extract(frames)
        expected = 20.0 * np.log10(0.5 / np.sqrt(2.0))
        self.assertAlmostEqual(float(np.median(payload["loudness_db"])), expected, delta=1.0)
        self.assertGreater(float(np.mean(payload["hf_ratio"])), 0.0)
        self.assertGreater(float(np.mean(payload["entropy"])), 0.0)

    def test_rms_matches_in_phase_and_anti_phase(self) -> None:
        mono = _sine(0.5, 0.4)
        in_phase = _extract(np.stack((mono, mono), axis=1))
        anti_phase = _extract(np.stack((mono, -mono), axis=1))
        np.testing.assert_allclose(
            in_phase["rms_db"], anti_phase["rms_db"], atol=0.5
        )


class SilenceRunTest(unittest.TestCase):
    def test_runs_measure_the_contiguous_low_energy_span(self) -> None:
        levels = np.asarray([-10.0, -90.0, -95.0, -85.0, -12.0])
        runs = silence_run_s(levels, DIGITAL_SILENCE_DBFS)
        expected = np.asarray([0.0, 0.03, 0.03, 0.03, 0.0])
        np.testing.assert_allclose(runs, expected, atol=1e-9)

    def test_extractor_emits_a_ten_millisecond_energy_grid(self) -> None:
        tone = _sine(1.0, 0.4)
        quiet = np.zeros(int(1.5 * SR), dtype=np.float32)
        frames = np.stack((np.concatenate((tone, quiet)), np.concatenate((tone, quiet))), axis=1)
        payload = _extract(frames)
        self.assertGreaterEqual(len(payload["rms_db"]), int(2.5 * SR / RMS_HOP) - 2)
        deep = silence_run_s(payload["rms_db"], DEEP_SILENCE_DBFS)
        self.assertAlmostEqual(float(deep.max()), 1.5, delta=0.05)
        self.assertGreater(float(np.median(payload["rms_db"][:50])), DEEP_SILENCE_DBFS)
        # 零采样比例与 dBFS 门限是两条独立证据。
        self.assertAlmostEqual(float(np.mean(payload["zero_fraction"][110:240])), 1.0, places=5)
        self.assertLess(float(np.mean(payload["zero_fraction"][:50])), 0.01)

    def test_grid_constant_matches_hop(self) -> None:
        self.assertAlmostEqual(FRAME_GRID_S, RMS_HOP / float(SR))


if __name__ == "__main__":
    unittest.main()
