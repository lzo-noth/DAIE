"""Blind audio-forensic features on the untouched 48 kHz stereo signal.

The forensic view answers "was this span of audio produced as an independent
insert?" It never decides "is this an ad" by itself: music, loudness jumps or
speaker changes are only cues. Its job is to preserve the information that the
16 kHz mono downmix destroys — high-frequency residual, codec roll-off, noise
floor, stereo correlation and phase continuity.

Rows are produced on a 10 ms grid (1024-sample frames at 48 kHz, 480-sample
hop). All processing is streaming and block-local, so a one-hour episode never
holds raw samples in memory.
"""

from __future__ import annotations

import numpy as np

FORENSIC_SAMPLE_RATE = 48000
FRAME = 1024
HOP = 480
FRAME_GRID_S = HOP / float(FORENSIC_SAMPLE_RATE)

N_BANDS = 16
BAND_EDGES_HZ = np.geomspace(200.0, 20000.0, N_BANDS + 1)
HF_START_HZ = 12000.0
PHASE_BAND_HZ = (1000.0, 8000.0)


def _band_bins(freqs: np.ndarray) -> list[tuple[int, int]]:
    edges = np.searchsorted(freqs, BAND_EDGES_HZ)
    last = len(freqs)
    return [
        (int(edges[i]), min(last, max(int(edges[i]) + 1, int(edges[i + 1]))))
        for i in range(N_BANDS)
    ]


def _empty() -> dict:
    return {
        "duration_s": 0.0,
        "frame_grid_s": FRAME_GRID_S,
        "loudness_db": np.zeros(0, dtype=np.float32),
        "hf_ratio": np.zeros(0, dtype=np.float32),
        "rolloff_hz": np.zeros(0, dtype=np.float32),
        "entropy": np.zeros(0, dtype=np.float32),
        "flatness": np.zeros(0, dtype=np.float32),
        "stereo_corr": np.zeros(0, dtype=np.float32),
        "side_mid_db": np.zeros(0, dtype=np.float32),
        "clip_ratio": np.zeros(0, dtype=np.float32),
        "phase_jump": np.zeros(0, dtype=np.float32),
        "bands": np.zeros((0, N_BANDS), dtype=np.float32),
    }


class StereoForensicExtractor:
    """Streaming 48 kHz stereo forensic frame extractor."""

    def __init__(self) -> None:
        self._carry = np.zeros((0, 2), dtype=np.float32)
        self._window = np.hanning(FRAME).astype(np.float32)
        self._freqs = np.fft.rfftfreq(FRAME, 1.0 / FORENSIC_SAMPLE_RATE)
        self._bands = _band_bins(self._freqs)
        self._band_mask = (self._freqs >= BAND_EDGES_HZ[0]) & (self._freqs <= BAND_EDGES_HZ[-1])
        self._hf_mask = (self._freqs >= HF_START_HZ) & (self._freqs <= BAND_EDGES_HZ[-1])
        self._phase_mask = (self._freqs >= PHASE_BAND_HZ[0]) & (self._freqs <= PHASE_BAND_HZ[1])
        self._prev_spec: np.ndarray | None = None
        self._samples = 0
        self._rows: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "loudness_db",
                "hf_ratio",
                "rolloff_hz",
                "entropy",
                "flatness",
                "stereo_corr",
                "side_mid_db",
                "clip_ratio",
                "phase_jump",
                "bands",
            )
        }

    def push(self, stereo: np.ndarray) -> None:
        stereo = np.asarray(stereo, dtype=np.float32)
        if stereo.size == 0:
            return
        if stereo.ndim == 1:
            stereo = np.stack((stereo, stereo), axis=1)
        self._samples += len(stereo)
        buffer = np.concatenate((self._carry, stereo), axis=0)
        if len(buffer) < FRAME:
            self._carry = buffer
            return
        count = 1 + (len(buffer) - FRAME) // HOP
        views = np.lib.stride_tricks.sliding_window_view(buffer, FRAME, axis=0)
        frames = np.ascontiguousarray(views[::HOP][:count].transpose(0, 2, 1))
        self._carry = buffer[count * HOP:]

        left = frames[:, :, 0]
        right = frames[:, :, 1]
        mix = frames.mean(axis=2)

        rms = np.sqrt(np.mean(mix * mix, axis=1) + 1e-12)
        loudness = 20.0 * np.log10(rms + 1e-7)

        windowed = mix * self._window
        spectra = np.fft.rfft(windowed, axis=1)
        power = (spectra.real ** 2 + spectra.imag ** 2).astype(np.float32)

        total = np.maximum(power[:, self._band_mask].sum(axis=1), 1e-12)
        hf = power[:, self._hf_mask].sum(axis=1)
        hf_ratio = hf / total

        cumulative = np.cumsum(power[:, self._band_mask], axis=1)
        band_freqs = self._freqs[self._band_mask]
        rolloff_index = np.argmax(cumulative >= 0.95 * total[:, None], axis=1)
        rolloff = band_freqs[rolloff_index]

        band_power = np.maximum(power[:, self._band_mask], 1e-12)
        probabilities = band_power / band_power.sum(axis=1, keepdims=True)
        entropy = -(probabilities * np.log(probabilities)).sum(axis=1) / np.log(band_power.shape[1])
        flatness = np.exp(np.mean(np.log(band_power), axis=1)) / (
            np.mean(band_power, axis=1) + 1e-12
        )

        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        covariance = np.mean(left_centered * right_centered, axis=1)
        variance = np.sqrt(
            np.mean(left_centered * left_centered, axis=1)
            * np.mean(right_centered * right_centered, axis=1)
            + 1e-12
        )
        stereo_corr = np.clip(covariance / variance, -1.0, 1.0)

        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        side_mid_db = 10.0 * np.log10(
            np.mean(side * side, axis=1) / (np.mean(mid * mid, axis=1) + 1e-12) + 1e-9
        )

        clip_ratio = np.mean(np.abs(frames) >= 0.98, axis=(1, 2))

        magnitude = np.sqrt(power)
        if self._prev_spec is None:
            previous = spectra
        else:
            previous = np.vstack((self._prev_spec[None, :], spectra[:-1]))
        phase_delta = np.abs(np.angle(spectra * np.conj(previous)))
        phase_weights = magnitude[:, self._phase_mask]
        weight_sum = phase_weights.sum(axis=1) + 1e-9
        phase_jump = (phase_delta[:, self._phase_mask] * phase_weights).sum(axis=1) / weight_sum
        phase_jump = phase_jump / np.pi
        self._prev_spec = spectra[-1]

        bands = np.stack(
            [
                np.mean(power[:, begin:end], axis=1)
                for begin, end in self._bands
            ],
            axis=1,
        )
        bands_db = 10.0 * np.log10(bands + 1e-12)

        self._rows["loudness_db"].append(loudness.astype(np.float32))
        self._rows["hf_ratio"].append(hf_ratio.astype(np.float32))
        self._rows["rolloff_hz"].append(rolloff.astype(np.float32))
        self._rows["entropy"].append(entropy.astype(np.float32))
        self._rows["flatness"].append(flatness.astype(np.float32))
        self._rows["stereo_corr"].append(stereo_corr.astype(np.float32))
        self._rows["side_mid_db"].append(side_mid_db.astype(np.float32))
        self._rows["clip_ratio"].append(clip_ratio.astype(np.float32))
        self._rows["phase_jump"].append(phase_jump.astype(np.float32))
        self._rows["bands"].append(bands_db.astype(np.float32))

    def finish(self) -> dict:
        if not self._rows["loudness_db"]:
            payload = _empty()
            payload["duration_s"] = self._samples / float(FORENSIC_SAMPLE_RATE)
            return payload
        payload: dict = {
            "duration_s": self._samples / float(FORENSIC_SAMPLE_RATE),
            "frame_grid_s": FRAME_GRID_S,
        }
        for key, items in self._rows.items():
            payload[key] = np.concatenate(items, axis=0)
        return payload
