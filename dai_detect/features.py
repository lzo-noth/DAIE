from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
FRAME = 400
HOP = 160
GRID = 10
N_MELS = 24
N_MFCC = 8
FMIN = 100.0
FMAX = 7000.0


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = FRAME, n_mels: int = N_MELS) -> np.ndarray:
    edges = _mel_to_hz(np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index], bins[index + 1], bins[index + 2]
        if right <= left:
            continue
        for k in range(left, center):
            bank[index, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            bank[index, k] = (right - k) / max(1, right - center)
    return bank


def dct_matrix(n_in: int = N_MELS, n_out: int = N_MFCC) -> np.ndarray:
    k = np.arange(1, n_out + 1)[:, None]
    n = np.arange(n_in)[None, :]
    basis = np.cos(np.pi / n_in * (n + 0.5) * k) * np.sqrt(2.0 / n_in)
    return basis.astype(np.float32)


def _aggregate(frames: np.ndarray) -> np.ndarray:
    usable = len(frames) // GRID * GRID
    if usable == 0:
        return frames[:0]
    head = frames[:usable]
    if head.ndim == 1:
        return head.reshape(-1, GRID).mean(axis=1)
    return head.reshape(-1, GRID, head.shape[1]).mean(axis=1)


class FeatureExtractor:
    """Streaming 16 kHz acoustic features on a 10 Hz grid.

    ``push_rows`` returns the newly available 100 Hz rows (and their 10 Hz
    aggregates) so a consumer can score audio while it is still playing.
    ``push``/``finish`` keep the original whole-episode behavior.
    """

    def __init__(self, retain: bool = True) -> None:
        self._carry = np.zeros(0, dtype=np.float32)
        self._window = np.hanning(FRAME).astype(np.float32)
        self._mel = mel_filterbank()
        self._dct = dct_matrix()
        self._band = np.s_[:, 2:176]
        self._retain = retain
        self._loud: list[np.ndarray] = []
        self._flat: list[np.ndarray] = []
        self._flux: list[np.ndarray] = []
        self._mfcc: list[np.ndarray] = []
        self._prev_norm: np.ndarray | None = None
        self._samples = 0
        self._coarse_carry = {
            "loudness_db": np.zeros(0, dtype=np.float32),
            "flatness": np.zeros(0, dtype=np.float32),
            "flux": np.zeros(0, dtype=np.float32),
            "mfcc": np.zeros((0, N_MFCC), dtype=np.float32),
        }

    def push(self, pcm: np.ndarray) -> None:
        rows = self._extract(pcm)
        if self._retain and len(rows["loudness_db"]):
            self._store(rows)

    def push_rows(self, pcm: np.ndarray) -> dict:
        """Return newly extracted fine rows plus their complete 10 Hz groups."""
        rows = self._extract(pcm)
        if self._retain and len(rows["loudness_db"]):
            self._store(rows)
        return {
            "fine": rows,
            "coarse": self._coarse(rows),
            "fine_grid_s": HOP / float(SAMPLE_RATE),
            "coarse_grid_s": HOP * GRID / float(SAMPLE_RATE),
            "duration_s": self._samples / float(SAMPLE_RATE),
        }

    def _store(self, rows: dict) -> None:
        self._loud.append(rows["loudness_db"])
        self._flat.append(rows["flatness"])
        self._flux.append(rows["flux"])
        self._mfcc.append(rows["mfcc"])

    def _extract(self, pcm: np.ndarray) -> dict:
        empty = {
            "loudness_db": np.zeros(0, dtype=np.float32),
            "flatness": np.zeros(0, dtype=np.float32),
            "flux": np.zeros(0, dtype=np.float32),
            "mfcc": np.zeros((0, N_MFCC), dtype=np.float32),
        }
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            return empty
        self._samples += len(pcm)
        buffer = np.concatenate((self._carry, pcm))
        if len(buffer) < FRAME:
            self._carry = buffer
            return empty
        count = 1 + (len(buffer) - FRAME) // HOP
        frames = np.lib.stride_tricks.sliding_window_view(buffer, FRAME)[::HOP][:count]

        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        loudness = 20.0 * np.log10(rms + 1e-7)

        spectra = np.fft.rfft(frames * self._window, axis=1)
        power = (spectra.real ** 2 + spectra.imag ** 2).astype(np.float32)
        magnitude = np.sqrt(power)

        band = power[self._band]
        flatness = np.exp(np.mean(np.log(band + 1e-10), axis=1)) / (np.mean(band, axis=1) + 1e-10)

        normalized = magnitude / (np.sum(magnitude, axis=1, keepdims=True) + 1e-9)
        previous = self._prev_norm if self._prev_norm is not None else normalized[0]
        diff = np.diff(normalized, axis=0, prepend=previous[None, :])
        flux = np.sum(np.maximum(diff, 0.0), axis=1)
        self._prev_norm = normalized[-1]

        logmel = np.log(np.maximum(magnitude @ self._mel.T, 1e-10))
        mfcc = logmel @ self._dct.T

        self._carry = buffer[count * HOP:]
        return {
            "loudness_db": loudness.astype(np.float32),
            "flatness": flatness.astype(np.float32),
            "flux": flux.astype(np.float32),
            "mfcc": mfcc.astype(np.float32),
        }

    def _coarse(self, rows: dict) -> dict:
        carry = self._coarse_carry
        combined = {
            key: np.concatenate((carry[key], rows[key]), axis=0)
            for key in ("loudness_db", "flatness", "flux", "mfcc")
        }
        usable = len(combined["loudness_db"]) // GRID * GRID
        if usable == 0:
            self._coarse_carry = combined
            return {
                "loudness_db": np.zeros(0, dtype=np.float32),
                "flatness": np.zeros(0, dtype=np.float32),
                "flux": np.zeros(0, dtype=np.float32),
                "mfcc": np.zeros((0, N_MFCC), dtype=np.float32),
            }
        output = {
            "loudness_db": combined["loudness_db"][:usable].reshape(-1, GRID).mean(axis=1),
            "flatness": combined["flatness"][:usable].reshape(-1, GRID).mean(axis=1),
            "flux": combined["flux"][:usable].reshape(-1, GRID).mean(axis=1),
            "mfcc": combined["mfcc"][:usable].reshape(-1, GRID, N_MFCC).mean(axis=1),
        }
        self._coarse_carry = {key: combined[key][usable:] for key in combined}
        return output

    def finish(self) -> dict:
        def concat(items: list[np.ndarray], width: int) -> np.ndarray:
            if not items:
                return np.zeros((0, width), dtype=np.float32)
            return np.concatenate(items, axis=0)

        loud = concat(self._loud, 1).reshape(-1)
        flat = concat(self._flat, 1).reshape(-1)
        flux = concat(self._flux, 1).reshape(-1)
        mfcc = concat(self._mfcc, N_MFCC)
        return {
            "duration_s": self._samples / float(SAMPLE_RATE),
            "grid_s": HOP * GRID / float(SAMPLE_RATE),
            "loudness_db": _aggregate(loud),
            "flatness": _aggregate(flat),
            "flux": _aggregate(flux),
            "mfcc": _aggregate(mfcc),
            "fine_loudness_db": loud,
            "fine_flux": flux,
            "fine_mfcc": mfcc,
            "fine_grid_s": HOP / float(SAMPLE_RATE),
        }


class Decimator:
    """Streaming 16 kHz -> 8 kHz low-pass decimator with global phase."""

    def __init__(self, taps: int = 63, cutoff_hz: float = 3600.0, sr: int = SAMPLE_RATE) -> None:
        offsets = np.arange(taps) - (taps - 1) / 2.0
        cutoff = cutoff_hz / sr
        kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets) * np.blackman(taps)
        kernel /= kernel.sum()
        self._kernel = kernel.astype(np.float32)
        self._state = np.zeros(taps - 1, dtype=np.float32)
        self._filtered = 0

    def push(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size == 0:
            return np.zeros(0, dtype=np.float32)
        buffer = np.concatenate((self._state, np.asarray(pcm, dtype=np.float32)))
        filtered = np.convolve(buffer, self._kernel, mode="valid")
        phase = self._filtered % 2
        output = filtered[phase::2]
        self._filtered += len(filtered)
        self._state = buffer[-(len(self._kernel) - 1):]
        return output.astype(np.float32)
