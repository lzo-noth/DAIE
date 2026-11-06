from __future__ import annotations

import numpy as np

SAMPLE_RATE = 8000
FRAME = 512
HOP = 128
FPS = SAMPLE_RATE / HOP

BIN_LO = 13
BIN_HI = 224
BIN_COUNT = BIN_HI - BIN_LO + 1
PEAKS_PER_100MS = 4
FAN_OUT = 8
DT_MAX = 64
FREQ_SPAN = 32
PEAK_OVER_MEAN_DB10 = 60

CONTEXT = 2
LOOKAHEAD = DT_MAX + CONTEXT
TAIL_ROWS = LOOKAHEAD + CONTEXT


class FingerprintExtractor:
    """Wang-style spectral-peak landmark hashes over streaming 8 kHz PCM.

    Keeps 68 frames of context so hashes do not depend on how the stream is
    chunked and anchors always see the full 64-frame target window.
    """

    def __init__(self) -> None:
        self._carry = np.zeros(0, dtype=np.float32)
        self._window = np.hanning(FRAME).astype(np.float32)
        self._hashes: list[int] = []
        self._frames: list[int] = []
        self._tail = np.zeros((0, BIN_COUNT), dtype=np.int16)
        self._tail_frame0 = 0
        self._finalized = False

    def push(self, pcm8k: np.ndarray) -> None:
        if self._finalized:
            raise RuntimeError("fingerprint extractor already finalized")
        pcm = np.asarray(pcm8k, dtype=np.float32)
        if pcm.size == 0:
            return
        buffer = np.concatenate((self._carry, pcm))
        if len(buffer) < FRAME:
            self._carry = buffer
            return
        count = 1 + (len(buffer) - FRAME) // HOP
        frames = np.lib.stride_tricks.sliding_window_view(buffer, FRAME)[::HOP][:count]
        magnitude = np.abs(np.fft.rfft(frames * self._window, axis=1))
        logmel = np.rint(200.0 * np.log10(magnitude[:, BIN_LO:BIN_HI + 1] + 1e-6)).astype(np.int16)
        self._carry = buffer[count * HOP:]

        combined = np.concatenate((self._tail, logmel), axis=0) if len(self._tail) else logmel
        frame0 = self._tail_frame0
        self._process(combined, frame0, final=False)
        keep = min(TAIL_ROWS, len(combined))
        self._tail = combined[-keep:]
        self._tail_frame0 = frame0 + len(combined) - keep

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._finalized:
            if len(self._tail):
                self._process(self._tail, self._tail_frame0, final=True)
                self._tail = np.zeros((0, BIN_COUNT), dtype=np.int16)
            self._finalized = True
        if not self._hashes:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
        return (
            np.asarray(self._hashes, dtype=np.int64),
            np.asarray(self._frames, dtype=np.int32),
        )

    def _process(self, logmel: np.ndarray, frame0: int, final: bool) -> None:
        count = len(logmel)
        if count < CONTEXT * 2 + 3:
            return
        core = logmel[CONTEXT:-CONTEXT]
        frame_mean = logmel.mean(axis=1).astype(np.float32)[CONTEXT:-CONTEXT]
        peak = core > (frame_mean[:, None] + PEAK_OVER_MEAN_DB10)
        middle = peak[:, 1:-1]
        middle &= core[:, 1:-1] >= core[:, :-2]
        middle &= core[:, 1:-1] > core[:, 2:]
        middle &= core[:, 1:-1] >= logmel[1:count - 3, 1:-1]
        middle &= core[:, 1:-1] >= logmel[3:count - 1, 1:-1]
        middle &= core[:, 1:-1] > logmel[0:count - 4, 1:-1]
        middle &= core[:, 1:-1] > logmel[4:count, 1:-1]
        peak[:, 1:-1] = middle
        peak[:, 0] = False
        peak[:, -1] = False
        rows, cols = np.nonzero(peak)
        if len(rows) == 0:
            return

        local_frames = rows + CONTEXT
        freqs = cols + BIN_LO
        mags = core[rows, cols]
        buckets = (frame0 + local_frames) // 6
        order = np.lexsort((-mags, buckets))
        sorted_buckets = buckets[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_buckets)) + 1]
        ends = np.r_[starts[1:], len(sorted_buckets)]
        ranks = np.arange(len(sorted_buckets)) - np.repeat(starts, ends - starts)
        keep = order[ranks < PEAKS_PER_100MS]
        keep = keep[np.argsort(local_frames[keep])]

        peak_frames = local_frames[keep]
        peak_freqs = freqs[keep]
        anchor_limit = count - 1 - CONTEXT if final else count - 1 - DT_MAX - CONTEXT
        for i in range(len(peak_frames)):
            anchor_local = int(peak_frames[i])
            if anchor_local > anchor_limit:
                break
            anchor_frame = int(frame0 + anchor_local)
            anchor_freq = int(peak_freqs[i])
            used = 0
            for j in range(i + 1, len(peak_frames)):
                delta = int(peak_frames[j]) - anchor_local
                if delta > DT_MAX:
                    break
                target_freq = int(peak_freqs[j])
                if abs(target_freq - anchor_freq) > FREQ_SPAN:
                    continue
                self._hashes.append((anchor_freq << 14) | (target_freq << 6) | (delta - 1))
                self._frames.append(anchor_frame)
                used += 1
                if used >= FAN_OUT:
                    break
