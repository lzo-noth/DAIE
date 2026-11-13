"""AudioSet (YAMNet) embedding view for offline representation experiments.

Production detection must stay dependency free, so the shipped pipeline keeps
using the 77 hand-crafted window features. This module exists to answer one
question offline: do pretrained acoustic embeddings separate ads from ordinary
content where the hand-crafted features cap out? Only
``scripts/probe_embedding.py`` imports it; TensorFlow is loaded lazily.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
PATCH_SAMPLES = 15360  # YAMNet's 0.96 s receptive field
HOP_SAMPLES = 7680  # 0.48 s, YAMNet's own frame hop
EMBEDDING_DIM = 1024

_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
_MODEL_FALLBACK_URL = "https://storage.googleapis.com/tfhub-modules/google/yamnet/1.tar.gz"


class YamnetEmbedder:
    """Stream one 16 kHz mono channel through YAMNet and keep the embeddings.

    The waveform is buffered into 0.96 s patches with a 0.48 s hop, so patch
    centres land on the detector's 0.5 s window grid.
    """

    def __init__(self, batch_size: int = 64, model_url: str = _MODEL_URL) -> None:
        self.batch_size = max(1, int(batch_size))
        self.model_url = model_url
        self._hub = None
        self._model = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._pending: list[np.ndarray] = []
        self._patches = 0
        self._embeddings: list[np.ndarray] = []
        self._times: list[np.ndarray] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        import tensorflow_hub as hub

        self._hub = hub
        try:
            self._model = hub.load(self.model_url)
        except Exception:
            # tfhub.dev 迁移期间不可用时退回同模型的 GCS 分发包。
            self._model = hub.load(_MODEL_FALLBACK_URL)

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._buffer = np.concatenate(
            (self._buffer, np.asarray(samples, dtype=np.float32).reshape(-1))
        )
        while len(self._buffer) >= PATCH_SAMPLES:
            self._pending.append(self._buffer[:PATCH_SAMPLES])
            self._buffer = self._buffer[HOP_SAMPLES:]
            if len(self._pending) >= self.batch_size:
                self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        self._load()
        batch = np.stack(self._pending)
        first_patch = self._patches
        count = len(self._pending)
        self._pending.clear()
        self._patches += count
        _, embeddings, _ = self._model(batch)
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 3:
            values = values.mean(axis=1)
        if values.ndim != 2 or values.shape[1] != EMBEDDING_DIM or values.shape[0] != count:
            raise ValueError(
                f"YAMNet 输出形状异常: {values.shape}（期望 ({count}, {EMBEDDING_DIM})）"
            )
        self._embeddings.append(values.astype(np.float16))
        starts = (first_patch + np.arange(count)) * HOP_SAMPLES
        self._times.append((starts + PATCH_SAMPLES / 2.0) / SAMPLE_RATE)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings, centres_s)`` for every complete patch."""
        self._flush()
        if not self._embeddings:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float16), np.zeros(0)
        return np.vstack(self._embeddings), np.concatenate(self._times)


def window_embeddings(
    times: np.ndarray,
    patch_centres: np.ndarray,
    embeddings: np.ndarray,
    half_s: float = 0.5,
) -> np.ndarray:
    """Average patch embeddings that fall inside each detector window."""
    if len(times) == 0:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if len(embeddings) == 0:
        return np.zeros((len(times), EMBEDDING_DIM), dtype=np.float32)
    ordered = np.argsort(patch_centres)
    centres = np.asarray(patch_centres)[ordered]
    values = np.asarray(embeddings, dtype=np.float32)[ordered]
    low = np.searchsorted(centres, np.asarray(times) - half_s, side="left")
    high = np.searchsorted(centres, np.asarray(times) + half_s, side="right")
    out = np.zeros((len(times), EMBEDDING_DIM), dtype=np.float32)
    for index, (begin, end) in enumerate(zip(low, high)):
        if end > begin:
            out[index] = values[begin:end].mean(axis=0)
        elif len(values):
            nearest = int(np.argmin(np.abs(centres - times[index])))
            out[index] = values[nearest]
    return out
