"""AudioSet (YAMNet) embedding view for offline representation experiments.

Production detection must stay dependency free, so the shipped pipeline keeps
using the 77 hand-crafted window features. This module exists to answer one
question offline: do pretrained acoustic embeddings separate ads from ordinary
content where the hand-crafted features cap out? Only
``scripts/probe_embedding.py`` imports it, and ONNX Runtime is imported lazily.

The weights are Google's YAMNet (AudioSet, 521 classes) exported to ONNX. Tensor
Flow Hub's original endpoints are retired, so the model is fetched from Hugging
Face mirrors that accept raw audio and expose the 1024-d embeddings.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
PATCH_SAMPLES = 15360  # YAMNet's 0.96 s receptive field
HOP_SAMPLES = 7680  # 0.48 s, YAMNet's own frame hop
EMBEDDING_DIM = 1024

MODEL_URLS = (
    "https://huggingface.co/andrelgomes/yamnet-onnx/resolve/main/yamnet.onnx",
    "https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
)


def ensure_model(path: Path) -> Path:
    """Download the ONNX weights once; they are a few megabytes."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in MODEL_URLS:
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = response.read()
            if len(payload) < 1_000_000:
                errors.append(f"{url}: 只有 {len(payload)} 字节")
                continue
            path.write_bytes(payload)
            return path
        except Exception as exc:  # 网络/镜像不可用，换下一个源
            errors.append(f"{url}: {exc}")
    raise RuntimeError("无法下载 YAMNet ONNX 权重：" + "; ".join(errors))


class YamnetEmbedder:
    """Stream one 16 kHz mono channel through YAMNet and keep the embeddings.

    The waveform is buffered into 0.96 s patches with a 0.48 s hop, so patch
    centres land on the detector's 0.5 s window grid.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threads: int = 1,
    ) -> None:
        self.model_path = Path(
            model_path or os.environ.get("YAMNET_MODEL_PATH") or "out/probe/yamnet.onnx"
        )
        self.threads = max(1, int(threads))
        self._session = None
        self._input = ""
        self._embedding_output = ""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._pending: list[np.ndarray] = []
        self._patches = 0
        self._embeddings: list[np.ndarray] = []
        self._times: list[np.ndarray] = []

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        path = ensure_model(self.model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        for output in self._session.get_outputs():
            shape = output.shape
            if len(shape) == 2 and shape[1] == EMBEDDING_DIM:
                self._embedding_output = output.name
                break
        if not self._embedding_output:
            names = [(item.name, item.shape) for item in self._session.get_outputs()]
            raise ValueError(f"YAMNet ONNX 没有 {EMBEDDING_DIM} 维 embedding 输出: {names}")

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self._buffer = np.concatenate(
            (self._buffer, np.asarray(samples, dtype=np.float32).reshape(-1))
        )
        while len(self._buffer) >= PATCH_SAMPLES:
            self._pending.append(self._buffer[:PATCH_SAMPLES])
            self._buffer = self._buffer[HOP_SAMPLES:]
            if len(self._pending) >= 32:
                self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        self._load()
        values = []
        for patch in self._pending:
            outputs = self._session.run([self._embedding_output], {self._input: patch})
            frames = np.asarray(outputs[0], dtype=np.float32)
            if frames.ndim != 2 or frames.shape[1] != EMBEDDING_DIM:
                raise ValueError(f"YAMNet embedding 形状异常: {frames.shape}")
            values.append(frames.mean(axis=0).astype(np.float16))
        first_patch = self._patches
        count = len(self._pending)
        self._pending.clear()
        self._patches += count
        self._embeddings.append(np.stack(values))
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
