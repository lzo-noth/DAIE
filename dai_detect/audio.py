from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .net import HTTP_USER_AGENT

SAMPLE_RATE = 16000
FORENSIC_SAMPLE_RATE = 48000


class AudioError(RuntimeError):
    pass


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise AudioError("找不到 ffmpeg；请安装 ffmpeg 或 imageio-ffmpeg") from exc


def _ffmpeg_env(proxy: Optional[str]) -> dict:
    env = os.environ.copy()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = proxy
    return env


def _is_remote(src: str) -> bool:
    return src.startswith(("http://", "https://"))


def source_key(src: str, version: str) -> str:
    if _is_remote(src):
        basis = f"url|{src}|{version}"
    else:
        path = Path(src)
        try:
            stat = path.stat()
            basis = f"file|{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{version}"
        except OSError:
            basis = f"file|{path.resolve()}|missing|{version}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of the exact source bytes (empty string when unavailable)."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def source_sha256(src: str) -> str:
    """Exact-byte identity for local files; remote URLs have no byte identity."""
    if _is_remote(src):
        return ""
    return file_sha256(src)


class PcmHasher:
    """Content identity of the decoded PCM stream.

    The hash is computed from 16-bit quantized samples, so it is stable across
    containers and repeated decodes of the same audio, but changes whenever the
    actual audio content changes — which is exactly what DAI does between two
    requests of the same episode URL.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def push(self, pcm: np.ndarray) -> None:
        values = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return
        quantized = np.clip(np.round(values * 32767.0), -32768.0, 32767.0).astype("<i2")
        self._digest.update(quantized.tobytes())

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _iter_raw(
    src: str,
    channels: int,
    block_s: float,
    proxy: Optional[str],
    sample_rate: int,
) -> Iterator[np.ndarray]:
    """Stream decoded PCM as ``(frames, channels)`` float32 blocks.

    One ffmpeg process produces the raw stream; callers reshape it. This keeps
    the forensic (48 kHz stereo) and model (16 kHz mono) views derived from the
    same decode pass.
    """
    ffmpeg = ffmpeg_path()
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if _is_remote(src):
        cmd += ["-user_agent", HTTP_USER_AGENT]
    cmd += [
        "-i",
        src,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=_ffmpeg_env(proxy),
        bufsize=0,
    )
    assert process.stdout is not None and process.stderr is not None
    block_bytes = max(2 * channels, int(block_s * sample_rate) * 2 * channels)
    pending = bytearray()
    completed = False
    try:
        while True:
            chunk = process.stdout.read(65536)
            if not chunk:
                break
            pending.extend(chunk)
            while len(pending) >= block_bytes:
                data = bytes(pending[:block_bytes])
                del pending[:block_bytes]
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                yield samples.reshape(-1, channels)
        remainder = len(pending) % (2 * channels)
        if remainder:
            del pending[len(pending) - remainder:]
        if pending:
            samples = np.frombuffer(bytes(pending), dtype="<i2").astype(np.float32) / 32768.0
            if len(samples) >= channels:
                yield samples.reshape(-1, channels)
        completed = True
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stderr = process.stderr.read().decode("utf-8", "replace").strip()
        process.stdout.close()
        process.stderr.close()
        if completed and process.returncode not in (0, None):
            raise AudioError(f"ffmpeg 解码失败 (rc={process.returncode}): {stderr[-800:]}")


def iter_pcm(
    src: str,
    block_s: float = 5.0,
    proxy: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    for block in _iter_raw(src, 1, block_s, proxy, sample_rate):
        yield block.reshape(-1)


def iter_pcm_stereo(
    src: str,
    block_s: float = 2.0,
    proxy: Optional[str] = None,
    sample_rate: int = FORENSIC_SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    """Yield 48 kHz stereo blocks shaped ``(frames, 2)`` for forensic analysis."""
    yield from _iter_raw(src, 2, block_s, proxy, sample_rate)


class DownmixResampler:
    """48 kHz stereo mix -> 16 kHz mono by FIR low-pass + integer decimation.

    The forensic view needs the untouched stereo signal; the acoustic models
    need the conventional 16 kHz mono signal. Both are produced in one decode
    pass and stay sample-aligned.
    """

    def __init__(
        self,
        src_rate: int = FORENSIC_SAMPLE_RATE,
        dst_rate: int = SAMPLE_RATE,
        cutoff_hz: float = 7000.0,
        taps: int = 61,
    ) -> None:
        if src_rate <= 0 or dst_rate <= 0 or src_rate % dst_rate:
            raise AudioError(f"不支持的重采样率: {src_rate} -> {dst_rate}")
        self.factor = src_rate // dst_rate
        offsets = np.arange(taps) - (taps - 1) / 2.0
        cutoff = min(cutoff_hz, dst_rate / 2.0 - 200.0) / src_rate
        kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets) * np.blackman(taps)
        kernel /= kernel.sum()
        self._kernel = kernel.astype(np.float32)
        self._state = np.zeros(taps - 1, dtype=np.float32)
        self._filtered = 0

    def push(self, mono: np.ndarray) -> np.ndarray:
        if mono.size == 0:
            return np.zeros(0, dtype=np.float32)
        buffer = np.concatenate((self._state, np.asarray(mono, dtype=np.float32)))
        filtered = np.convolve(buffer, self._kernel, mode="valid")
        phase = (-self._filtered) % self.factor
        output = filtered[phase:: self.factor]
        self._filtered += len(filtered)
        self._state = buffer[-(len(self._kernel) - 1):]
        return output.astype(np.float32)


def save_cache(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_cache(path: Path) -> Optional[dict]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except (OSError, ValueError):
        return None
