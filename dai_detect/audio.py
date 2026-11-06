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


class AudioError(RuntimeError):
    pass


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


def iter_pcm(
    src: str,
    block_s: float = 5.0,
    proxy: Optional[str] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Iterator[np.ndarray]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioError("找不到 ffmpeg，请先安装并加入 PATH")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-user_agent",
        HTTP_USER_AGENT,
        "-i",
        src,
        "-vn",
        "-ac",
        "1",
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
    block_bytes = max(2, int(block_s * sample_rate) * 2)
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
                yield np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        if len(pending) % 2:
            del pending[-1:]
        if pending:
            yield np.frombuffer(bytes(pending), dtype="<i2").astype(np.float32) / 32768.0
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
