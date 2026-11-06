from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000


def speech(duration_s: float, seed: int, amplitude: float = 0.08) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(duration_s * SR)
    noise = rng.standard_normal(count).astype(np.float32)
    smooth = np.ones(25, dtype=np.float32) / 25.0
    noise = np.convolve(noise, smooth, mode="same")
    time = np.arange(count) / SR
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * time + rng.uniform(0, 6.28)) ** 2
    phrases = (np.sin(2 * np.pi * 0.28 * time + 1.0) > -0.8).astype(np.float32)
    signal = noise * syllables * phrases
    return (signal / (np.max(np.abs(signal)) + 1e-9) * amplitude).astype(np.float32)


def ad(duration_s: float, seed: int = 7, amplitude: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(duration_s * SR)
    signal = np.zeros(count, dtype=np.float32)
    scale = np.array([110.0, 130.81, 146.83, 164.81, 196.0])
    notes = np.array([440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880.0])
    for index in range(int(np.ceil(duration_s / 0.5))):
        start = int(index * 0.5 * SR)
        end = min(count, int((index + 1) * 0.5 * SR))
        if start >= count:
            break
        time = np.arange(end - start) / SR
        root = scale[rng.integers(0, len(scale))]
        bass = np.sin(2 * np.pi * root * time) + 0.5 * np.sin(2 * np.pi * root * 2 * time)
        signal[start:end] += (bass * np.exp(-3.0 * time) * 0.35).astype(np.float32)
    for index in range(int(duration_s / 0.25)):
        start = int(index * 0.25 * SR)
        end = min(count, int((index + 1) * 0.25 * SR))
        freq = notes[rng.integers(0, len(notes))]
        time = np.arange(end - start) / SR
        envelope = np.minimum(1.0, time / 0.01) * np.exp(-5.0 * time)
        signal[start:end] += (np.sin(2 * np.pi * freq * time) * envelope * 0.3).astype(np.float32)
    for index in range(int(duration_s / 0.5)):
        start = int((index * 0.5 + 0.05) * SR)
        length = int(0.03 * SR)
        end = min(count, start + length)
        if start >= count:
            break
        burst = rng.standard_normal(end - start).astype(np.float32)
        burst *= np.exp(-np.arange(end - start) / (0.008 * SR)) * 0.25
        signal[start:end] += burst
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * amplitude
    return signal.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def make_episode(path: Path, parts: list[np.ndarray]) -> None:
    write_wav(path, np.concatenate(parts))


def make_standard_episodes(directory: Path) -> tuple[Path, Path, Path, float, float]:
    jingle = ad(20.0)
    first = directory / "episode1.wav"
    second = directory / "episode2.wav"
    third = directory / "episode3.wav"
    make_episode(first, [speech(30.0, 11), jingle, speech(35.0, 12)])
    make_episode(second, [speech(45.0, 13), jingle, speech(20.0, 14)])
    make_episode(third, [speech(55.0, 15)])
    return first, second, third, 30.0, 45.0
