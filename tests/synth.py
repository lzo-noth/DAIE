from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000


def speech(duration_s: float, seed: int, amplitude: float = 0.08, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(duration_s * sr)
    noise = rng.standard_normal(count).astype(np.float32)
    smooth_size = max(5, int(round(25 * sr / SR)))
    smooth = np.ones(smooth_size, dtype=np.float32) / smooth_size
    noise = np.convolve(noise, smooth, mode="same")
    time = np.arange(count) / sr
    syllables = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * time + rng.uniform(0, 6.28)) ** 2
    phrases = (np.sin(2 * np.pi * 0.28 * time + 1.0) > -0.8).astype(np.float32)
    signal = noise * syllables * phrases
    return (signal / (np.max(np.abs(signal)) + 1e-9) * amplitude).astype(np.float32)


def ad(duration_s: float, seed: int = 7, amplitude: float = 0.35, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(duration_s * sr)
    signal = np.zeros(count, dtype=np.float32)
    scale = np.array([110.0, 130.81, 146.83, 164.81, 196.0])
    notes = np.array([440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880.0])
    for index in range(int(np.ceil(duration_s / 0.5))):
        start = int(index * 0.5 * sr)
        end = min(count, int((index + 1) * 0.5 * sr))
        if start >= count:
            break
        time = np.arange(end - start) / sr
        root = scale[rng.integers(0, len(scale))]
        bass = np.sin(2 * np.pi * root * time) + 0.5 * np.sin(2 * np.pi * root * 2 * time)
        signal[start:end] += (bass * np.exp(-3.0 * time) * 0.35).astype(np.float32)
    for index in range(int(duration_s / 0.25)):
        start = int(index * 0.25 * sr)
        end = min(count, int((index + 1) * 0.25 * sr))
        freq = notes[rng.integers(0, len(notes))]
        time = np.arange(end - start) / sr
        envelope = np.minimum(1.0, time / 0.01) * np.exp(-5.0 * time)
        signal[start:end] += (np.sin(2 * np.pi * freq * time) * envelope * 0.3).astype(np.float32)
    for index in range(int(duration_s / 0.5)):
        start = int((index * 0.5 + 0.05) * sr)
        length = int(0.03 * sr)
        end = min(count, start + length)
        if start >= count:
            break
        burst = rng.standard_normal(end - start).astype(np.float32)
        burst *= np.exp(-np.arange(end - start) / (0.008 * sr)) * 0.25
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


def stereo(samples: np.ndarray, seed: int, width: float = 0.02, sr: int = SR) -> np.ndarray:
    """Give a mono signal slight L/R decorrelation so stereo cues are defined."""
    rng = np.random.default_rng(seed)
    count = len(samples)
    room = rng.standard_normal((count, 2)).astype(np.float32)
    smooth_size = max(5, int(round(45 * sr / SR)))
    for channel in range(2):
        room[:, channel] = np.convolve(room[:, channel], np.ones(smooth_size) / smooth_size, mode="same")
    room /= np.max(np.abs(room)) + 1e-9
    common = samples
    left = common + room[:, 0] * width
    right = common + room[:, 1] * width
    return np.stack((left, right), axis=1).astype(np.float32)


def write_stereo_wav(path: Path, frames: np.ndarray, sr: int = SR) -> None:
    clipped = np.clip(frames, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.reshape(-1).tobytes())


def make_stereo_episode(path: Path, parts: list[np.ndarray], sr: int) -> None:
    write_stereo_wav(path, np.concatenate(parts, axis=0), sr=sr)


def extract_features(samples: np.ndarray):
    from dai_detect.features import Decimator, FeatureExtractor
    from dai_detect.fingerprint import FPS, FingerprintExtractor
    from dai_detect.models import EpisodeFeatures

    extractor = FeatureExtractor()
    decimator = Decimator()
    fingerprinter = FingerprintExtractor()
    for start in range(0, len(samples), SR):
        chunk = samples[start:start + SR]
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


def make_standard_episodes(directory: Path) -> tuple[Path, Path, Path, float, float]:
    jingle = ad(20.0)
    first = directory / "episode1.wav"
    second = directory / "episode2.wav"
    third = directory / "episode3.wav"
    make_episode(first, [speech(30.0, 11), jingle, speech(35.0, 12)])
    make_episode(second, [speech(45.0, 13), jingle, speech(20.0, 14)])
    make_episode(third, [speech(55.0, 15)])
    return first, second, third, 30.0, 45.0
