from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio import iter_pcm, load_cache, save_cache, source_key
from .boundaries import acoustic_candidates, segment_evidence
from .features import Decimator, FeatureExtractor
from .fingerprint import FPS, FingerprintExtractor
from .matching import find_repetitions
from .models import Candidate, Config, Episode, EpisodeFeatures, EpisodeResult, MatchRef

FEATURE_VERSION = "features-v1"


def _cache_path(cache_dir: Path, episode: Episode) -> Path:
    return cache_dir / f"{source_key(episode.audio_url, FEATURE_VERSION)}.npz"


def _from_cache(payload: dict) -> Optional[EpisodeFeatures]:
    version = payload.get("version")
    if version is None or str(version) != FEATURE_VERSION:
        return None
    try:
        return EpisodeFeatures(
            duration_s=float(payload["duration_s"]),
            grid_s=float(payload["grid_s"]),
            loudness_db=payload["loudness_db"],
            flatness=payload["flatness"],
            flux=payload["flux"],
            mfcc=payload["mfcc"],
            hashes=payload["hashes"],
            hash_frames=payload["hash_frames"],
            fingerprint_fps=float(payload["fingerprint_fps"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def analyze_episode(
    episode: Episode,
    cfg: Config,
    cache_dir: Path,
    fresh: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> EpisodeFeatures:
    cache_file = _cache_path(cache_dir, episode)
    if cfg.cache and not fresh:
        cached = load_cache(cache_file)
        if cached is not None:
            features = _from_cache(cached)
            if features is not None:
                if progress:
                    progress(f"缓存命中: {episode.title}")
                return features

    if progress:
        progress(f"解码并分析: {episode.title}")
    collector = FeatureExtractor()
    decimator = Decimator()
    fingerprinter = FingerprintExtractor()
    seconds = 0.0
    for block in iter_pcm(episode.audio_url, proxy=cfg.proxy):
        collector.push(block)
        fingerprinter.push(decimator.push(block))
        seconds += len(block) / 16000.0
        if progress and int(seconds) % 60 == 0 and int(seconds) > 0:
            progress(f"  已处理 {int(seconds // 60)} 分钟")
    extracted = collector.finish()
    hashes, hash_frames = fingerprinter.finish()
    features = EpisodeFeatures(
        duration_s=float(extracted["duration_s"]),
        grid_s=float(extracted["grid_s"]),
        loudness_db=extracted["loudness_db"],
        flatness=extracted["flatness"],
        flux=extracted["flux"],
        mfcc=extracted["mfcc"],
        hashes=hashes,
        hash_frames=hash_frames,
        fingerprint_fps=FPS,
    )
    if cfg.cache:
        save_cache(
            cache_file,
            {
                "version": np.array(FEATURE_VERSION),
                "duration_s": np.float64(features.duration_s),
                "grid_s": np.float64(features.grid_s),
                "loudness_db": features.loudness_db,
                "flatness": features.flatness,
                "flux": features.flux,
                "mfcc": features.mfcc,
                "hashes": features.hashes,
                "hash_frames": features.hash_frames,
                "fingerprint_fps": np.float64(features.fingerprint_fps),
            },
        )
    return features


def _overlap_ratio(first: Candidate, second: Candidate) -> float:
    overlap = max(0.0, min(first.end_s, second.end_s) - max(first.start_s, second.start_s))
    smallest = max(1e-6, min(first.duration_s, second.duration_s))
    return overlap / smallest


def _repetition_candidate(
    repetition,
    features: EpisodeFeatures,
    feature_episodes: list[Episode],
    feature_durations: list[float],
    cfg: Config,
) -> Candidate:
    pad = cfg.match_pad_s
    start = max(0.0, repetition.start_frame / FPS - pad)
    end = min(features.duration_s, (repetition.end_frame + 1) / FPS + pad)
    evidence = segment_evidence(features, start, end)
    ratio_strength = min(repetition.match_ratio / 0.15, 1.0)
    support = min(len(repetition.refs) / 2.0, 1.0)
    score = (
        0.55 * ratio_strength
        + 0.25 * support
        + 0.10 * evidence["musicness"]
        + 0.05 * min(evidence["loudness_dev_db"] / 6.0, 1.0)
        + 0.05 * evidence["position"]
    )
    matches = []
    for ref in repetition.refs:
        other = feature_episodes[ref.episode_index]
        other_duration = feature_durations[ref.episode_index]
        matches.append(
            MatchRef(
                episode_id=other.id,
                start_s=round(max(0.0, ref.start_frame / FPS - pad), 2),
                end_s=round(min(other_duration, (ref.end_frame + 1) / FPS + pad), 2),
                match_ratio=round(ref.match_ratio, 3),
            )
        )
    evidence = {**evidence, "matched_episodes": len(repetition.refs), "votes": repetition.votes}
    return Candidate(
        start_s=round(start, 2),
        end_s=round(end, 2),
        score=round(min(score, 0.99), 3),
        tier="repeated",
        evidence=evidence,
        matches=matches,
    )


def _merge_candidates(
    repeated: list[Candidate],
    acoustic: list[Candidate],
    min_score: float,
) -> list[Candidate]:
    kept_acoustic = [
        candidate
        for candidate in acoustic
        if not any(_overlap_ratio(candidate, other) >= 0.5 for other in repeated)
    ]
    combined = [*repeated, *kept_acoustic]
    combined = [candidate for candidate in combined if candidate.score >= min_score]
    combined.sort(key=lambda item: (-item.score, item.start_s))
    return combined


def run(
    episodes: list[Episode],
    cfg: Config,
    cache_dir: Path,
    fresh: bool = False,
    min_score: float = 0.5,
    progress: Optional[Callable[[str], None]] = None,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    features: list[EpisodeFeatures] = []
    feature_episodes: list[Episode] = []
    feature_result_indices: list[int] = []
    for episode in episodes:
        try:
            extracted = analyze_episode(episode, cfg, cache_dir, fresh=fresh, progress=progress)
        except Exception as exc:  # noqa: BLE001 - per-episode failure must not stop the run
            message = str(exc) or exc.__class__.__name__
            results.append(EpisodeResult(episode=episode, error=message))
            if progress:
                progress(f"失败: {episode.title}: {message}")
            continue
        results.append(EpisodeResult(episode=episode, duration_s=extracted.duration_s))
        features.append(extracted)
        feature_episodes.append(episode)
        feature_result_indices.append(len(results) - 1)

    repetitions = find_repetitions(features, cfg)
    feature_durations = [feature.duration_s for feature in features]
    for feature_index, result_index in enumerate(feature_result_indices):
        extracted = features[feature_index]
        repeated = [
            _repetition_candidate(
                repetition,
                extracted,
                feature_episodes,
                feature_durations,
                cfg,
            )
            for repetition in repetitions[feature_index]
        ]
        acoustic = acoustic_candidates(extracted, cfg)
        results[result_index].candidates = _merge_candidates(repeated, acoustic, min_score)
    return results


def make_episode_from_path(path: str) -> Episode:
    resolved = Path(path).resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
    return Episode(id=digest, title=resolved.name, audio_url=str(resolved))
