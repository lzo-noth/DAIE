from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .audio import ffmpeg_path, iter_pcm, load_cache, save_cache, source_key
from .boundaries import segment_evidence
from .calibration import Calibration, apply_calibration
from .features import Decimator, FeatureExtractor
from .fingerprint import FPS, FingerprintExtractor
from .insertion import fine_boundaries, insertion_candidates, refine_bounds
from .matching import find_repetitions
from .models import Candidate, Config, Episode, EpisodeFeatures, EpisodeResult, MatchRef

FEATURE_VERSION = "features-v2"


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
            fine_loudness_db=payload["fine_loudness_db"],
            fine_flux=payload["fine_flux"],
            fine_mfcc=payload["fine_mfcc"],
            fine_grid_s=float(payload["fine_grid_s"]),
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
    audio_out: Optional[Path] = None,
) -> EpisodeFeatures:
    cache_file = _cache_path(cache_dir, episode)
    if cfg.cache and not fresh and audio_out is None:
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
    writer = None
    if audio_out is not None:
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        writer = wave.open(str(audio_out), "wb")
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
    seconds = 0.0
    try:
        for block in iter_pcm(episode.audio_url, proxy=cfg.proxy):
            collector.push(block)
            fingerprinter.push(decimator.push(block))
            if writer is not None:
                pcm = np.clip(block * 32767.0, -32768, 32767).astype("<i2")
                writer.writeframes(pcm.tobytes())
            seconds += len(block) / 16000.0
            if progress and int(seconds) % 60 == 0 and int(seconds) > 0:
                progress(f"  已处理 {int(seconds // 60)} 分钟")
    finally:
        if writer is not None:
            writer.close()
    extracted = collector.finish()
    hashes, hash_frames = fingerprinter.finish()
    features = EpisodeFeatures(
        duration_s=float(extracted["duration_s"]),
        grid_s=float(extracted["grid_s"]),
        loudness_db=extracted["loudness_db"],
        flatness=extracted["flatness"],
        flux=extracted["flux"],
        mfcc=extracted["mfcc"],
        fine_loudness_db=extracted["fine_loudness_db"],
        fine_flux=extracted["fine_flux"],
        fine_mfcc=extracted["fine_mfcc"],
        fine_grid_s=float(extracted["fine_grid_s"]),
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
                "fine_loudness_db": features.fine_loudness_db,
                "fine_flux": features.fine_flux,
                "fine_mfcc": features.fine_mfcc,
                "fine_grid_s": np.float64(features.fine_grid_s),
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
    boundaries,
    cfg: Config,
) -> Candidate:
    pad = cfg.match_pad_s
    start = max(0.0, repetition.start_frame / FPS - pad)
    end = min(features.duration_s, (repetition.end_frame + 1) / FPS + pad)
    refined_start, refined_end, refine_strength = refine_bounds(start, end, boundaries)
    refined = False
    if refine_strength >= 0.5 and refined_end - refined_start >= 5.0:
        start, end = refined_start, refined_end
        refined = True
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
    evidence = {
        **evidence,
        "matched_episodes": len(repetition.refs),
        "votes": repetition.votes,
        "refined": refined,
        "refine_strength": round(refine_strength, 3),
    }
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
    inserted: list[Candidate],
) -> list[Candidate]:
    for candidate in repeated:
        overlaps = [other for other in inserted if _overlap_ratio(candidate, other) >= 0.5]
        if overlaps:
            best = max(overlaps, key=lambda other: other.score)
            candidate.evidence["insertion_score"] = best.score
            candidate.evidence["insertion_kind"] = best.evidence.get("kind")
    kept_inserted = [
        candidate
        for candidate in inserted
        if not any(_overlap_ratio(candidate, other) >= 0.5 for other in repeated)
    ]
    combined = [*repeated, *kept_inserted]
    combined.sort(key=lambda item: (-item.score, item.start_s))
    return combined


def _apply_slot_consistency(
    results: list[EpisodeResult],
    successful_indices: list[int],
) -> None:
    groups: dict[str, list[tuple[int, Candidate]]] = {}
    for result_index in successful_indices:
        source = results[result_index].episode.source or "default"
        for candidate in results[result_index].candidates:
            if candidate.tier == "inserted":
                groups.setdefault(source, []).append((result_index, candidate))
    for items in groups.values():
        if len(items) < 2:
            continue
        for result_index, candidate in items:
            matches = [
                other
                for other_index, other in items
                if other_index != result_index
                and abs(candidate.start_s - other.start_s) <= 60.0
                and 0.4 <= candidate.duration_s / max(other.duration_s, 1e-6) <= 2.5
            ]
            if not matches:
                continue
            candidate.evidence["slot_matches"] = len(matches)
            candidate.evidence["slot_delta_s"] = round(
                min(abs(candidate.start_s - other.start_s) for other in matches), 1
            )
            # LLM labels showed slot matches are common on content. Only boost
            # a slot-consistent candidate that already has acoustic ad evidence.
            ad_like = (
                candidate.evidence.get("musicness", 0.0) >= 0.3
                or candidate.evidence.get("loudness_dev_db", 0.0) >= 5.0
            )
            if ad_like:
                candidate.score = round(
                    min(0.95, candidate.score + (0.05 if len(matches) == 1 else 0.08)), 3
                )


def _write_clips(
    result: EpisodeResult,
    audio_path: Path,
    clips_dir: Path,
    limit: int = 40,
) -> None:
    if not audio_path.exists():
        return
    clips_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", result.episode.title)[:40] or result.episode.id[:8]
    for index, candidate in enumerate(result.candidates[:limit]):
        start = max(0.0, candidate.start_s)
        duration = min(10.0, candidate.end_s - start)
        if duration < 1.0:
            continue
        output = clips_dir / f"{slug}_{index:02d}_{int(candidate.start_s)}s_{candidate.tier}.mp3"
        command = [
            ffmpeg_path(),
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.2f}",
            "-t",
            f"{duration:.2f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            str(output),
        ]
        try:
            code = subprocess.run(command, capture_output=True, timeout=60, check=False).returncode
        except (OSError, subprocess.TimeoutExpired):
            continue
        if code == 0 and output.exists():
            candidate.evidence["clip"] = output.name


def run(
    episodes: list[Episode],
    cfg: Config,
    cache_dir: Path,
    fresh: bool = False,
    min_score: float = 0.5,
    progress: Optional[Callable[[str], None]] = None,
    clips_dir: Optional[Path] = None,
) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    features: list[EpisodeFeatures] = []
    feature_episodes: list[Episode] = []
    feature_result_indices: list[int] = []
    audio_paths: dict[int, Path] = {}
    audio_root = Path(tempfile.mkdtemp(prefix="dai-audio-")) if clips_dir is not None else None
    effective_fresh = fresh or clips_dir is not None
    try:
        for episode in episodes:
            try:
                audio_out = audio_root / f"{len(results)}.wav" if audio_root is not None else None
                extracted = analyze_episode(
                    episode,
                    cfg,
                    cache_dir,
                    fresh=effective_fresh,
                    progress=progress,
                    audio_out=audio_out,
                )
            except Exception as exc:  # noqa: BLE001 - per-episode failure must not stop the run
                message = str(exc) or exc.__class__.__name__
                results.append(EpisodeResult(episode=episode, error=message))
                if progress:
                    progress(f"失败: {episode.title}: {message}")
                continue
            results.append(EpisodeResult(episode=episode, duration_s=extracted.duration_s))
            if audio_out is not None:
                audio_paths[len(results) - 1] = audio_out
            features.append(extracted)
            feature_episodes.append(episode)
            feature_result_indices.append(len(results) - 1)

        repetitions = find_repetitions(features, cfg)
        feature_durations = [feature.duration_s for feature in features]
        boundaries = [fine_boundaries(feature) for feature in features]
        for feature_index, result_index in enumerate(feature_result_indices):
            extracted = features[feature_index]
            repeated = [
                _repetition_candidate(
                    repetition,
                    extracted,
                    feature_episodes,
                    feature_durations,
                    boundaries[feature_index],
                    cfg,
                )
                for repetition in repetitions[feature_index]
            ]
            inserted = insertion_candidates(extracted, cfg, boundaries[feature_index])
            results[result_index].candidates = _merge_candidates(repeated, inserted)

        _apply_slot_consistency(results, feature_result_indices)
        calibration = Calibration.load()
        if calibration is not None:
            for result_index in feature_result_indices:
                results[result_index].candidates = apply_calibration(
                    results[result_index].candidates, calibration
                )
        for result_index in feature_result_indices:
            results[result_index].candidates = [
                candidate for candidate in results[result_index].candidates if candidate.score >= min_score
            ]
            results[result_index].candidates.sort(key=lambda item: (-item.score, item.start_s))
        if clips_dir is not None:
            for result_index, audio_path in audio_paths.items():
                _write_clips(results[result_index], audio_path, clips_dir)
        return results
    finally:
        if audio_root is not None:
            shutil.rmtree(audio_root, ignore_errors=True)


def make_episode_from_path(path: str) -> Episode:
    resolved = Path(path).resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
    return Episode(id=digest, title=resolved.name, audio_url=str(resolved))
