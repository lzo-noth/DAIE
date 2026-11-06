from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Episode:
    id: str
    title: str
    audio_url: str
    published: Optional[str] = None
    duration_s: Optional[float] = None
    source: Optional[str] = None


@dataclass
class MatchRef:
    episode_id: str
    start_s: float
    end_s: float
    match_ratio: float


@dataclass
class Candidate:
    start_s: float
    end_s: float
    score: float
    tier: str
    evidence: dict = field(default_factory=dict)
    matches: list[MatchRef] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class EpisodeResult:
    episode: Episode
    duration_s: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    error: Optional[str] = None
    audio_id: str = ""
    audio_id_kind: str = ""
    source_sha256: str = ""


@dataclass
class EpisodeFeatures:
    duration_s: float
    grid_s: float
    loudness_db: np.ndarray
    flatness: np.ndarray
    flux: np.ndarray
    mfcc: np.ndarray
    fine_loudness_db: np.ndarray
    fine_flux: np.ndarray
    fine_mfcc: np.ndarray
    fine_grid_s: float
    hashes: np.ndarray
    hash_frames: np.ndarray
    fingerprint_fps: float
    audio_id: str = ""
    audio_id_kind: str = ""
    source_sha256: str = ""

    def times(self) -> np.ndarray:
        return np.arange(len(self.loudness_db), dtype=np.float64) * self.grid_s

    def fine_times(self) -> np.ndarray:
        return np.arange(len(self.fine_loudness_db), dtype=np.float64) * self.fine_grid_s

    def frame_count(self) -> int:
        return int(round(self.duration_s * self.fingerprint_fps))


@dataclass(frozen=True)
class Config:
    min_run_s: float = 8.0
    max_gap_s: float = 2.0
    min_match_ratio: float = 0.05
    match_pad_s: float = 1.0
    acoustic_min_s: float = 8.0
    acoustic_max_s: float = 180.0
    min_acoustic_score: float = 0.55
    max_acoustic_per_episode: int = 8
    insertion_min_s: float = 8.0
    insertion_max_s: float = 180.0
    min_insertion_score: float = 0.52
    max_insertions_per_episode: int = 24
    max_hash_df: int = 50
    min_votes: int = 8
    cache: bool = True
    proxy: Optional[str] = None
