from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .fingerprint import FPS
from .matching import find_repetitions
from .models import Config

BANK_VERSION = "dai-bank-v1"


@dataclass
class AdCreative:
    """A fingerprint set for one ad creative, frames relative to its start."""

    id: str
    duration_s: float
    hashes: np.ndarray
    frames: np.ndarray
    score: float = 0.8
    label: str = "repeated"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hashes = np.asarray(self.hashes, dtype=np.int64)
        self.frames = np.asarray(self.frames, dtype=np.int32)
        if len(self.hashes) != len(self.frames):
            raise ValueError("hashes and frames must have the same length")

    @property
    def frame_duration(self) -> int:
        return int(round(self.duration_s * FPS))


class AdBank:
    def __init__(self, creatives: Optional[list[AdCreative]] = None) -> None:
        self.creatives: list[AdCreative] = list(creatives or [])
        self._sorted_hashes: Optional[np.ndarray] = None
        self._posting_frames: Optional[np.ndarray] = None
        self._posting_creatives: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.creatives)

    def add(self, creative: AdCreative) -> None:
        self.creatives.append(creative)
        self._sorted_hashes = None

    def index(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._sorted_hashes is None:
            if not self.creatives:
                self._sorted_hashes = np.zeros(0, dtype=np.int64)
                self._posting_frames = np.zeros(0, dtype=np.int32)
                self._posting_creatives = np.zeros(0, dtype=np.int16)
            else:
                hashes = np.concatenate([creative.hashes for creative in self.creatives])
                frames = np.concatenate([creative.frames for creative in self.creatives])
                creative_ids = np.concatenate(
                    [
                        np.full(len(creative.hashes), index, dtype=np.int16)
                        for index, creative in enumerate(self.creatives)
                    ]
                )
                order = np.lexsort((frames, hashes))
                self._sorted_hashes = hashes[order].astype(np.int64)
                self._posting_frames = frames[order].astype(np.int32)
                self._posting_creatives = creative_ids[order]
        return self._sorted_hashes, self._posting_frames, self._posting_creatives

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.creatives:
            hashes = np.zeros(0, dtype=np.int64)
            frames = np.zeros(0, dtype=np.int32)
            creative_ids = np.zeros(0, dtype=np.int16)
            durations = np.zeros(0, dtype=np.float32)
            scores = np.zeros(0, dtype=np.float32)
            ids = np.zeros(0, dtype=np.str_)
            labels = np.zeros(0, dtype=np.str_)
            meta = np.array("[]")
        else:
            hashes = np.concatenate([creative.hashes for creative in self.creatives])
            frames = np.concatenate([creative.frames for creative in self.creatives])
            creative_ids = np.concatenate(
                [
                    np.full(len(creative.hashes), index, dtype=np.int16)
                    for index, creative in enumerate(self.creatives)
                ]
            )
            durations = np.asarray([creative.duration_s for creative in self.creatives], dtype=np.float32)
            scores = np.asarray([creative.score for creative in self.creatives], dtype=np.float32)
            ids = np.asarray([creative.id for creative in self.creatives], dtype=np.str_)
            labels = np.asarray([creative.label for creative in self.creatives], dtype=np.str_)
            meta = np.array(json.dumps([creative.meta for creative in self.creatives], ensure_ascii=False))
        tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                tmp,
                version=np.array(BANK_VERSION),
                hashes=hashes.astype(np.int64),
                frames=frames.astype(np.int32),
                creative=creative_ids.astype(np.int16),
                durations=durations,
                scores=scores,
                ids=ids,
                labels=labels,
                meta=meta,
            )
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> Optional["AdBank"]:
        target = Path(path)
        if not target.exists():
            return None
        try:
            with np.load(target, allow_pickle=False) as data:
                if str(data["version"].item()) != BANK_VERSION:
                    return None
                hashes = data["hashes"]
                frames = data["frames"]
                creative_ids = data["creative"]
                durations = data["durations"]
                scores = data["scores"]
                ids = data["ids"]
                labels = data["labels"]
                raw_meta = data["meta"].item() if data["meta"].ndim == 0 else "[]"
        except (OSError, ValueError, KeyError, IndexError):
            return None
        try:
            metas = json.loads(raw_meta)
        except ValueError:
            metas = []
        creatives: list[AdCreative] = []
        for index in range(len(durations)):
            mask = creative_ids == index
            creatives.append(
                AdCreative(
                    id=str(ids[index]) if len(ids) else f"creative-{index}",
                    duration_s=float(durations[index]),
                    hashes=hashes[mask],
                    frames=frames[mask],
                    score=float(scores[index]),
                    label=str(labels[index]) if len(labels) else "repeated",
                    meta=metas[index] if index < len(metas) and isinstance(metas[index], dict) else {},
                )
            )
        return cls(creatives)


def _interval_key(episode_index: int, start_frame: int, end_frame: int) -> tuple[int, int, int]:
    return episode_index, int(start_frame) // 8, int(end_frame) // 8


def _fixed_slot(intervals: list[tuple[int, int, int]], features) -> bool:
    """Shared intros/outros repeat at almost the same absolute position."""
    if len(intervals) < 2:
        return False
    starts = [start / FPS for _, start, _ in intervals]
    durations = [features[index].duration_s for index, _, _ in intervals]
    span = max(starts) - min(starts)
    near_start = all(start < 180.0 for start in starts)
    near_end = all(start > duration - 180.0 for start, duration in zip(starts, durations))
    if near_start and span <= 60.0:
        return True
    if near_end and span <= 60.0:
        return True
    return False


def _creative_hashes(intervals: list[tuple[int, int, int]], features) -> tuple[np.ndarray, np.ndarray]:
    hashes: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for episode_index, start, end in intervals:
        feature = features[episode_index]
        mask = (feature.hash_frames >= start) & (feature.hash_frames <= end)
        hashes.append(feature.hashes[mask].astype(np.int64))
        frames.append((feature.hash_frames[mask].astype(np.int64) - start).astype(np.int32))
    if not hashes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    combined_hashes = np.concatenate(hashes)
    combined_frames = np.concatenate(frames)
    pairs = np.unique(np.column_stack((combined_hashes, combined_frames)), axis=0)
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int32)


def _creative_id(hashes: np.ndarray) -> str:
    digest = hashlib.sha1(np.ascontiguousarray(np.sort(hashes), dtype=np.int64).tobytes())
    return f"rep-{digest.hexdigest()[:12]}"


def build_bank(
    features,
    cfg: Optional[Config] = None,
    include_fixed: bool = False,
    min_duration_s: float = 8.0,
    max_duration_s: float = 180.0,
    min_match_ratio: float = 0.08,
) -> AdBank:
    """Derive reusable ad creatives from repeated fingerprint matches.

    Shared intros/outros repeat at the same absolute slot, so they are excluded
    by default: skipping those would misfire at the start/end of every episode.
    """
    cfg = cfg or Config()
    if len(features) < 2:
        return AdBank()
    repetitions = find_repetitions(features, cfg)
    seen_intervals: set[tuple[int, int, int]] = set()
    seen_creatives: set[str] = set()
    creatives: list[AdCreative] = []
    for query_index, query_repetitions in enumerate(repetitions):
        for repetition in query_repetitions:
            intervals = [(query_index, repetition.start_frame, repetition.end_frame)]
            intervals.extend(
                (ref.episode_index, ref.start_frame, ref.end_frame) for ref in repetition.refs
            )
            intervals = [
                item
                for item in intervals
                if 0 <= item[0] < len(features) and item[2] > item[1]
            ]
            if len({item[0] for item in intervals}) < 2:
                continue
            keys = {_interval_key(*item) for item in intervals}
            if keys & seen_intervals:
                continue
            durations = [(end - start) / FPS for _, start, end in intervals]
            duration = float(np.median(durations))
            if duration < min_duration_s or duration > max_duration_s:
                continue
            if repetition.match_ratio < min_match_ratio:
                continue
            if not include_fixed and _fixed_slot(intervals, features):
                continue
            hashes, frames = _creative_hashes(intervals, features)
            if len(hashes) < 50:
                continue
            creative_id = _creative_id(hashes)
            if creative_id in seen_creatives:
                seen_intervals.update(keys)
                continue
            seen_intervals.update(keys)
            seen_creatives.add(creative_id)
            starts = [start / FPS for _, start, _ in intervals]
            creatives.append(
                AdCreative(
                    id=creative_id,
                    duration_s=duration,
                    hashes=hashes,
                    frames=frames,
                    score=min(0.95, 0.78 + 0.15 * min(repetition.match_ratio, 1.0)),
                    label="repeated",
                    meta={
                        "match_ratio": round(repetition.match_ratio, 3),
                        "votes": int(repetition.votes),
                        "episodes": len({item[0] for item in intervals}),
                        "position_spread_s": round(max(starts) - min(starts), 1),
                        "intervals": [
                            {"episode": int(index), "start_s": round(start / FPS, 1), "end_s": round(end / FPS, 1)}
                            for index, start, end in intervals
                        ],
                    },
                )
            )
    creatives.sort(key=lambda creative: (-creative.score, creative.id))
    return AdBank(creatives)
