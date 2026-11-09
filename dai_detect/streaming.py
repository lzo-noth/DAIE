from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bank import AdBank, AdCreative
from .features import Decimator, FeatureExtractor
from .fingerprint import FPS, FingerprintExtractor
from .insertion import Boundary
from .models import Candidate, Episode

SAMPLE_RATE = 16000
DELTA_BIN = 4


@dataclass
class StreamConfig:
    """Tuning for playback-time detection.

    The defaults favour precision: a bank hit must accumulate a dense cluster
    of aligned fingerprint votes before it is announced, and an unknown
    (acoustic-only) ad has to be a clean, music/loudness-delimited insert.
    """

    feature_history_s: float = 180.0
    boundary_lookahead_s: float = 1.0
    boundary_history_s: float = 4.0
    boundary_min_gap_s: float = 6.0
    boundary_score: float = 0.52
    provisional_s: float = 2.0
    acoustic_open_score: float = 0.56
    acoustic_close_score: float = 0.55
    pre_roll_open: bool = True
    pre_roll_score: float = 0.64
    min_candidate_s: float = 8.0
    max_candidate_s: float = 180.0
    report_late: bool = True
    bank_min_votes: int = 12
    bank_min_span_s: float = 0.6
    bank_min_ratio: float = 0.10
    bank_max_hash_df: int = 200
    bank_timeout_s: float = 2.5
    bank_pad_s: float = 0.25
    dedupe_overlap: float = 0.5


@dataclass
class StreamEvent:
    action: str  # "start" | "end" | "cancel"
    candidate: Candidate
    emitted_at_s: float
    creative_id: Optional[str] = None


class _RollingFeatures:
    """Bounded-window storage for 100 Hz acoustic rows."""

    KEYS = ("loudness_db", "flatness", "flux", "mfcc")

    def __init__(self, max_seconds: float, grid_s: float) -> None:
        self.grid_s = grid_s
        self.max_frames = max(10, int(round(max_seconds / grid_s)))
        self.chunks: dict[str, list[np.ndarray]] = {key: [] for key in self.KEYS}
        self.chunk_lengths: list[int] = []
        self.base = 0
        self.total = 0
        self._cache: Optional[tuple[dict[str, np.ndarray], int, int]] = None

    def append(self, rows: dict) -> None:
        length = len(rows["loudness_db"])
        if length == 0:
            return
        for key in self.KEYS:
            self.chunks[key].append(rows[key])
        self.chunk_lengths.append(length)
        self.total += length
        self._cache = None
        excess = self.total - self.max_frames - self.base
        while excess > 0 and self.chunk_lengths:
            first = self.chunk_lengths[0]
            if excess >= first:
                for key in self.KEYS:
                    self.chunks[key].pop(0)
                self.chunk_lengths.pop(0)
                self.base += first
                excess -= first
            else:
                for key in self.KEYS:
                    self.chunks[key][0] = self.chunks[key][0][excess:]
                self.chunk_lengths[0] -= excess
                self.base += excess
                excess = 0

    def materialize(self) -> tuple[dict[str, np.ndarray], int, int]:
        if self._cache is not None:
            return self._cache
        arrays = {}
        for key in self.KEYS:
            chunks = self.chunks[key]
            if chunks:
                arrays[key] = np.concatenate(chunks, axis=0)
            elif key == "mfcc":
                arrays[key] = np.zeros((0, 8), dtype=np.float32)
            else:
                arrays[key] = np.zeros(0, dtype=np.float32)
        self._cache = (arrays, self.base, self.total)
        return self._cache


@dataclass
class _VoteState:
    creative: int
    delta: int
    count: int = 0
    min_q: int = 0
    max_q: int = 0
    min_ref: int = 0
    max_ref: int = 0
    last_seen_s: float = 0.0
    opened: bool = False


class OnlineMatcher:
    """Incremental fingerprint matching against a bank of known creatives."""

    def __init__(self, bank: AdBank, cfg: StreamConfig) -> None:
        self.bank = bank
        self.cfg = cfg
        self._hashes, self._frames, self._creatives = bank.index()
        self._votes: dict[tuple[int, int], _VoteState] = {}
        self._active: dict[int, Candidate] = {}
        self._last_match_s: dict[int, float] = {}

    def _density(self, creative_index: int) -> float:
        creative = self.bank.creatives[creative_index]
        frames = max(1, creative.frame_duration)
        return len(creative.hashes) / float(frames)

    def query(self, hashes: np.ndarray, frames: np.ndarray, time_s: float) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if len(hashes) == 0 or len(self._hashes) == 0:
            return events
        left = np.searchsorted(self._hashes, hashes, side="left")
        right = np.searchsorted(self._hashes, hashes, side="right")
        counts = right - left
        usable = (counts > 0) & (counts <= self.cfg.bank_max_hash_df)
        if not usable.any():
            return self._poll(time_s, events)
        left, counts, positions = left[usable], counts[usable], np.flatnonzero(usable)
        total = int(counts.sum())
        if total == 0:
            return self._poll(time_s, events)
        query_slots = np.repeat(positions, counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        postings = np.repeat(left, counts) + offsets
        query_frames = frames[query_slots].astype(np.int64)
        posting_frames = self._frames[postings].astype(np.int64)
        creative_ids = self._creatives[postings].astype(np.int64)
        raw_delta = query_frames - posting_frames
        delta = (raw_delta // DELTA_BIN) * DELTA_BIN + DELTA_BIN // 2
        keys = creative_ids * 1_000_000 + (delta + 500_000)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1, len(sorted_keys)]
        for start, end in zip(boundaries, boundaries[1:]):
            group = order[start:end]
            key = int(sorted_keys[start])
            creative = key // 1_000_000
            delta_value = int(round(key % 1_000_000)) - 500_000
            state_key = (creative, delta_value)
            state = self._votes.get(state_key)
            if state is None:
                state = _VoteState(
                    creative=creative,
                    delta=delta_value,
                    min_q=int(query_frames[group].min()),
                    max_q=int(query_frames[group].max()),
                    min_ref=int(posting_frames[group].min()),
                    max_ref=int(posting_frames[group].max()),
                )
                self._votes[state_key] = state
            else:
                state.min_q = min(state.min_q, int(query_frames[group].min()))
                state.max_q = max(state.max_q, int(query_frames[group].max()))
                state.min_ref = min(state.min_ref, int(posting_frames[group].min()))
                state.max_ref = max(state.max_ref, int(posting_frames[group].max()))
            state.count += len(group)
            state.last_seen_s = time_s
            if state.opened or state.creative in self._active:
                self._last_match_s[state.creative] = time_s
            else:
                self._maybe_open(state, time_s, events)
        self._prune(time_s)
        return self._poll(time_s, events)

    def _maybe_open(self, state: _VoteState, time_s: float, events: list[StreamEvent]) -> None:
        if state.creative in self._active:
            return
        span_s = (state.max_q - state.min_q + 1) / FPS
        if state.count < self.cfg.bank_min_votes or span_s < self.cfg.bank_min_span_s:
            return
        creative = self.bank.creatives[state.creative]
        expected = max(1.0, (state.max_ref - state.min_ref + 1) * self._density(state.creative))
        ratio = state.count / expected
        if ratio < self.cfg.bank_min_ratio:
            return
        start_s = max(0.0, state.delta / FPS - self.cfg.bank_pad_s)
        end_s = start_s + creative.duration_s
        score = min(0.99, 0.80 + 0.15 * min(ratio, 1.0) + 0.04 * creative.score)
        candidate = Candidate(
            start_s=round(start_s, 2),
            end_s=round(end_s, 2),
            score=round(score, 3),
            tier="repeated",
            evidence={
                "source": "bank",
                "creative": creative.id,
                "votes": state.count,
                "match_ratio": round(min(ratio, 1.0), 3),
                "span_s": round(span_s, 1),
                "expected_duration_s": round(creative.duration_s, 1),
                "detected_at_s": round(time_s, 2),
                "latency_s": round(max(0.0, time_s - start_s), 2),
            },
        )
        state.opened = True
        self._active[state.creative] = candidate
        self._last_match_s[state.creative] = time_s
        events.append(
            StreamEvent(
                action="start",
                candidate=candidate,
                emitted_at_s=round(time_s, 2),
                creative_id=creative.id,
            )
        )

    def _poll(self, time_s: float, events: list[StreamEvent]) -> list[StreamEvent]:
        for creative_index, candidate in list(self._active.items()):
            expected_end = candidate.end_s
            last_match = self._last_match_s.get(creative_index, time_s)
            if time_s >= expected_end + 0.3:
                candidate.end_s = round(expected_end, 2)
                events.append(
                    StreamEvent("end", candidate, round(time_s, 2), creative_id=candidate.evidence.get("creative"))
                )
                del self._active[creative_index]
            elif time_s >= last_match + self.cfg.bank_timeout_s:
                truncated_end = min(expected_end, last_match + 0.5)
                candidate.end_s = round(max(candidate.start_s + 1.0, truncated_end), 2)
                candidate.evidence["truncated"] = True
                candidate.score = round(max(0.6, candidate.score - 0.1), 3)
                events.append(
                    StreamEvent("end", candidate, round(time_s, 2), creative_id=candidate.evidence.get("creative"))
                )
                del self._active[creative_index]
        return events

    def _prune(self, time_s: float) -> None:
        if len(self._votes) < 2000:
            return
        stale = [
            key
            for key, state in self._votes.items()
            if not state.opened and time_s - state.last_seen_s > 30.0
        ]
        for key in stale:
            del self._votes[key]

    def finish(self, time_s: float) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for creative_index, candidate in list(self._active.items()):
            if time_s < candidate.end_s:
                candidate.end_s = round(max(candidate.start_s + 1.0, time_s), 2)
                candidate.evidence["truncated"] = True
            events.append(
                StreamEvent("end", candidate, round(time_s, 2), creative_id=candidate.evidence.get("creative"))
            )
            del self._active[creative_index]
        return events


class StreamingDetector:
    """Feed 16 kHz mono PCM chunks in playback order; get ad events out."""

    def __init__(
        self,
        bank: Optional[AdBank] = None,
        config: Optional[StreamConfig] = None,
        episode: Optional[Episode] = None,
        duration_hint_s: Optional[float] = None,
    ) -> None:
        self.cfg = config or StreamConfig()
        self.episode = episode
        self.duration_hint_s = duration_hint_s
        self.features = FeatureExtractor(retain=False)
        self.decimator = Decimator()
        self.fingerprint = FingerprintExtractor()
        self.rolling = _RollingFeatures(self.cfg.feature_history_s, 0.01)
        self.bank = bank
        self.matcher = OnlineMatcher(bank, self.cfg) if bank is not None and len(bank) else None
        self.audio_time_s = 0.0
        self.events: list[StreamEvent] = []
        self.closed: list[Candidate] = []
        self._segment_start_s = 0.0
        self._segment_start_boundary = Boundary(0.0, 0.25, "episode_start")
        self._boundary_cursor = 0
        self._last_boundary_frame = None
        self._acoustic: Optional[Candidate] = None
        self._acoustic_start_s = 0.0
        self._next_eval_s = 0.0
        self._bank_active: list[Candidate] = []

    # -- audio input -----------------------------------------------------
    def push(self, pcm: np.ndarray) -> list[StreamEvent]:
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            return []
        self.audio_time_s += len(pcm) / float(SAMPLE_RATE)
        rows = self.features.push_rows(pcm)
        self.rolling.append(rows["fine"])
        self.fingerprint.push(self.decimator.push(pcm))
        hashes, frames = self.fingerprint.drain()

        output: list[StreamEvent] = []
        if self.matcher is not None:
            output.extend(self._handle_bank(self.matcher.query(hashes, frames, self.audio_time_s)))

        boundaries = self._emit_boundaries(final=False)
        for boundary in boundaries:
            output.extend(self._on_boundary(boundary))
        output.extend(self._maybe_open_acoustic(self.audio_time_s))
        self.events.extend(output)
        return output

    def finish(self) -> list[StreamEvent]:
        output: list[StreamEvent] = []
        for boundary in self._emit_boundaries(final=True):
            output.extend(self._on_boundary(boundary))
        output.extend(self._finalize_segment(self.audio_time_s, None))
        if self.matcher is not None:
            output.extend(self._handle_bank(self.matcher.finish(self.audio_time_s)))
        output.extend(self._resolve_overlaps())
        self.events.extend(output)
        return output

    # -- boundaries ------------------------------------------------------
    def _emit_boundaries(self, final: bool) -> list[Boundary]:
        arrays, base, total = self.rolling.materialize()
        if len(arrays["loudness_db"]) < 40 or total <= base:
            return []
        grid = self.rolling.grid_s
        lookahead = int(round(self.cfg.boundary_lookahead_s / grid))
        half = 30
        last_center = total - 1 if final else total - lookahead - 1
        start_center = max(self._boundary_cursor, base + half, 1)
        if last_center < start_center:
            return []
        loud = arrays["loudness_db"]
        flat = arrays["flatness"]
        flux = arrays["flux"]
        mfcc = arrays["mfcc"]
        stride = max(1, len(loud) // 2000)
        sampled_mfcc = mfcc[::stride]
        normalized_mfcc = np.empty_like(mfcc)
        for dim in range(mfcc.shape[1]):
            values = sampled_mfcc[:, dim]
            median = float(np.median(values))
            spread = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-6)
            normalized_mfcc[:, dim] = (mfcc[:, dim] - median) / spread
        flux_high = max(float(np.percentile(flux[::stride], 95)), 1e-6)
        loud_floor = float(np.percentile(loud[::stride], 10))
        emitted: list[Boundary] = []
        score_cache: dict[int, tuple[float, float, float]] = {}

        def score_at(center: int) -> tuple[float, float, float]:
            cached = score_cache.get(center)
            if cached is not None:
                return cached
            first = max(0, center - base - 300)
            mid = max(1, center - base)
            post = min(len(loud), center - base + int(round(1.0 / grid)))
            if mid <= first or post <= mid:
                score_cache[center] = (0.0, 0.0, 0.0)
                return score_cache[center]
            pre_loud = float(np.mean(loud[first:mid]))
            post_loud = float(np.mean(loud[mid:post]))
            level = abs(post_loud - pre_loud)
            pre_timbre = np.mean(normalized_mfcc[first:mid], axis=0)
            post_timbre = np.mean(normalized_mfcc[mid:post], axis=0)
            timbre = float(np.sqrt(np.mean((post_timbre - pre_timbre) ** 2)))
            local_flux = float(
                np.mean(flux[max(0, center - base - 50):min(len(flux), center - base + 50)])
            )
            score = (
                0.40 * min(level / 6.0, 1.0)
                + 0.30 * min(timbre / 2.0, 1.0)
                + 0.30 * min(local_flux / flux_high, 1.0)
            )
            score_cache[center] = (score, level, timbre)
            return score_cache[center]

        minimum_gap = int(round(self.cfg.boundary_min_gap_s / grid))
        for center in range(start_center, last_center + 1):
            score, level, timbre = score_at(center)
            best = max(
                score_at(other)[0]
                for other in range(center - half, center + half + 1)
                if other >= base + 1 and other <= total - 1
            )
            if score + 1e-9 < best or score < self.cfg.boundary_score:
                continue
            if self._last_boundary_frame is not None and center - self._last_boundary_frame < minimum_gap:
                continue
            post_loud = float(
                np.mean(
                    loud[
                        max(0, center - base):
                        min(len(loud), center - base + int(round(1.0 / grid)))
                    ]
                )
            )
            kind = "silence" if post_loud < loud_floor + 6.0 else "cut"
            emitted.append(
                Boundary(
                    time_s=round(center * grid, 2),
                    score=round(min(score, 1.0), 3),
                    kind=kind,
                    level_db=round(level, 2),
                    timbre=round(min(timbre / 2.0, 1.0), 3),
                )
            )
            self._last_boundary_frame = center
        self._boundary_cursor = last_center + 1
        return emitted

    # -- acoustic segments ----------------------------------------------
    def _on_boundary(self, boundary: Boundary) -> list[StreamEvent]:
        output: list[StreamEvent] = []
        output.extend(self._finalize_segment(boundary.time_s, boundary))
        self._segment_start_s = boundary.time_s
        self._segment_start_boundary = boundary
        self._next_eval_s = boundary.time_s + self.cfg.provisional_s
        return output

    def _finalize_segment(
        self,
        end_s: float,
        end_boundary: Optional[Boundary],
    ) -> list[StreamEvent]:
        start = self._segment_start_s
        duration = end_s - start
        output: list[StreamEvent] = []
        if duration < self.cfg.min_candidate_s or duration > self.cfg.max_candidate_s:
            if self._acoustic is not None and self._acoustic_start_s == start:
                candidate = self._acoustic
                candidate.end_s = round(end_s, 2)
                candidate.evidence["provisional"] = False
                output.append(StreamEvent("cancel", candidate, round(self.audio_time_s, 2)))
                self._acoustic = None
            return output

        stats = self._segment_stats(start, end_s, self._segment_start_boundary, end_boundary)
        score = stats["score"]
        if self._acoustic is not None and self._acoustic_start_s == start:
            candidate = self._acoustic
            candidate.end_s = round(end_s, 2)
            candidate.score = round(max(score, candidate.score), 3)
            candidate.evidence.update(stats["evidence"])
            candidate.evidence["provisional"] = False
            candidate.evidence["detected_at_s"] = round(self.audio_time_s, 2)
            if score >= self.cfg.acoustic_close_score and stats["evidence"]["ad_evidence"]:
                output.append(StreamEvent("end", candidate, round(self.audio_time_s, 2)))
                self._remember(candidate)
            else:
                output.append(StreamEvent("cancel", candidate, round(self.audio_time_s, 2)))
            self._acoustic = None
        elif (
            self.cfg.report_late
            and score >= self.cfg.acoustic_close_score
            and stats["evidence"]["ad_evidence"]
            and not self._bank_overlaps(start, end_s)
        ):
            candidate = Candidate(
                start_s=round(start, 2),
                end_s=round(end_s, 2),
                score=round(score, 3),
                tier="inserted",
                evidence={**stats["evidence"], "provisional": False, "late": True, "detected_at_s": round(self.audio_time_s, 2)},
            )
            output.append(StreamEvent("start", candidate, round(self.audio_time_s, 2)))
            output.append(StreamEvent("end", candidate, round(self.audio_time_s, 2)))
            self._remember(candidate)
        return output

    def _maybe_open_acoustic(self, time_s: float) -> list[StreamEvent]:
        if self._acoustic is not None:
            self._acoustic.end_s = round(max(self._acoustic.end_s, time_s), 2)
            return []
        start = self._segment_start_s
        if time_s - start < self.cfg.provisional_s or time_s < self._next_eval_s:
            return []
        if start < 1.0 and not self.cfg.pre_roll_open:
            return []
        if self._bank_overlaps(start, time_s):
            return []
        stats = self._segment_stats(start, time_s, self._segment_start_boundary, None, provisional=True)
        threshold = (
            self.cfg.pre_roll_score if start < 1.0 else self.cfg.acoustic_open_score
        )
        if stats["score"] < threshold or not stats["evidence"]["ad_evidence"]:
            self._next_eval_s = time_s + 1.5
            return []
        if stats["evidence"]["host_like"]:
            self._next_eval_s = time_s + 1.5
            return []
        candidate = Candidate(
            start_s=round(start, 2),
            end_s=round(time_s, 2),
            score=round(stats["score"], 3),
            tier="inserted",
            evidence={
                **stats["evidence"],
                "source": "acoustic",
                "provisional": True,
                "detected_at_s": round(time_s, 2),
                "latency_s": round(max(0.0, time_s - start), 2),
            },
        )
        self._acoustic = candidate
        self._acoustic_start_s = start
        return [StreamEvent("start", candidate, round(time_s, 2))]

    def _segment_stats(
        self,
        start_s: float,
        end_s: float,
        start_boundary: Optional[Boundary],
        end_boundary: Optional[Boundary],
        provisional: bool = False,
    ) -> dict:
        arrays, base, total = self.rolling.materialize()
        grid = self.rolling.grid_s
        loud = arrays["loudness_db"]
        flat = arrays["flatness"]
        if len(loud) < 5:
            return {"score": 0.0, "evidence": {"ad_evidence": False, "host_like": True}}
        first = int(np.clip(round(start_s / grid) - base, 0, len(loud) - 1))
        last = int(np.clip(round(end_s / grid) - base, first + 1, len(loud)))
        segment_loud = loud[first:last]
        segment_flat = flat[first:last]
        baseline_loud = float(np.median(loud)) if len(loud) else 0.0
        segment_loud_median = float(np.median(segment_loud)) if len(segment_loud) else baseline_loud
        loudness_dev = abs(segment_loud_median - baseline_loud)
        segment_flat_median = float(np.median(segment_flat)) if len(segment_flat) else 0.0
        tonality_rank = float(np.mean(flat < segment_flat_median)) if len(flat) else 0.5
        musicness = float(np.clip((0.5 - tonality_rank) / 0.5, 0.0, 1.0))
        boundary_scores = [
            boundary.score
            for boundary in (start_boundary, end_boundary)
            if boundary is not None and boundary.kind not in {"episode_start", "episode_end"}
        ]
        boundary_strength = float(np.mean(boundary_scores)) if boundary_scores else 0.25
        timbre_values = [
            boundary.timbre
            for boundary in (start_boundary, end_boundary)
            if boundary is not None and boundary.kind not in {"episode_start", "episode_end"}
        ]
        timbre_strength = float(np.mean(timbre_values)) if timbre_values else 0.0
        timbre_min = min(timbre_values) if timbre_values else 0.0
        duration = max(0.0, end_s - start_s)
        if start_s < 120.0:
            position = 1.0
        elif self.duration_hint_s and end_s > self.duration_hint_s - 150.0:
            position = 0.9
        else:
            position = 0.35
        duration_prior = 0.8 if 15.0 <= duration <= 60.0 else (0.5 if duration > 60.0 else 0.6)
        ad_evidence = musicness >= 0.3 or loudness_dev >= 5.0
        host_like = musicness < 0.25 and loudness_dev < 3.0 and timbre_min < 0.7
        score = (
            0.34 * boundary_strength
            + 0.20 * timbre_strength
            + 0.14 * musicness
            + 0.12 * min(loudness_dev / 6.0, 1.0)
            + 0.10 * position
            + 0.10 * duration_prior
        )
        if boundary_strength < 0.35:
            score *= 0.55
        if not ad_evidence:
            score -= 0.10
        if host_like:
            score -= 0.22
        if provisional and duration < 4.0:
            score -= 0.05
        evidence = {
            "kind": "silence" if (end_boundary and end_boundary.kind == "silence") else "cut",
            "musicness": round(musicness, 3),
            "loudness_dev_db": round(loudness_dev, 2),
            "boundary_start": round(start_boundary.score, 3) if start_boundary else 0.0,
            "boundary_end": round(end_boundary.score, 3) if end_boundary else 0.0,
            "boundary_strength": round(boundary_strength, 3),
            "timbre_change": round(timbre_strength, 3),
            "timbre_min": round(timbre_min, 3),
            "position": position,
            "ad_evidence": bool(ad_evidence),
            "host_like": bool(host_like),
        }
        return {"score": round(float(np.clip(score, 0.0, 0.95)), 3), "evidence": evidence}

    # -- bank results ----------------------------------------------------
    def _handle_bank(self, events: list[StreamEvent]) -> list[StreamEvent]:
        output = list(events)
        for event in events:
            candidate = event.candidate
            if event.action == "start":
                if self._acoustic is not None and self._overlap(
                    self._acoustic.start_s, self._acoustic.end_s, candidate.start_s, candidate.end_s
                ) >= self.cfg.dedupe_overlap:
                    output.append(StreamEvent("cancel", self._acoustic, round(self.audio_time_s, 2)))
                    self._acoustic = None
                self._bank_active.append(candidate)
            elif event.action == "end":
                if candidate in self._bank_active:
                    self._bank_active.remove(candidate)
                self._remember(candidate)
        return output

    def _resolve_overlaps(self) -> list[StreamEvent]:
        output: list[StreamEvent] = []
        for acoustic in [candidate for candidate in self.closed if candidate.tier == "inserted"]:
            match = next(
                (
                    repeated
                    for repeated in self.closed
                    if repeated.tier == "repeated"
                    and self._overlap(acoustic.start_s, acoustic.end_s, repeated.start_s, repeated.end_s)
                    >= self.cfg.dedupe_overlap
                ),
                None,
            )
            if match is not None:
                self.closed.remove(acoustic)
                output.append(StreamEvent("cancel", acoustic, round(self.audio_time_s, 2)))
        repeated = [candidate for candidate in self.closed if candidate.tier == "repeated"]
        for first in repeated:
            for second in repeated:
                if first is second or first not in self.closed or second not in self.closed:
                    continue
                if self._overlap(first.start_s, first.end_s, second.start_s, second.end_s) >= self.cfg.dedupe_overlap:
                    loser = first if first.score < second.score else second
                    self.closed.remove(loser)
                    output.append(StreamEvent("cancel", loser, round(self.audio_time_s, 2)))
        return output

    def _remember(self, candidate: Candidate) -> None:
        if not any(
            existing.start_s == candidate.start_s and existing.end_s == candidate.end_s
            for existing in self.closed
        ):
            self.closed.append(candidate)

    def _bank_overlaps(self, start_s: float, end_s: float) -> bool:
        for candidate in self._bank_active:
            if self._overlap(start_s, end_s, candidate.start_s, candidate.end_s) >= self.cfg.dedupe_overlap:
                return True
        for candidate in self.closed:
            if candidate.tier == "repeated" and self._overlap(
                start_s, end_s, candidate.start_s, candidate.end_s
            ) >= self.cfg.dedupe_overlap:
                return True
        return False

    @staticmethod
    def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
        overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
        smallest = max(1e-6, min(max(1e-6, end_a - start_a), max(1e-6, end_b - start_b)))
        return overlap / smallest
