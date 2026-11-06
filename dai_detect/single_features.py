"""Multiscale window features for single-recording ad detection.

The detector scores one 1.0 s window every 0.5 s. Each window aggregates the
10 ms acoustic rows from :mod:`features` and the 10 ms forensic rows from
:mod:`forensic`, then expresses them relative to the episode baseline. That
relative encoding is what lets the same model work on different shows with
different microphones, loudness targets and room acoustics.

Feature names are stable and ordered; trained models store this exact list so
runtime cannot silently misalign columns.
"""

from __future__ import annotations

import numpy as np

WINDOW_HOP_S = 0.5
WINDOW_SPAN_S = 1.0

_BAND_NAMES = [f"band_{index:02d}" for index in range(16)]
WINDOW_FEATURES: list[str] = [
    "loud_mean",
    "loud_std",
    "loud_p10",
    "loud_dev",
    "musicness",
    "flux_mean",
    "flux_p95",
    "mfcc_0",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_dev",
    "mfcc_delta_prev",
    "hf_mean",
    "hf_p10",
    "hf_dev",
    "rolloff_mean",
    "entropy_mean",
    "entropy_std",
    "specflat_mean",
    "corr_mean",
    "corr_std",
    "corr_min",
    "corr_dev",
    "width_mean",
    "width_dev",
    "clip_mean",
    "phase_mean",
    "floor_db",
    "floor_dev",
    "band_dist",
    *_BAND_NAMES,
    "d_loud_prev",
    "d_loud_next",
    "d_hf_prev",
    "d_hf_next",
    "d_corr_prev",
    "d_corr_next",
    "d_width_prev",
    "d_width_next",
    "d_entropy_prev",
    "d_entropy_next",
]


def _percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def _median(values: np.ndarray, axis=None, default: float = 0.0):
    if values.size == 0:
        return default
    return np.median(values, axis=axis)


def build_windows(
    mono: dict,
    forensic: dict,
    hop_s: float = WINDOW_HOP_S,
    span_s: float = WINDOW_SPAN_S,
) -> dict:
    """Build the fixed-length feature matrix for one episode.

    ``mono`` is the payload of ``FeatureExtractor.finish()`` and ``forensic``
    the payload of ``StereoForensicExtractor.finish()``.
    """
    grid_m = float(mono.get("fine_grid_s") or 0.01)
    grid_f = float(forensic.get("frame_grid_s") or 0.01)
    duration = min(
        float(mono.get("duration_s") or 0.0),
        float(forensic.get("duration_s") or 0.0),
    )
    loud = np.asarray(mono["fine_loudness_db"], dtype=np.float64)
    flat = np.asarray(mono["fine_flatness"], dtype=np.float64)
    flux = np.asarray(mono["fine_flux"], dtype=np.float64)
    mfcc = np.asarray(mono["fine_mfcc"], dtype=np.float64)
    if mfcc.ndim != 2 or mfcc.shape[1] != 8:
        raise ValueError("MFCC 特征维度异常，期望 8 维")

    f_loud = np.asarray(forensic["loudness_db"], dtype=np.float64)
    hf = np.asarray(forensic["hf_ratio"], dtype=np.float64)
    rolloff = np.asarray(forensic["rolloff_hz"], dtype=np.float64)
    entropy = np.asarray(forensic["entropy"], dtype=np.float64)
    specflat = np.asarray(forensic["flatness"], dtype=np.float64)
    corr = np.asarray(forensic["stereo_corr"], dtype=np.float64)
    width = np.asarray(forensic["side_mid_db"], dtype=np.float64)
    clip = np.asarray(forensic["clip_ratio"], dtype=np.float64)
    phase = np.asarray(forensic["phase_jump"], dtype=np.float64)
    bands = np.asarray(forensic["bands"], dtype=np.float64)

    if duration <= 0 or len(loud) == 0 or len(f_loud) == 0 or len(flat) == 0:
        return {
            "times": np.zeros(0, dtype=np.float64),
            "features": np.zeros((0, len(WINDOW_FEATURES)), dtype=np.float32),
            "names": list(WINDOW_FEATURES),
            "duration_s": duration,
            "hop_s": hop_s,
            "span_s": span_s,
        }

    loud_base = _median(loud)
    flat_base = max(_median(flat), 1e-6)
    mfcc_base = _median(mfcc, axis=0)
    band_base = _median(bands, axis=0)
    hf_base = _median(hf)
    corr_base = _median(corr)
    width_base = _median(width)
    entropy_base = _median(entropy)
    floor_base = _percentile(loud, 10)

    span = min(span_s, duration)
    if duration <= span:
        count = 1
    else:
        count = int(np.floor((duration - span) / hop_s)) + 1
    count = max(1, count)

    matrix = np.zeros((count, len(WINDOW_FEATURES)), dtype=np.float32)
    times = np.zeros(count, dtype=np.float64)
    mono_frame_count = len(loud)
    forensic_frame_count = len(f_loud)
    band_count = min(bands.shape[1], len(_BAND_NAMES))

    for index in range(count):
        start = index * hop_s
        end = start + span
        times[index] = start + span / 2.0

        m0 = int(np.clip(round(start / grid_m), 0, max(mono_frame_count - 1, 0)))
        m1 = int(np.clip(round(end / grid_m), m0 + 1, mono_frame_count))
        f0 = int(np.clip(round(start / grid_f), 0, max(forensic_frame_count - 1, 0)))
        f1 = int(np.clip(round(end / grid_f), f0 + 1, forensic_frame_count))

        window_loud = loud[m0:m1]
        window_flat = flat[m0:m1]
        window_flux = flux[m0:m1]
        window_mfcc = mfcc[m0:m1]
        window_hf = hf[f0:f1]
        window_corr = corr[f0:f1]
        window_width = width[f0:f1]
        window_entropy = entropy[f0:f1]
        window_bands = bands[f0:f1]

        mfcc_mean = window_mfcc.mean(axis=0)
        band_median = np.median(window_bands, axis=0)
        loud_mean = float(window_loud.mean())
        floor_db = _percentile(window_loud, 10)
        flat_median = float(np.median(window_flat))
        musicness = float(np.clip((flat_base - flat_median) / flat_base, 0.0, 1.0))

        row = [
            loud_mean,
            float(window_loud.std()),
            floor_db,
            loud_mean - loud_base,
            musicness,
            float(window_flux.mean()),
            _percentile(window_flux, 95),
            *[float(value) for value in mfcc_mean],
            float(np.sqrt(np.mean((mfcc_mean - mfcc_base) ** 2))),
            0.0,  # mfcc_delta_prev, filled below
            float(window_hf.mean()),
            _percentile(window_hf, 10),
            float(window_hf.mean() - hf_base),
            float(rolloff[f0:f1].mean()),
            float(window_entropy.mean()),
            float(window_entropy.std()),
            float(specflat[f0:f1].mean()),
            float(window_corr.mean()),
            float(window_corr.std()),
            float(window_corr.min()),
            float(window_corr.mean() - corr_base),
            float(window_width.mean()),
            float(window_width.mean() - width_base),
            float(clip[f0:f1].mean()),
            float(phase[f0:f1].mean()),
            floor_db,
            float(floor_db - floor_base),
            float(np.sqrt(np.mean((band_median - band_base) ** 2))),
            *[float(value) for value in band_median[:band_count]],
            *[0.0] * (len(_BAND_NAMES) - band_count),
            *[0.0] * 10,  # local deltas, filled below
        ]
        matrix[index] = np.nan_to_num(np.asarray(row, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    columns = {name: index for index, name in enumerate(WINDOW_FEATURES)}
    for index in range(count):
        m0 = int(np.clip(round(index * hop_s / grid_m), 0, max(mono_frame_count - 1, 0)))
        m1 = int(np.clip(round((index * hop_s + span) / grid_m), m0 + 1, mono_frame_count))
        current_mfcc = mfcc[m0:m1].mean(axis=0)
        previous_mfcc = mfcc[max(0, m0 - int(span / grid_m)):m0].mean(axis=0) if m0 > 0 else current_mfcc
        matrix[index, columns["mfcc_delta_prev"]] = float(
            np.sqrt(np.mean((current_mfcc - previous_mfcc) ** 2))
        )

    pairs = [
        ("d_loud_prev", "loud_mean"),
        ("d_hf_prev", "hf_mean"),
        ("d_corr_prev", "corr_mean"),
        ("d_width_prev", "width_mean"),
        ("d_entropy_prev", "entropy_mean"),
        ("d_loud_next", "loud_mean"),
        ("d_hf_next", "hf_mean"),
        ("d_corr_next", "corr_mean"),
        ("d_width_next", "width_mean"),
        ("d_entropy_next", "entropy_mean"),
    ]
    for target, source in pairs:
        column = columns[source]
        reference = matrix[:, column]
        if target.endswith("_prev"):
            deltas = np.diff(reference, prepend=reference[0])
        else:
            deltas = np.diff(reference, append=reference[-1])
            deltas = -deltas
        matrix[:, columns[target]] = deltas

    return {
        "times": times,
        "features": matrix,
        "names": list(WINDOW_FEATURES),
        "duration_s": duration,
        "hop_s": hop_s,
        "span_s": span,
        "baseline": {
            "loudness_db": loud_base,
            "flatness": flat_base,
            "hf_ratio": hf_base,
            "stereo_corr": corr_base,
            "side_mid_db": width_base,
            "entropy": entropy_base,
            "noise_floor_db": floor_base,
        },
    }


def window_labels(
    windows: dict,
    ads: list[dict],
    positive_coverage: float = 0.75,
    negative_clearance: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return labelled window indices and labels from ground-truth ads.

    A window is positive when at least ``positive_coverage`` of it overlaps an
    ad; negative when it touches no ad and is at least ``negative_clearance``
    seconds away from every ad edge. Ambiguous windows are dropped.
    """
    span = float(windows["span_s"])
    times = np.asarray(windows["times"], dtype=np.float64)
    indices: list[int] = []
    labels: list[int] = []
    intervals = [(float(ad["start_s"]), float(ad["end_s"])) for ad in ads]
    for index, center in enumerate(times):
        start, end = center - span / 2.0, center + span / 2.0
        best = 0.0
        nearest = float("inf")
        for ad_start, ad_end in intervals:
            overlap = max(0.0, min(end, ad_end) - max(start, ad_start))
            best = max(best, overlap / span)
            gap = max(ad_start - end, start - ad_end, 0.0)
            nearest = min(nearest, gap)
        if best >= positive_coverage:
            indices.append(index)
            labels.append(1)
        elif best <= 1e-6 and nearest >= negative_clearance:
            indices.append(index)
            labels.append(0)
    return np.asarray(indices, dtype=np.int64), np.asarray(labels, dtype=np.float64)


def select_features(features: np.ndarray, names: list[str], wanted: list[str]) -> np.ndarray:
    """Reorder/reject columns so a stored model receives exactly its features."""
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in wanted if name not in positions]
    if missing:
        raise ValueError(f"特征缺失: {', '.join(missing[:5])}")
    return np.asarray(features)[:, [positions[name] for name in wanted]]
