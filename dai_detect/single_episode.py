"""Single-recording blind DAI ad detection.

Runtime contract: the only input is one audio file (local path or URL). No
original audio, no sibling episodes, no transcript, no per-episode reference
material. Everything below is computed from that one decode pass.

Pipeline: 48 kHz stereo forensic decode -> 16 kHz mono acoustic decode ->
1.0 s windows every 0.5 s -> window ad probabilities -> temporal decode ->
boundary refinement -> timeline JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .ad_classifier import HeuristicAdModel, load_ad_model
from .audio import DownmixResampler, iter_pcm_stereo, source_sha256
from .boundary_model import load_boundary_model, score_boundaries
from .features import FeatureExtractor
from .forensic import StereoForensicExtractor
from .refinement import refine_events
from .single_features import build_windows
from .temporal_decoder import decode_windows

TIMELINE_VERSION = 1


@dataclass(frozen=True)
class SingleConfig:
    hop_s: float = 0.5
    span_s: float = 1.0
    min_ad_s: float = 6.0
    max_ad_s: float = 600.0
    max_gap_s: float = 2.0
    enter_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    search_s: float = 2.5
    refine: bool = True
    min_confidence: float = 0.0
    model_path: Optional[str] = None
    boundary_model_path: Optional[str] = None
    proxy: Optional[str] = None


def extract_features(
    src: str,
    proxy: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[dict, dict]:
    """Decode and extract both views from one stream."""
    collector = FeatureExtractor()
    forensic = StereoForensicExtractor()
    downmix = DownmixResampler()
    seconds = 0.0
    next_report = 60.0
    for block in iter_pcm_stereo(src, block_s=2.0, proxy=proxy):
        forensic.push(block)
        mono = downmix.push(block.mean(axis=1))
        collector.push(mono)
        seconds += len(block) / 48000.0
        if progress and seconds >= next_report:
            progress(f"  已处理 {int(seconds // 60)} 分钟")
            next_report += 60.0
    mono_payload = collector.finish()
    mono_payload["audio_id_kind"] = "downmix_48k_to_16k"
    return mono_payload, forensic.finish()


def _slot(start: float, end: float, duration: float) -> str:
    if start < 120.0:
        return "pre_roll"
    if end > duration - 150.0:
        return "post_roll"
    return "mid_roll"


def analyze_single(
    src: str,
    cfg: SingleConfig | None = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    cfg = cfg or SingleConfig()
    if progress:
        progress(f"解码: {src}")
    mono, forensic = extract_features(src, proxy=cfg.proxy, progress=progress)
    duration = min(float(mono["duration_s"]), float(forensic["duration_s"]))

    windows = build_windows(mono, forensic, hop_s=cfg.hop_s, span_s=cfg.span_s)
    model = load_ad_model(cfg.model_path)
    if progress and isinstance(model, HeuristicAdModel):
        progress("警告: 未找到训练模型，使用启发式兜底（指标不可作为验收依据）")
    probabilities = model.predict(windows["features"], windows["names"])

    threshold = float(getattr(model, "threshold", 0.5))
    enter = cfg.enter_threshold if cfg.enter_threshold is not None else min(
        0.75, max(0.45, threshold + 0.05)
    )
    exit_threshold = cfg.exit_threshold if cfg.exit_threshold is not None else max(
        0.20, enter - 0.15
    )

    decoded = decode_windows(
        windows["times"],
        probabilities,
        hop_s=cfg.hop_s,
        span_s=cfg.span_s,
        min_ad_s=cfg.min_ad_s,
        max_gap_s=cfg.max_gap_s,
        max_ad_s=cfg.max_ad_s,
        enter_threshold=enter,
        exit_threshold=exit_threshold,
    )

    if cfg.refine:
        boundary_model = load_boundary_model(cfg.boundary_model_path)
        change = score_boundaries(mono, forensic, boundary_model)
        events = refine_events(decoded, change, search_s=cfg.search_s, min_duration=cfg.min_ad_s)
    else:
        events = [
            {
                "start": round(event.start, 2),
                "end": round(event.end, 2),
                "start_s": round(event.start, 2),
                "end_s": round(event.end, 2),
                "duration_s": round(event.duration, 2),
                "confidence": round(min(0.99, event.probability), 3),
                "evidence": {
                    "mean_probability": round(event.probability, 3),
                    "peak_probability": round(event.peak, 3),
                    "refined": False,
                },
            }
            for event in decoded
        ]

    ads = []
    for event in events:
        if event["confidence"] < cfg.min_confidence:
            continue
        ads.append(
            {
                **event,
                "type": "third_party_inserted",
                "slot": _slot(event["start"], event["end"], duration),
            }
        )
    ads.sort(key=lambda item: item["start"])

    kind = "learned" if not isinstance(model, HeuristicAdModel) else "heuristic"
    payload = {
        "version": TIMELINE_VERSION,
        "mode": "single_recording_blind",
        "source": src,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "audio": {
            "id": str(mono.get("audio_id", "")),
            "id_kind": str(mono.get("audio_id_kind", "")),
            "source_sha256": source_sha256(src),
        },
        "config": asdict(cfg),
        "model": {
            "classifier": kind,
            "source": getattr(model, "source", ""),
            "version": getattr(model, "model_version", ""),
            "data_version": getattr(model, "data_version", ""),
            "threshold": threshold,
            "enter_threshold": round(enter, 3),
            "exit_threshold": round(exit_threshold, 3),
        },
        "ads": ads,
        "stats": {
            "windows": int(len(windows["times"])),
            "window_hop_s": cfg.hop_s,
            "window_span_s": cfg.span_s,
            "mean_probability": round(float(probabilities.mean()), 4) if len(probabilities) else 0.0,
            "peak_probability": round(float(probabilities.max()), 4) if len(probabilities) else 0.0,
            "decoded_events": len(decoded),
        },
    }
    if progress:
        progress(f"完成: {len(ads)} 个广告区间")
    return payload


def write_timeline(path: str | Path, timeline: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
