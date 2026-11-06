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
from .decode_config import resolve_thresholds
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
    require_models: bool = False
    require_calibrated_models: bool = False


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


def _calibration_problems(
    model: object, boundary_model: Optional[object]
) -> list[str]:
    """Return reasons the model artifacts fail strict calibration acceptance."""
    if isinstance(model, HeuristicAdModel):
        return ["广告分类器回退到了启发式实现，未通过校准验收"]
    if boundary_model is None:
        return ["边界模型缺失，未通过校准验收"]
    problems: list[str] = []
    classifier_calibration = getattr(model, "deployment_calibration", {}) or {}
    boundary_calibration = getattr(boundary_model, "deployment_calibration", {}) or {}
    for label, calibration in (
        ("分类模型", classifier_calibration),
        ("边界模型", boundary_calibration),
    ):
        if calibration.get("calibration_process_valid") is not True:
            problems.append(f"{label}的 calibration_process_valid 不为 true")
        if calibration.get("final_pipeline_valid") is not True:
            problems.append(f"{label}的 final_pipeline_valid 不为 true")
        if calibration.get("converged") is not True:
            problems.append(f"{label}的阈值校准未收敛")
    for key in ("onset_threshold", "offset_threshold"):
        left = classifier_calibration.get(key)
        right = boundary_calibration.get(key)
        if left is None or right is None:
            problems.append(f"分类模型与边界模型缺少一致的 {key}")
        elif abs(float(left) - float(right)) > 1e-6:
            problems.append(
                f"分类模型与边界模型的 {key} 不一致"
                f"（{float(left):.4f} vs {float(right):.4f}）"
            )
    return problems


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
    boundary_model = load_boundary_model(cfg.boundary_model_path)
    if cfg.require_models:
        if isinstance(model, HeuristicAdModel):
            raise ValueError("要求训练模型，但广告分类器回退到了启发式实现")
        if boundary_model is None:
            raise ValueError("要求训练模型，但边界模型缺失")
        if not getattr(model, "data_version", ""):
            raise ValueError("要求训练模型，但分类模型缺少 data_version")
        if not getattr(boundary_model, "data_version", ""):
            raise ValueError("要求训练模型，但边界模型缺少 data_version")
    if cfg.require_calibrated_models:
        problems = _calibration_problems(model, boundary_model)
        if problems:
            raise ValueError("严格校准部署检查失败：" + "；".join(problems))
    elif not isinstance(model, HeuristicAdModel):
        classifier_calibration = getattr(model, "deployment_calibration", {}) or {}
        boundary_calibration = (
            getattr(boundary_model, "deployment_calibration", {}) or {}
            if boundary_model is not None
            else {}
        )
        invalid = [
            calibration
            for calibration in (classifier_calibration, boundary_calibration)
            if calibration and calibration.get("calibration_process_valid") is not True
        ]
        if invalid and progress:
            progress(
                "警告: 模型 deployment_calibration 未通过校准验收；"
                "结果不能作为校准验收依据（严格模式请使用 --require-calibrated-models）"
            )
    if (
        boundary_model is not None
        and not isinstance(model, HeuristicAdModel)
        and getattr(model, "data_version", "")
        and getattr(boundary_model, "data_version", "")
        and model.data_version != boundary_model.data_version
    ):
        raise ValueError(
            "分类模型与边界模型的 data_version 不一致，拒绝混用不同训练批次"
        )
    if progress and isinstance(model, HeuristicAdModel):
        progress("警告: 未找到训练模型，使用启发式兜底（指标不可作为验收依据）")
    probabilities = model.predict(windows["features"], windows["names"])

    threshold = float(getattr(model, "threshold", 0.5))
    enter, exit_threshold = resolve_thresholds(
        threshold, cfg.enter_threshold, cfg.exit_threshold
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
            "git_commit": getattr(model, "git_commit", ""),
            "threshold": threshold,
            "enter_threshold": round(enter, 3),
            "exit_threshold": round(exit_threshold, 3),
            "boundary": {
                "source": getattr(boundary_model, "source", ""),
                "version": getattr(boundary_model, "version", ""),
                "data_version": getattr(boundary_model, "data_version", ""),
                "git_commit": getattr(boundary_model, "git_commit", ""),
                "kind": getattr(boundary_model, "kind", "heuristic" if boundary_model is None else ""),
                "onset_threshold": getattr(boundary_model, "onset_threshold", None),
                "offset_threshold": getattr(boundary_model, "offset_threshold", None),
            },
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
