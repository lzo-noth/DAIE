#!/usr/bin/env python3
"""Blind test for one recording.

The detector sees only ``--audio``. Ground truth is loaded *after* the
timeline is produced, and only to score it. This enforces the single-audio
runtime contract during evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.evaluation import evaluate_events
from dai_detect.single_episode import SingleConfig, analyze_single, write_timeline

ACCEPTANCE = {
    "event_precision": 0.90,
    "event_recall": 0.90,
    "boundary_p95_s": 0.50,
}


def _load_ground_truth(path: Path) -> tuple[list[dict], float | None, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        ads = payload
        duration = None
        identity: dict = {}
    else:
        ads = payload.get("ads") or []
        duration = payload.get("duration_s")
        audio_meta = payload.get("audio") or {}
        identity = {
            "source_sha256": str(
                payload.get("source_sha256") or audio_meta.get("source_sha256") or ""
            ),
            "audio_id": str(payload.get("audio_id") or audio_meta.get("id") or ""),
            "audio_id_kind": str(
                payload.get("audio_id_kind") or audio_meta.get("id_kind") or ""
            ),
        }
    rows = []
    for ad in ads:
        rows.append(
            {
                "start_s": float(ad["start_s"] if "start_s" in ad else ad["start"]),
                "end_s": float(ad["end_s"] if "end_s" in ad else ad["end"]),
            }
        )
    return rows, float(duration) if duration else None, identity


def _identity_checks(
    timeline_audio: dict,
    ground_truth: dict,
    input_is_local: bool,
) -> tuple[list[str], list[str]]:
    """Return ``(missing_fields, value_mismatches)``.

    Missing identity is a different failure from a mismatched identity: the
    former means the label cannot be proven to belong to any audio, the latter
    means it belongs to a different one. Neither is silently accepted.
    """
    missing: list[str] = []
    mismatches: list[str] = []
    has_identity = bool(ground_truth.get("source_sha256") or ground_truth.get("audio_id"))
    if not has_identity:
        required = "audio_id 与 audio_id_kind"
        if input_is_local:
            required += "，以及 source_sha256"
        missing.append(f"真值缺少音频身份：必须提供 {required}")
        return missing, mismatches

    if not ground_truth.get("audio_id"):
        missing.append("真值缺少 audio_id（解码 PCM 内容指纹）")
    if not ground_truth.get("audio_id_kind"):
        missing.append("真值缺少 audio_id_kind")
    if input_is_local and not ground_truth.get("source_sha256"):
        missing.append("本地输入音频的真值缺少 source_sha256")

    gt_source = ground_truth.get("source_sha256") or ""
    input_source = timeline_audio.get("source_sha256") or ""
    if gt_source and input_source and gt_source != input_source:
        mismatches.append("source_sha256 不匹配：输入音频与标注音频不是同一份")

    gt_audio_id = ground_truth.get("audio_id") or ""
    input_audio_id = timeline_audio.get("id") or ""
    if gt_audio_id:
        if not input_audio_id:
            mismatches.append("时间轴缺少 audio_id，无法校验音频内容")
        elif gt_audio_id != input_audio_id:
            mismatches.append("audio_id 不匹配：输入音频是另一个 DAI 变体")
    gt_kind = ground_truth.get("audio_id_kind") or ""
    input_kind = timeline_audio.get("id_kind") or ""
    if gt_kind and input_kind and gt_kind != input_kind:
        mismatches.append(f"audio_id_kind 不一致（真值 {gt_kind}, 输入 {input_kind}）")
    return missing, mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="唯一输入：一集完整音频")
    parser.add_argument("--model", type=Path, help="广告分类模型 JSON")
    parser.add_argument("--boundary-model", type=Path, help="边界模型 JSON")
    parser.add_argument("--ground-truth", type=Path, help="人工真值；检测完成后才读取")
    parser.add_argument("--out", type=Path, help="保存检测时间轴 JSON")
    parser.add_argument("--metrics", type=Path, help="保存评测指标 JSON")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-ad", type=float, default=6.0)
    parser.add_argument(
        "--allow-audio-mismatch",
        action="store_true",
        help="真值音频身份不匹配时仍评分（默认拒绝，避免评测另一个 DAI 变体）",
    )
    parser.add_argument(
        "--allow-unverified-ground-truth",
        action="store_true",
        help="真值缺少音频身份字段时仍评分（默认拒绝，仅用于调试）",
    )
    parser.add_argument(
        "--require-models",
        action="store_true",
        help="要求分类模型和边界模型都存在且 data_version 一致",
    )
    parser.add_argument(
        "--require-calibrated-models",
        action="store_true",
        help="要求两个模型都通过 deployment_calibration 校准验收",
    )
    parser.add_argument("--proxy")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = SingleConfig(
        min_ad_s=args.min_ad,
        enter_threshold=args.threshold,
        model_path=str(args.model) if args.model else None,
        boundary_model_path=str(args.boundary_model) if args.boundary_model else None,
        proxy=args.proxy,
        require_models=args.require_models,
        require_calibrated_models=args.require_calibrated_models,
    )
    progress = None if args.quiet else lambda message: print(message, file=sys.stderr, flush=True)
    timeline = analyze_single(args.audio, cfg, progress=progress)
    if args.out:
        write_timeline(args.out, timeline)

    print(json.dumps({"blind_detection": {"duration_s": timeline["duration_s"], "ads": timeline["ads"]}}, ensure_ascii=False))

    if not args.ground_truth:
        return 0

    # Ground truth is read only after the timeline above is fixed.
    ground_truth, gt_duration, gt_identity = _load_ground_truth(args.ground_truth)
    timeline_audio = timeline.get("audio") or {}
    input_is_local = bool(timeline_audio.get("source_sha256"))
    missing, mismatches = _identity_checks(timeline_audio, gt_identity, input_is_local)
    if missing:
        for problem in missing:
            print(f"[identity] {problem}", file=sys.stderr)
        if not args.allow_unverified_ground_truth:
            print(
                "[identity] 拒绝评分：真值必须携带可校验的音频身份；"
                "仅调试时可用 --allow-unverified-ground-truth",
                file=sys.stderr,
            )
            return 3
    if mismatches:
        for problem in mismatches:
            print(f"[identity] {problem}", file=sys.stderr)
        if not args.allow_audio_mismatch:
            print(
                "[identity] 拒绝评分：真值与输入音频不是同一份；"
                "确认无误可用 --allow-audio-mismatch 覆盖",
                file=sys.stderr,
            )
            return 3
    metrics = evaluate_events(
        timeline["ads"],
        ground_truth,
        duration_s=timeline["duration_s"] or gt_duration,
    )
    checks = {
        "event_precision": (
            metrics["event_precision"] is not None
            and metrics["event_precision"] >= ACCEPTANCE["event_precision"]
        ),
        "event_recall": (
            metrics["event_recall"] is not None
            and metrics["event_recall"] >= ACCEPTANCE["event_recall"]
        ),
        "boundary_p95_s": (
            metrics["boundary_p95_s"] is not None
            and metrics["boundary_p95_s"] <= ACCEPTANCE["boundary_p95_s"]
        ),
    }
    passed = all(checks.values())
    report = {
        "audio": args.audio,
        "model": timeline["model"],
        "metrics": {key: metrics[key] for key in (
            "event_precision",
            "event_recall",
            "event_f1",
            "boundary_p95_s",
            "boundary_mean_s",
            "false_positives_per_hour",
            "matched",
            "false_positives",
            "false_negatives",
            "iou_threshold",
        )},
        "acceptance": ACCEPTANCE,
        "checks": checks,
        "passed": passed,
        "performance_targets_met": passed,
        "performance_targets_source": "independent_blind_test",
        "matches": metrics["matches"],
        "unmatched_predictions": metrics["unmatched_predictions"],
        "unmatched_ground_truth": metrics["unmatched_ground_truth"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
