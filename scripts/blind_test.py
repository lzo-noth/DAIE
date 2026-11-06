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
        identity = {
            "source_sha256": str(payload.get("source_sha256") or ""),
            "audio_id": str(payload.get("audio_id") or ""),
            "audio_id_kind": str(payload.get("audio_id_kind") or ""),
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


def _identity_mismatches(timeline_audio: dict, ground_truth: dict) -> list[str]:
    problems: list[str] = []
    if ground_truth.get("source_sha256"):
        if timeline_audio.get("source_sha256") and timeline_audio["source_sha256"] != ground_truth["source_sha256"]:
            problems.append("source_sha256 不匹配：输入音频与标注音频不是同一份")
        elif not timeline_audio.get("source_sha256") and not ground_truth.get("audio_id"):
            problems.append("输入音频不是本地文件且真值没有 audio_id，无法校验是否为同一份")
    if ground_truth.get("audio_id"):
        if not timeline_audio.get("id"):
            problems.append("真值有 audio_id，但检测时间轴缺少音频内容指纹")
        elif timeline_audio["id"] != ground_truth["audio_id"]:
            problems.append("audio_id 不匹配：输入音频是另一个 DAI 变体")
        kind = ground_truth.get("audio_id_kind") or ""
        if kind and timeline_audio.get("id_kind") and kind != timeline_audio["id_kind"]:
            problems.append(f"audio_id_kind 不一致（真值 {kind}, 输入 {timeline_audio['id_kind']}）")
    return problems


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
    parser.add_argument("--proxy")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = SingleConfig(
        min_ad_s=args.min_ad,
        enter_threshold=args.threshold,
        model_path=str(args.model) if args.model else None,
        boundary_model_path=str(args.boundary_model) if args.boundary_model else None,
        proxy=args.proxy,
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
    if gt_identity:
        problems = _identity_mismatches(timeline.get("audio") or {}, gt_identity)
        if problems:
            for problem in problems:
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
        "event_precision": metrics["event_precision"] >= ACCEPTANCE["event_precision"],
        "event_recall": metrics["event_recall"] >= ACCEPTANCE["event_recall"],
        "boundary_p95_s": (
            metrics["boundary_p95_s"] is not None
            and metrics["boundary_p95_s"] <= ACCEPTANCE["boundary_p95_s"]
        ),
    }
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
        "passed": all(checks.values()),
        "matches": metrics["matches"],
        "unmatched_predictions": metrics["unmatched_predictions"],
        "unmatched_ground_truth": metrics["unmatched_ground_truth"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
