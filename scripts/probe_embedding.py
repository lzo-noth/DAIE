#!/usr/bin/env python3
"""Offline feasibility probe: hand-crafted windows vs AudioSet embeddings.

The shipped detector scores hand-crafted features per 1.0 s window. This script
asks the one question that matters before touching production code: does a
pretrained acoustic representation (YAMNet, AudioSet) separate ads from ordinary
content better than the hand-crafted set, on the same episodes and the same
feed-grouped folds?

Two levels are reported, because a window classifier that looks decent can still
produce useless events:

* window level — AUC and recall at fixed specificity;
* event level — full window predictions decoded with the deployed decoder,
  refined with the boundary evidence, then matched against ground truth, giving
  event precision/recall/F1, false positives per hour, boundary percentiles and
  per-feed failures.

Nothing here runs in the runtime path. Output is a small JSON plus the final
event metrics printed to the log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dai_detect.acoustic_embedding import YamnetEmbedder, window_embeddings
from dai_detect.audio import DownmixResampler, iter_pcm_stereo
from dai_detect.boundary_model import boundary_change_matrix, heuristic_head_scores
from dai_detect.decode_config import DEFAULT_MAX_GAP_S, DEFAULT_SEARCH_S, resolve_thresholds
from dai_detect.evaluation import evaluate_events
from dai_detect.features import FeatureExtractor
from dai_detect.forensic import StereoForensicExtractor
from dai_detect.metrics import roc_auc
from dai_detect.refinement import refine_events
from dai_detect.single_features import (
    WINDOW_FEATURES,
    WINDOW_HOP_S,
    WINDOW_SPAN_S,
    build_windows,
    window_labels,
)
from dai_detect.target_scope import (
    IGNORED_PRODUCTION_STYLES,
    ad_style,
)
from dai_detect.temporal_decoder import decode_windows
from scripts.train_single_episode import (
    MIN_AD_S,
    _balance,
    _fit_logistic,
    _group_folds,
    _load_annotations,
    _predict,
    _resolve_audio,
)

SPECIFICITIES = (0.90, 0.95, 0.98, 0.99)

VIEWS = ("all", "produced", "host_read")
KINDS = ("baseline", "yamnet", "baseline+yamnet")


def _style_ads(ads: list[dict], view: str) -> list[dict]:
    """Ads in one evaluation scope; style-unknown ads are never included.

    ``all`` 是“全部已完成风格标注的广告”（produced + host_read），不是“所有
    广告”；unknown/缺失的区间在监督和评测里都必须屏蔽。
    """
    known = [ad for ad in ads if ad_style(ad) not in IGNORED_PRODUCTION_STYLES]
    if view == "all":
        return known
    return [ad for ad in known if ad_style(ad) == view]


def _ignored_ads(ads: list[dict]) -> list[dict]:
    return [ad for ad in ads if ad_style(ad) in IGNORED_PRODUCTION_STYLES]


def _view_rows(windows: dict, ads: list[dict], view: str) -> tuple[np.ndarray, np.ndarray]:
    """Kept window rows and labels for one scope.

    Windows ambiguous for one scope may be usable in another, so each scope
    keeps its own rows instead of sharing a single mask.
    """
    indices, labels = window_labels(
        windows, _style_ads(ads, view), ignore=_ignored_ads(ads)
    )
    return np.asarray(indices, dtype=np.int64), labels.astype(np.int8)


def _extract(job: dict) -> dict:
    """Decode one episode once and cache every view and feature set.

    Full-window matrices are kept (not only labelled rows) because event-level
    evaluation has to decode the whole episode, not just the labelled windows.
    """
    collector = FeatureExtractor()
    forensic = StereoForensicExtractor()
    downmix = DownmixResampler()
    embedder = None if job.get("skip_embedding") else YamnetEmbedder()
    for block in iter_pcm_stereo(job["audio"], block_s=2.0, proxy=job.get("proxy")):
        forensic.push(block)
        mono = downmix.push(block.mean(axis=1))
        collector.push(mono)
        if embedder is not None:
            embedder.push(mono)
    mono_payload = collector.finish()
    forensic_payload = forensic.finish()
    windows = build_windows(mono_payload, forensic_payload)
    times = np.asarray(windows["times"], dtype=np.float64)
    base = np.asarray(windows["features"], dtype=np.float32)
    if embedder is None:
        embedding = np.zeros((len(times), 0), dtype=np.float16)
    else:
        patches, centres = embedder.finish()
        matrix = window_embeddings(times, centres, patches)
        scale = matrix.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        # float16 只为控制 90 集全量时的内存；训练前统一转回 float32。
        embedding = ((matrix - matrix.mean(axis=0)) / scale).astype(np.float16)
    change = boundary_change_matrix(mono_payload, forensic_payload)
    if len(change["times"]):
        onset, offset = heuristic_head_scores(change["features"])
    else:
        onset = offset = np.zeros(0, dtype=np.float32)
    ads = job.get("ads") or []
    views = {}
    for view in VIEWS:
        indices, labels = _view_rows(windows, ads, view)
        views[view] = {"rows": indices, "labels": labels}
    return {
        "feed": job["feed"],
        "episode": job["episode"],
        "duration_s": float(windows.get("duration_s") or 0.0),
        "times": times,
        "base": base,
        "embedding": embedding,
        "boundary": {
            "times": np.asarray(change["times"], dtype=np.float64),
            "onset": onset,
            "offset": offset,
        },
        "ads": ads,
        "views": views,
    }


def _run_parallel(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        results = []
        for index, job in enumerate(jobs):
            began = time.time()
            results.append(_extract(job))
            print(
                f"[extract] {index + 1}/{len(jobs)} {job['feed']} / {job['episode']}"
                f" -> {len(results[-1]['views']['all']['labels'])} 窗口，用时 {time.time() - began:.0f}s",
                flush=True,
            )
        return results
    print(f"[extract] 并行解码 {len(jobs)} 集，workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = []
        for index, payload in enumerate(pool.map(_extract, jobs)):
            results.append(payload)
            print(
                f"[extract] {index + 1}/{len(jobs)} {payload['feed']} / {payload['episode']}"
                f" -> {len(payload['views']['all']['labels'])} 窗口",
                flush=True,
            )
        return results


def _pool_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    fold_thresholds: list[tuple[list[int], dict[str, float] | None]],
    target: str,
    hours: float,
) -> dict:
    """Pool one specificity target using thresholds fixed outside the test fold.

    The threshold must never be a quantile of the held-out fold's own negative
    scores: that is label leakage in the decision rule and makes the reported
    specificity unreachable in deployment.
    """
    positives = 0
    detected = 0
    false_positives = 0
    negatives = 0
    for fold, thresholds in fold_thresholds:
        if not fold or not thresholds or thresholds.get(target) is None:
            continue
        truth = labels[fold]
        predicted = scores[fold] >= thresholds[target]
        positives += int(np.count_nonzero(truth == 1))
        detected += int(np.count_nonzero(predicted & (truth == 1)))
        negatives += int(np.count_nonzero(truth == 0))
        false_positives += int(np.count_nonzero(predicted & (truth == 0)))
    return {
        "specificity": round(1.0 - false_positives / negatives, 4) if negatives else None,
        "recall": round(detected / positives, 4) if positives else None,
        "false_positives_per_hour": round(false_positives / hours, 1) if hours else None,
    }


def _inner_thresholds(
    X_all: np.ndarray,
    labels: np.ndarray,
    rows_by_episode: list[np.ndarray],
    items: list[dict],
    train_episodes: list[int],
    ratio: int,
    inner_folds: int,
) -> dict[str, float] | None:
    """Specificity thresholds from inner grouped CV inside the outer train set.

    Outer test episodes never contribute scores, labels or quantiles here; the
    outer model is only ever evaluated against thresholds learned from data it
    was allowed to see.
    """
    if len(train_episodes) < 2 or inner_folds < 2:
        return None
    groups = [str(items[episode]["feed"]) for episode in train_episodes]
    if len(set(groups)) < 2:
        return None
    folds = _group_folds(groups, k=min(inner_folds, len(set(groups))))
    inner_oof = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in folds:
        held_positions = set(fold)
        held = [train_episodes[position] for position in fold]
        kept = [
            episode
            for position, episode in enumerate(train_episodes)
            if position not in held_positions
        ]
        train_rows = np.concatenate([rows_by_episode[episode] for episode in kept])
        held_rows = np.concatenate([rows_by_episode[episode] for episode in held])
        truth = labels[train_rows]
        if len(np.unique(truth)) < 2 or held_rows.size == 0:
            continue
        model = _fit_logistic(*_balance(X_all[train_rows], truth, ratio=ratio))
        inner_oof[held_rows] = _predict(model, X_all[held_rows])
    negatives = inner_oof[(labels == 0) & ~np.isnan(inner_oof)]
    if negatives.size == 0:
        return None
    return {
        f"{target:.2f}": float(np.quantile(negatives, target))
        for target in SPECIFICITIES
    }


def _window_folds(
    episode_folds: list[list[int]], bounds: list[tuple[int, int]]
) -> list[list[int]]:
    """Expand episode-level folds into the window-row indices they cover.

    ``_group_folds`` groups episodes; the probe evaluates single windows, so a
    fold has to become a list of window rows. Indexing the window matrix with
    episode ids silently trains on everything and tests on the first few rows.
    """
    return [
        [row for episode in fold for row in range(*bounds[episode])]
        for fold in episode_folds
    ]


def _episode_matrix(item: dict, kind: str) -> np.ndarray:
    if kind == "yamnet":
        return item["embedding"].astype(np.float32)
    if kind == "baseline+yamnet":
        return np.hstack([item["base"], item["embedding"].astype(np.float32)])
    return item["base"]


def _refined_events(item: dict, probabilities: np.ndarray, threshold: float) -> list[dict]:
    """Decode one episode exactly like the deployed pipeline.

    Boundary refinement uses this episode's heuristic head scores: the probe is
    comparing representations, so the boundary head must not be a second
    variable. The full decoder and refinement chain still runs, and the report
    states which boundary model was used.
    """
    enter, exit_value = resolve_thresholds(threshold)
    events = decode_windows(
        item["times"],
        probabilities,
        hop_s=WINDOW_HOP_S,
        span_s=WINDOW_SPAN_S,
        min_ad_s=MIN_AD_S,
        max_gap_s=DEFAULT_MAX_GAP_S,
        enter_threshold=enter,
        exit_threshold=exit_value,
    )
    if not events:
        return []
    boundary = item["boundary"]
    if len(boundary["times"]) == 0:
        return [
            {"start": event.start, "end": event.end} for event in events
        ]
    change = {
        "times": boundary["times"],
        "onset_scores": boundary["onset"],
        "offset_scores": boundary["offset"],
        "scores": np.maximum(boundary["onset"], boundary["offset"]),
        "kind": "heuristic_onset_offset",
    }
    return refine_events(events, change, search_s=DEFAULT_SEARCH_S, min_duration=MIN_AD_S)


def _empty_event_bucket() -> dict:
    return {
        "matched": 0,
        "predicted": 0,
        "ground_truth": 0,
        "errors": [],
        "per_feed": {},
        "false_positive_events": [],
    }


def _event_summary(bucket: dict, hours: float, feed_hours: dict[str, float]) -> dict:
    matched = bucket["matched"]
    predicted = bucket["predicted"]
    truth = bucket["ground_truth"]
    precision = matched / predicted if predicted else None
    recall = matched / truth if truth else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    errors = np.asarray(bucket["errors"], dtype=np.float64)
    per_feed = {}
    for feed, counts in sorted(bucket["per_feed"].items()):
        feed_span = feed_hours.get(feed, 0.0)
        per_feed[feed] = {
            "matched": counts["matched"],
            "false_positives": counts["false_positives"],
            "false_negatives": counts["false_negatives"],
            "ground_truth": counts["ground_truth"],
            "false_positives_per_hour": (
                round(counts["false_positives"] / feed_span, 2) if feed_span else None
            ),
        }
    details = bucket["false_positive_events"]
    return {
        "event_precision": round(precision, 4) if precision is not None else None,
        "event_recall": round(recall, 4) if recall is not None else None,
        "event_f1": round(f1, 4) if f1 is not None else None,
        "predicted_events": predicted,
        "matched": matched,
        "ground_truth_events": truth,
        "false_positives_per_hour": (
            round((predicted - matched) / hours, 1) if hours else None
        ),
        "boundary_p50_s": round(float(np.percentile(errors, 50)), 3) if errors.size else None,
        "boundary_p90_s": round(float(np.percentile(errors, 90)), 3) if errors.size else None,
        "boundary_p95_s": round(float(np.percentile(errors, 95)), 3) if errors.size else None,
        "boundary_model": "heuristic_onset_offset",
        "threshold_source": "outer_train_inner_cv",
        "false_positive_events": details[:100],
        "false_positive_events_truncated": len(details) > 100,
        "per_feed": per_feed,
    }


def _evaluate(
    name: str,
    kind: str,
    items: list[dict],
    view: str,
    rows_by_episode: list[np.ndarray],
    labels: np.ndarray,
    folds: list[list[int]],
    episode_folds: list[list[int]],
    hours: float,
    feed_hours: dict[str, float],
    ratio: int,
    inner_folds: int,
) -> dict:
    """Window-level and event-level evaluation for one feature set.

    Every threshold is selected by grouped CV *inside* the outer training set.
    The held-out fold only supplies scores to be thresholded; its labels never
    choose the threshold.
    """
    began = time.time()
    X_all = np.vstack(
        [_episode_matrix(item, kind)[rows] for item, rows in zip(items, rows_by_episode)]
    )
    oof = np.zeros(len(labels), dtype=np.float64)
    event_buckets = {target: _empty_event_bucket() for target in SPECIFICITIES}
    fold_thresholds: list[tuple[list[int], dict[str, float] | None]] = []
    scored_folds: list[list[int]] = []
    all_episodes = list(range(len(items)))
    for fold_rows, fold_episodes in zip(folds, episode_folds):
        if not fold_rows:
            continue
        test = np.zeros(len(labels), dtype=bool)
        test[fold_rows] = True
        train = np.flatnonzero(~test)
        if len(np.unique(labels[train])) < 2:
            continue
        model = _fit_logistic(*_balance(X_all[train], labels[train], ratio=ratio))
        fold_scores = _predict(model, X_all[fold_rows])
        oof[fold_rows] = fold_scores
        scored_folds.append(fold_rows)
        held = set(fold_episodes)
        train_episodes = [episode for episode in all_episodes if episode not in held]
        thresholds = _inner_thresholds(
            X_all,
            labels,
            rows_by_episode,
            items,
            train_episodes,
            ratio,
            inner_folds,
        )
        fold_thresholds.append((fold_rows, thresholds))
        if not thresholds:
            continue
        probabilities = {
            episode: _predict(model, _episode_matrix(items[episode], kind))
            for episode in fold_episodes
        }
        for target in SPECIFICITIES:
            key = f"{target:.2f}"
            threshold = thresholds[key]
            bucket = event_buckets[target]
            for episode in fold_episodes:
                item = items[episode]
                truth = [
                    {"start_s": float(ad["start_s"]), "end_s": float(ad["end_s"])}
                    for ad in _style_ads(item["ads"], view)
                ]
                predicted = _refined_events(item, probabilities[episode], threshold)
                metrics = evaluate_events(predicted, truth)
                bucket["matched"] += len(metrics["matches"])
                bucket["predicted"] += metrics["predicted_events"]
                bucket["ground_truth"] += metrics["ground_truth_events"]
                for match in metrics["matches"]:
                    bucket["errors"].extend(
                        [float(match["start_error"]), float(match["end_error"])]
                    )
                for event in metrics["unmatched_predictions"]:
                    start = float(event.get("start_s", event.get("start")))
                    end = float(event.get("end_s", event.get("end")))
                    gaps = [
                        max(float(gt["start_s"]) - end, start - float(gt["end_s"]), 0.0)
                        for gt in truth
                    ]
                    if len(bucket["false_positive_events"]) < 400:
                        bucket["false_positive_events"].append(
                            {
                                "feed": item["feed"],
                                "episode": item["episode"],
                                "start_s": round(start, 2),
                                "end_s": round(end, 2),
                                "duration_s": round(end - start, 2),
                                "confidence": event.get("confidence"),
                                "nearest_gt_gap_s": round(min(gaps), 2) if gaps else None,
                            }
                        )
                feed_counts = bucket["per_feed"].setdefault(
                    item["feed"],
                    {
                        "matched": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "ground_truth": 0,
                    },
                )
                feed_counts["matched"] += len(metrics["matches"])
                feed_counts["false_positives"] += int(metrics["false_positives"])
                feed_counts["false_negatives"] += int(metrics["false_negatives"])
                feed_counts["ground_truth"] += metrics["ground_truth_events"]

    if not scored_folds:
        report = {
            "features": int(X_all.shape[1]),
            "skipped": "没有任何折可以训练（标签只有一类或折为空）",
            "fit_seconds": round(time.time() - began, 1),
        }
        print(f"[cv] {name}: {json.dumps(report, ensure_ascii=False)}", flush=True)
        return report
    fold_auc = [
        roc_auc(labels[fold], oof[fold])
        for fold in scored_folds
        if len(np.unique(labels[fold])) > 1
    ]
    with_thresholds = sum(1 for _, thresholds in fold_thresholds if thresholds)
    report = {
        "features": int(X_all.shape[1]),
        "auc_macro": round(float(np.mean(fold_auc)), 4) if fold_auc else None,
        "auc_fold": [round(value, 4) for value in fold_auc],
        "auc_pooled": round(float(roc_auc(labels, oof)), 4),
        "threshold_source": "outer_train_inner_cv",
        "folds_with_thresholds": with_thresholds,
        "recall_at_specificity": {},
        "event": {},
        "fit_seconds": round(time.time() - began, 1),
    }
    for target in SPECIFICITIES:
        key = f"{target:.2f}"
        report["recall_at_specificity"][key] = _pool_at_thresholds(
            oof, labels, fold_thresholds, key, hours
        )
        report["event"][key] = _event_summary(
            event_buckets[target], hours, feed_hours
        )
    print(f"[cv] {name}: {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=24, help="0=全部")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="外层训练集内部用于选阈值的分组折数（留出集不参与）",
    )
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--skip-embedding", action="store_true", help="只跑基线，用于自检")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    base_dir = args.annotations.parent
    rows = _load_annotations(args.annotations)
    if args.max_episodes:
        rows = rows[: args.max_episodes]
    jobs = [
        {
            "audio": _resolve_audio(row["audio"], base_dir),
            "feed": str(row.get("feed") or "unknown"),
            "episode": str(row.get("episode") or ""),
            "ads": row.get("ads") or [],
            "skip_embedding": args.skip_embedding,
        }
        for row in rows
    ]
    feeds = {job["feed"] for job in jobs}
    if len(feeds) < 2:
        raise SystemExit("至少需要两个 feed 才能做分组交叉验证")

    workers = args.workers or int(os.environ.get("PROBE_WORKERS", "0") or 0)
    if workers <= 0:
        workers = min(4, os.cpu_count() or 1)
    began = time.time()
    extracted = _run_parallel(jobs, max(1, min(workers, len(jobs))))
    print(f"[time] 特征提取完成: {time.time() - began:.0f}s", flush=True)

    groups = [item["feed"] for item in extracted]
    hours = sum(item["duration_s"] for item in extracted) / 3600.0
    feed_hours: dict[str, float] = {}
    for item in extracted:
        feed_hours[item["feed"]] = feed_hours.get(item["feed"], 0.0) + item["duration_s"] / 3600.0
    episode_folds = _group_folds(groups, k=args.folds)
    reports = {
        "episodes": len(extracted),
        "feeds": len(set(groups)),
        "hours": round(hours, 2),
        "folds": len(episode_folds),
        "views": {},
    }
    kinds = ("baseline",) if args.skip_embedding else KINDS
    for view in VIEWS:
        labels = np.concatenate(
            [item["views"][view]["labels"] for item in extracted]
        ).astype(np.float64)
        rows_by_episode = [item["views"][view]["rows"] for item in extracted]
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for item in extracted:
            count = len(item["views"][view]["labels"])
            bounds.append((cursor, cursor + count))
            cursor += count
        folds = _window_folds(episode_folds, bounds)
        if sum(len(fold) for fold in folds) != len(labels):
            raise SystemExit(f"{view} 折划分没有覆盖全部窗口行，指标会失真")
        entry = {
            "windows": int(len(labels)),
            "positive_windows": int(labels.sum()),
            "features": {},
        }
        for kind in kinds:
            entry["features"][kind] = _evaluate(
                f"{view}/{kind}",
                kind,
                extracted,
                view,
                rows_by_episode,
                labels,
                folds,
                episode_folds,
                hours,
                feed_hours,
                args.negatives_per_positive,
                args.inner_folds,
            )
        reports["views"][view] = entry
    reports["feature_names"] = {"baseline": WINDOW_FEATURES}
    reports["elapsed_s"] = round(time.time() - began, 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
