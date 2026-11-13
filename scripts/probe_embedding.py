#!/usr/bin/env python3
"""Offline feasibility probe: hand-crafted windows vs AudioSet embeddings.

The shipped detector scores 77 hand-crafted features per 1.0 s window. On the
90-episode release annotations those features cap out at ~0.76 window AUC, and
at 99% specificity they keep only ~1% of positive windows, so the event-level
pipeline cannot be rescued downstream. This script asks the one question that
matters before touching production code: does a pretrained acoustic
representation (YAMNet, AudioSet) separate ads from ordinary content on the
same episodes and the same grouped folds?

It changes nothing in the runtime path. Output is a small JSON with AUC and
recall-at-specificity per feature set.
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
from dai_detect.features import FeatureExtractor
from dai_detect.forensic import StereoForensicExtractor
from dai_detect.metrics import roc_auc
from dai_detect.single_features import WINDOW_FEATURES, build_windows, window_labels
from scripts.train_single_episode import (
    _balance,
    _fit_logistic,
    _group_folds,
    _load_annotations,
    _predict,
    _resolve_audio,
)

SPECIFICITIES = (0.90, 0.95, 0.98, 0.99)


def _extract(job: dict) -> dict:
    """Decode one episode once and return labelled windows for every view."""
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
    windows = build_windows(collector.finish(), forensic.finish())
    indices, labels = window_labels(windows, job.get("ads") or [])
    times = np.asarray(windows["times"], dtype=np.float64)[indices]
    base = np.asarray(windows["features"], dtype=np.float32)[indices]
    if embedder is None:
        embedding = np.zeros((len(indices), 0), dtype=np.float32)
    else:
        patches, centres = embedder.finish()
        embedding = window_embeddings(times, centres, patches)
        scale = embedding.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        embedding = ((embedding - embedding.mean(axis=0)) / scale).astype(np.float32)
    return {
        "feed": job["feed"],
        "episode": job["episode"],
        "duration_s": float(windows.get("duration_s") or 0.0),
        "labels": labels.astype(np.int8),
        "base": base,
        "embedding": embedding,
    }


def _run_parallel(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        results = []
        for index, job in enumerate(jobs):
            began = time.time()
            results.append(_extract(job))
            print(
                f"[extract] {index + 1}/{len(jobs)} {job['feed']} / {job['episode']}"
                f" -> {len(results[-1]['labels'])} 窗口，用时 {time.time() - began:.0f}s",
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
                f" -> {len(payload['labels'])} 窗口",
                flush=True,
            )
        return results


def _recall_at_specificity(
    oof: np.ndarray, labels: np.ndarray, folds: list[list[int]], target: float, hours: float
) -> dict:
    """Pick each fold's own threshold at ``target`` specificity, then pool.

    Fold-local thresholds are required: the per-fold score scales differ, which
    is why macro AUC is reported alongside the pooled number.
    """
    positives = 0
    detected = 0
    false_positives = 0
    negatives = 0
    for fold in folds:
        scores = oof[fold]
        truth = labels[fold]
        negative_scores = scores[truth == 0]
        if len(negative_scores) == 0:
            continue
        threshold = float(np.quantile(negative_scores, target))
        predicted = scores >= threshold
        positives += int(np.count_nonzero(truth == 1))
        detected += int(np.count_nonzero(predicted & (truth == 1)))
        negatives += int(np.count_nonzero(truth == 0))
        false_positives += int(np.count_nonzero(predicted & (truth == 0)))
    return {
        "specificity": round(1.0 - false_positives / negatives, 4) if negatives else None,
        "recall": round(detected / positives, 4) if positives else None,
        "false_positives_per_hour": round(false_positives / hours, 1) if hours else None,
    }


def _evaluate(
    name: str,
    X: np.ndarray,
    labels: np.ndarray,
    folds: list[list[int]],
    hours: float,
    ratio: int,
) -> dict:
    began = time.time()
    oof = np.zeros(len(labels), dtype=np.float64)
    for fold in folds:
        test = np.zeros(len(labels), dtype=bool)
        test[fold] = True
        train = np.flatnonzero(~test)
        X_train, y_train = _balance(X[train], labels[train], ratio=ratio)
        model = _fit_logistic(X_train, y_train)
        oof[fold] = _predict(model, X[fold])
    fold_auc = [roc_auc(labels[fold], oof[fold]) for fold in folds if len(np.unique(labels[fold])) > 1]
    report = {
        "features": int(X.shape[1]),
        "auc_macro": round(float(np.mean(fold_auc)), 4) if fold_auc else None,
        "auc_fold": [round(value, 4) for value in fold_auc],
        "auc_pooled": round(float(roc_auc(labels, oof)), 4),
        "recall_at_specificity": {},
        "fit_seconds": round(time.time() - began, 1),
    }
    for target in SPECIFICITIES:
        report["recall_at_specificity"][f"{target:.2f}"] = _recall_at_specificity(
            oof, labels, folds, target, hours
        )
    print(f"[cv] {name}: {json.dumps(report, ensure_ascii=False)}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=24, help="0=全部")
    parser.add_argument("--folds", type=int, default=6)
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

    labels = np.concatenate([item["labels"] for item in extracted]).astype(np.float64)
    base = np.vstack([item["base"] for item in extracted])
    groups = [item["feed"] for item in extracted]
    hours = sum(item["duration_s"] for item in extracted) / 3600.0
    folds = _group_folds(groups, k=args.folds)

    reports = {
        "episodes": len(extracted),
        "feeds": len({item["feed"] for item in extracted}),
        "windows": int(len(labels)),
        "positive_windows": int(labels.sum()),
        "hours": round(hours, 2),
        "folds": len(folds),
        "features": {},
    }
    reports["features"]["baseline"] = _evaluate(
        "baseline", base, labels, folds, hours, args.negatives_per_positive
    )
    if not args.skip_embedding:
        embedding = np.vstack([item["embedding"] for item in extracted])
        reports["features"]["yamnet"] = _evaluate(
            "yamnet", embedding, labels, folds, hours, args.negatives_per_positive
        )
        reports["features"]["baseline+yamnet"] = _evaluate(
            "baseline+yamnet",
            np.hstack([base, embedding]),
            labels,
            folds,
            hours,
            args.negatives_per_positive,
        )
    reports["feature_names"] = {"baseline": WINDOW_FEATURES}
    reports["elapsed_s"] = round(time.time() - began, 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
