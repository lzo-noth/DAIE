#!/usr/bin/env python3
"""Generate a deterministic synthetic set for the training pipeline smoke test.

This is an engineering smoke test, not a performance-acceptance dataset. It
creates three training feeds and one unseen blind feed, then snapshots the
blind feed so strict model loading and independence checks can run in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import synth

SR = 48000


def _episode(index: int, sr: int = SR) -> np.ndarray:
    return np.concatenate(
        [
            synth.stereo(synth.speech(12.0, 1 + index, sr=sr), 2 + index, sr=sr),
            synth.stereo(synth.ad(14.0, seed=3 + index, sr=sr), 4 + index, sr=sr),
            synth.stereo(synth.speech(12.0, 5 + index, sr=sr), 6 + index, sr=sr),
        ],
        axis=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("out/pipeline"))
    args = parser.parse_args()
    root = args.out_dir
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)

    rows = []
    for index, feed in enumerate(("A", "B", "C", "D")):
        path = root / "audio" / f"{feed}.wav"
        synth.write_stereo_wav(path, _episode(index), sr=SR)
        if feed != "D":
            rows.append(
                {
                    "audio": str(path),
                    "feed": feed,
                    "episode": feed,
                    "ads": [
                        {
                            "start_s": 12.0,
                            "end_s": 26.0,
                            "type": "third_party_inserted",
                        }
                    ],
                }
            )
    (root / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = root / "blind_manifest.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "snapshot_audio.py"),
            "--source",
            str(root / "audio" / "D.wav"),
            "--feed",
            "D",
            "--episode",
            "D",
            "--out-dir",
            str(root / "snapshots"),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )
    entry = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    ground_truth = {
        "duration_s": 38.0,
        "ads": [{"start_s": 12.0, "end_s": 26.0}],
        "source_sha256": entry["source_sha256"],
        "audio_id": entry["audio_id"],
        "audio_id_kind": entry["audio_id_kind"],
    }
    (root / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"audio": entry["audio"], "annotations": str(root / "annotations.jsonl")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
