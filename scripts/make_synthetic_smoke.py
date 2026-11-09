#!/usr/bin/env python3
"""Generate a deterministic synthetic set for the training pipeline smoke test.

This is an engineering smoke test, not a performance-acceptance dataset. It
creates training feeds with ads, one training feed without ads, one blind feed
with an ad and one blind feed without ads. The no-ad episodes matter: a set
where every episode contains an ad can never show whether the detector keeps
quiet on ordinary production changes, music beds or loudness jumps.

The no-ad blind feed is snapshotted too so strict model loading, independence
checks and the "zero false positives on an empty ground truth" path run in CI.
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


def _crossfade(first: np.ndarray, second: np.ndarray, seconds: float = 0.5) -> np.ndarray:
    """Join two stereo parts with a linear crossfade.

    Real DAI audio splices ads in without a hard cut; a click-free crossfade
    makes the boundary detector face the same shape it will see in production.
    """
    fade = max(1, int(seconds * SR))
    if len(first) < fade or len(second) < fade:
        return np.concatenate([first, second], axis=0)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    joined = first.copy()
    joined[-fade:] = first[-fade:] * (1.0 - ramp) + second[:fade] * ramp
    return np.concatenate([joined, second[fade:]], axis=0)


def _episode(index: int, sr: int = SR) -> tuple[np.ndarray, dict]:
    """Episode with an intro bed, a speech lead-in, one ad break and an outro."""
    intro = synth.stereo(synth.ad(2.0, seed=40 + index, sr=sr), 41 + index, sr=sr)
    speech = synth.stereo(synth.speech(12.0, 1 + index, sr=sr), 2 + index, sr=sr)
    ad_break = synth.stereo(synth.ad(14.0, seed=3 + index, sr=sr), 4 + index, sr=sr)
    outro = synth.stereo(synth.speech(12.0, 5 + index, sr=sr), 6 + index, sr=sr)
    fade_intro = 0.2
    frames = _crossfade(intro, speech, fade_intro)
    lead_in = (len(intro) + len(speech)) / sr - fade_intro
    fade_ad = 0.2
    frames = _crossfade(frames, ad_break, fade_ad)
    start = lead_in - fade_ad
    end = start + len(ad_break) / sr
    frames = _crossfade(frames, outro, 0.2)
    ad = {"start_s": round(start, 2), "end_s": round(end, 2), "type": "third_party_ad"}
    return frames, ad


def _negative_episode(index: int, sr: int = SR) -> np.ndarray:
    """No ad, but ordinary cuts, a music bed and a loudness step.

    These are exactly the events that produce false positives on real shows,
    so they must appear in both the training control set and the blind set.
    """
    intro = synth.stereo(synth.ad(4.0, seed=51 + index, sr=sr), 52 + index, sr=sr)
    speech = synth.stereo(synth.speech(12.0, 21 + index, sr=sr), 22 + index, sr=sr)
    music = synth.stereo(synth.ad(6.0, seed=23 + index, sr=sr), 24 + index, sr=sr)
    louder = synth.stereo(
        synth.speech(13.0, 25 + index, amplitude=0.25, sr=sr), 26 + index, sr=sr
    )
    return _crossfade(
        _crossfade(_crossfade(intro, speech, 0.4), music, 0.4), louder, 0.3
    )


def _snapshot(root: Path, name: str, feed: str, path: Path) -> dict:
    manifest = root / f"blind_{feed}.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "snapshot_audio.py"),
            "--source",
            str(path),
            "--feed",
            feed,
            "--episode",
            feed,
            "--out-dir",
            str(root / "snapshots"),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )
    return json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])


def _ground_truth(entry: dict, duration: float, ads: list[dict]) -> dict:
    return {
        "duration_s": duration,
        "ads": ads,
        "source_sha256": entry["source_sha256"],
        "audio_id": entry["audio_id"],
        "audio_id_kind": entry["audio_id_kind"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("out/pipeline"))
    args = parser.parse_args()
    root = args.out_dir
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)

    rows = []
    positive_ads: dict[str, dict] = {}
    positive_duration: dict[str, float] = {}
    for index, feed in enumerate(("A", "B", "C", "D")):
        path = root / "audio" / f"{feed}.wav"
        frames, ad = _episode(index)
        synth.write_stereo_wav(path, frames, sr=SR)
        positive_ads[feed] = ad
        positive_duration[feed] = len(frames) / SR
        if feed != "D":
            rows.append(
                {
                    "audio": str(path),
                    "feed": feed,
                    "episode": feed,
                    "ads": [ad],
                }
            )
    negative_duration: dict[str, float] = {}
    for index, feed in enumerate(("E", "F")):
        path = root / "audio" / f"{feed}.wav"
        frames = _negative_episode(index)
        negative_duration[feed] = len(frames) / SR
        synth.write_stereo_wav(path, frames, sr=SR)
        if feed != "F":
            rows.append(
                {
                    "audio": str(path),
                    "feed": feed,
                    "episode": feed,
                    "ads": [],
                }
            )
    (root / "annotations.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    manifest = root / "blind_manifest.jsonl"
    positive_entry = _snapshot(root, "D", "D", root / "audio" / "D.wav")
    negative_entry = _snapshot(root, "F", "F", root / "audio" / "F.wav")
    manifest.write_text(
        "\n".join(
            json.dumps(entry, ensure_ascii=False)
            for entry in (positive_entry, negative_entry)
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "ground_truth.json").write_text(
        json.dumps(
            _ground_truth(
                positive_entry,
                positive_duration["D"],
                [positive_ads["D"]],
            )
        ),
        encoding="utf-8",
    )
    (root / "ground_truth_no_ad.json").write_text(
        json.dumps(_ground_truth(negative_entry, negative_duration["F"], [])),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audio": positive_entry["audio"],
                "audio_no_ad": negative_entry["audio"],
                "annotations": str(root / "annotations.jsonl"),
                "blind_manifest": str(manifest),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
