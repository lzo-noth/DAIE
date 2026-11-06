from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dai_detect.cli import main
from dai_detect.models import Config
from dai_detect.pipeline import analyze_episode, make_episode_from_path, run
from tests import synth


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-test-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        # Tests exercise the heuristic pipeline, not the repo-level learned
        # calibration artifact (which is trained on real LLM labels).
        previous = os.environ.get("DAI_CALIBRATION")
        os.environ["DAI_CALIBRATION"] = str(self.directory / "no-calibration.json")
        if previous is None:
            self.addCleanup(os.environ.pop, "DAI_CALIBRATION", None)
        else:
            self.addCleanup(os.environ.__setitem__, "DAI_CALIBRATION", previous)

    def test_shared_ad_found_in_both_episodes(self) -> None:
        first, second, third, first_ad, second_ad = synth.make_standard_episodes(self.directory)
        episodes = [make_episode_from_path(str(path)) for path in (first, second, third)]
        cfg = Config(cache=False)
        results = run(episodes, cfg, self.directory / "cache", min_score=0.5)

        repeated = {
            result.episode.title: [c for c in result.candidates if c.tier == "repeated"]
            for result in results
        }
        self.assertTrue(repeated["episode1.wav"], "episode1 未检出重复广告")
        self.assertTrue(repeated["episode2.wav"], "episode2 未检出重复广告")
        self.assertFalse(repeated["episode3.wav"], "episode3 不应有重复广告")

        for name, expected in (("episode1.wav", first_ad), ("episode2.wav", second_ad)):
            best = max(
                repeated[name],
                key=lambda candidate: _overlap(
                    candidate.start_s, candidate.end_s, expected, expected + 20.0
                ),
            )
            captured = _overlap(best.start_s, best.end_s, expected, expected + 20.0)
            self.assertGreaterEqual(captured, 18.0, f"{name} 命中不足: {best}")
            self.assertLessEqual(abs(best.start_s - expected), 3.0)
            self.assertLessEqual(abs(best.end_s - (expected + 20.0)), 3.0)
            self.assertTrue(best.matches)
            self.assertTrue(all(match.episode_id for match in best.matches))

    def test_cache_round_trip(self) -> None:
        import numpy as np

        path = self.directory / "cached.wav"
        synth.write_wav(path, synth.speech(3.0, 21))
        episode = make_episode_from_path(str(path))
        cache_dir = self.directory / "cache"
        first = analyze_episode(episode, Config(cache=True), cache_dir, fresh=True)
        self.assertTrue(list(cache_dir.glob("*.npz")), "缓存文件未写入")
        self.assertFalse(list(cache_dir.glob("*.tmp*")), "临时缓存文件未清理")
        second = analyze_episode(episode, Config(cache=True), cache_dir, fresh=False)
        np.testing.assert_array_equal(first.hashes, second.hashes)
        self.assertAlmostEqual(first.duration_s, second.duration_s, delta=1e-6)

    def test_single_episode_inserted_ad_flagged(self) -> None:
        path = self.directory / "single.wav"
        synth.make_episode(
            path,
            [synth.speech(20.0, 51), synth.ad(20.0), synth.speech(20.0, 52)],
        )
        results = run(
            [make_episode_from_path(str(path))],
            Config(cache=False),
            self.directory / "cache-single",
            min_score=0.4,
        )
        inserted = [candidate for candidate in results[0].candidates if candidate.tier == "inserted"]
        best = max(
            (_overlap(candidate.start_s, candidate.end_s, 20.0, 40.0) for candidate in inserted),
            default=0.0,
        )
        self.assertGreaterEqual(best, 10.0, results[0].candidates)

    def test_different_ads_same_slot_boosted(self) -> None:
        first = self.directory / "slot-a.wav"
        second = self.directory / "slot-b.wav"
        synth.make_episode(first, [synth.speech(20.0, 61), synth.ad(20.0, seed=7), synth.speech(20.0, 62)])
        synth.make_episode(second, [synth.speech(20.0, 63), synth.ad(20.0, seed=8), synth.speech(20.0, 64)])
        results = run(
            [make_episode_from_path(str(first)), make_episode_from_path(str(second))],
            Config(cache=False),
            self.directory / "cache-slot",
            min_score=0.5,
        )
        for result in results:
            self.assertFalse(
                [candidate for candidate in result.candidates if candidate.tier == "repeated"],
                "不同 creative 不应产生重复匹配",
            )
            inserted = [candidate for candidate in result.candidates if candidate.tier == "inserted"]
            matching = [
                candidate
                for candidate in inserted
                if _overlap(candidate.start_s, candidate.end_s, 20.0, 40.0) >= 10.0
            ]
            self.assertTrue(matching, result.candidates)
            self.assertTrue(any(candidate.evidence.get("slot_matches", 0) >= 1 for candidate in matching), matching)

    def test_clips_written_for_candidates(self) -> None:
        path = self.directory / "clip.wav"
        synth.make_episode(path, [synth.speech(20.0, 71), synth.ad(20.0), synth.speech(20.0, 72)])
        clips_dir = self.directory / "clips"
        results = run(
            [make_episode_from_path(str(path))],
            Config(cache=False),
            self.directory / "cache-clip",
            min_score=0.4,
            clips_dir=clips_dir,
        )
        clips = list(clips_dir.glob("*.mp3"))
        self.assertTrue(clips, "未生成试听片段")
        self.assertTrue(any("clip" in candidate.evidence for candidate in results[0].candidates))

    def test_cli_writes_report(self) -> None:
        jingle = synth.ad(12.0)
        first = self.directory / "a.wav"
        second = self.directory / "b.wav"
        synth.make_episode(first, [synth.speech(15.0, 1), jingle, synth.speech(15.0, 2)])
        synth.make_episode(second, [synth.speech(20.0, 3), jingle, synth.speech(10.0, 4)])
        report_path = self.directory / "report.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    "files",
                    str(first),
                    str(second),
                    "--report",
                    str(report_path),
                    "--no-cache",
                    "--quiet",
                    "--min-run",
                    "6",
                ]
            )
        self.assertEqual(code, 0, stderr.getvalue())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["episodes"]), 2)
        tiers = [candidate["tier"] for episode in payload["episodes"] for candidate in episode["candidates"]]
        self.assertIn("repeated", tiers)
        self.assertIn("候选", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
