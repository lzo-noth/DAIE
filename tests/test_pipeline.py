from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dai_detect.cli import main
from dai_detect.models import Config
from dai_detect.pipeline import make_episode_from_path, run
from tests import synth


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="dai-test-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

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
