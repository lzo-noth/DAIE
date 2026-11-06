from __future__ import annotations

import unittest

import numpy as np

from dai_detect.insertion import fine_boundaries, insertion_candidates
from dai_detect.models import Config
from tests import synth


def _overlap(candidate, start: float, end: float) -> float:
    return max(0.0, min(candidate.end_s, end) - max(candidate.start_s, start))


class InsertionTest(unittest.TestCase):
    def test_detects_single_inserted_ad_without_repetition(self) -> None:
        ad = synth.ad(20.0)
        samples = np.concatenate([synth.speech(20.0, 31), ad, synth.speech(20.0, 32)])
        features = synth.extract_features(samples)
        boundaries = fine_boundaries(features)
        times = [boundary.time_s for boundary in boundaries]
        self.assertTrue(any(abs(time - 20.0) < 0.6 for time in times), times)
        self.assertTrue(any(abs(time - 40.0) < 0.6 for time in times), times)

        candidates = insertion_candidates(features, Config(), boundaries)
        captured = [candidate for candidate in candidates if _overlap(candidate, 20.0, 40.0) >= 10.0]
        self.assertTrue(captured, candidates)
        for candidate in candidates:
            self.assertFalse(
                candidate.duration_s >= 10.0 and _overlap(candidate, 20.0, 40.0) == 0.0,
                f"广告外的整段内容被误报: {candidate}",
            )

    def test_continuous_host_speech_not_flagged(self) -> None:
        features = synth.extract_features(synth.speech(60.0, 41))
        candidates = insertion_candidates(features, Config())
        strong = [candidate for candidate in candidates if candidate.duration_s >= 12.0 and candidate.score >= 0.7]
        self.assertFalse(strong, strong)


if __name__ == "__main__":
    unittest.main()
