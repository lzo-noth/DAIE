from __future__ import annotations

import unittest

import numpy as np

from dai_detect.speech_cues import CUE_FEATURES, attach_speech_cues, cue_matrix


class SpeechCueTest(unittest.TestCase):
    def test_matrix_matches_names(self) -> None:
        segments = [
            {"start_s": 0.0, "end_s": 4.0, "text": "Welcome back to the show."},
            {
                "start_s": 10.0,
                "end_s": 22.0,
                "text": "This message comes from Acme. Use code SAVE20 at acme dot com.",
            },
        ]
        times = np.arange(0.5, 30.0, 0.5)
        matrix = cue_matrix(segments, times)
        self.assertEqual(matrix.shape, (len(times), len(CUE_FEATURES)))

    def test_markers_fire_inside_the_ad_read(self) -> None:
        segments = [
            {"start_s": 0.0, "end_s": 10.0, "text": "We are talking about music today."},
            {
                "start_s": 20.0,
                "end_s": 30.0,
                "text": "This message comes from Acme. Use code SAVE20 at acme dot com.",
            },
        ]
        times = np.arange(0.5, 40.0, 0.5)
        matrix = cue_matrix(segments, times)
        columns = {name: index for index, name in enumerate(CUE_FEATURES)}

        def row(at: float) -> np.ndarray:
            return matrix[int(np.argmin(np.abs(times - at)))]

        self.assertEqual(float(row(5.0)[columns["cue_marker_now"]]), 0.0)
        self.assertGreater(float(row(25.0)[columns["cue_marker_now"]]), 0.0)
        # 标记附近的“距离最近标记”应远小于无标记段落。
        self.assertLess(float(row(25.0)[columns["cue_since_marker"]]), 5.0)
        self.assertGreater(float(row(5.0)[columns["cue_since_marker"]]), 5.0)
        self.assertGreater(float(row(25.0)[columns["cue_code_near"]]), 0.0)

    def test_missing_transcript_is_neutral(self) -> None:
        times = np.arange(0.5, 5.0, 0.5)
        matrix = cue_matrix([], times)
        columns = {name: index for index, name in enumerate(CUE_FEATURES)}
        self.assertTrue(np.all(matrix[:, columns["cue_marker_now"]] == 0.0))
        # 中性值：既没有任何标记，也不给出“刚说过广告”的假信号。
        self.assertTrue(np.all(matrix[:, columns["cue_since_marker"]] == 30.0))

    def test_attach_keeps_feature_names_aligned(self) -> None:
        windows = {
            "times": np.arange(0.5, 5.0, 0.5),
            "features": np.zeros((9, 3), dtype=np.float32),
            "names": ["a", "b", "c"],
            "span_s": 1.0,
        }
        merged = attach_speech_cues(
            windows,
            [{"start_s": 0.0, "end_s": 5.0, "text": "sponsored by acme"}],
        )
        self.assertEqual(merged["features"].shape[1], len(merged["names"]))
        self.assertEqual(merged["names"][:3], ["a", "b", "c"])
        self.assertEqual(merged["names"][3:], CUE_FEATURES)


if __name__ == "__main__":
    unittest.main()
