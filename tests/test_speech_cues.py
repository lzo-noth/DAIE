from __future__ import annotations

import unittest

import numpy as np

from dai_detect import asr
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

    def test_phrase_times_keep_only_sponsor_phrases(self) -> None:
        from dai_detect.speech_cues import phrase_times

        segments = [
            {"start_s": 0.0, "end_s": 10.0, "text": "We talk about music and code today."},
            {
                "start_s": 20.0,
                "end_s": 30.0,
                "text": "This message comes from Acme. Visit acme dot com.",
            },
        ]
        times = phrase_times(segments)
        self.assertTrue(times, times)
        self.assertTrue(all(19.0 <= value <= 31.0 for value in times), times)

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

    @unittest.skipIf(asr.available(), "faster-whisper 已安装时不会走缺依赖分支")
    def test_missing_asr_engine_explains_the_alternative(self) -> None:
        with self.assertRaises(RuntimeError) as context:
            asr.transcribe_words("unused.wav")
        self.assertIn("faster-whisper", str(context.exception))

    def test_asr_decode_produces_sixteen_kilohertz_mono(self) -> None:
        import tempfile

        from tests import synth

        sr = 48000
        tone = (0.1 * np.sin(2 * np.pi * 440.0 * np.arange(sr) / sr)).astype(np.float32)
        frames = np.stack([tone, tone], axis=1)
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/tone.wav"
            synth.write_stereo_wav(path, frames, sr=sr)
            audio = asr._mono_16k(path)
        self.assertEqual(audio.dtype, np.float32)
        self.assertAlmostEqual(len(audio) / 16000.0, 1.0, delta=0.1)
        self.assertLess(float(np.max(np.abs(audio))), 1.0)


if __name__ == "__main__":
    unittest.main()
