from __future__ import annotations

import unittest

from dai_detect.ad_events import split_ad_events, split_annotation_row


def _ad(style: str = "produced", start: float = 0.0, end: float = 60.0) -> dict:
    return {
        "start_s": start,
        "end_s": end,
        "type": "third_party_ad",
        "ad_content_type": "third_party_ad",
        "ad_production_style": style,
        "insertion_mechanism": "unknown",
    }


def _segments(spans: list[tuple[float, float]], brands: list[str], host: list[bool]):
    return [
        {"start_s": start, "end_s": end} for start, end in spans
    ], {
        str(index): {
            "is_ad": True,
            "ad_type": "third_party_ad",
            "brand": brand,
            "host_read": is_host,
            "confidence": 0.9,
        }
        for index, (brand, is_host) in enumerate(zip(brands, host))
    }


class SplitAdEventsTest(unittest.TestCase):
    def test_brand_change_splits_a_block(self) -> None:
        segments, results = _segments(
            [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)],
            ["Alpha", "Beta", "Alpha"],
            [False, False, False],
        )
        events = split_ad_events(_ad(), segments, results)
        self.assertEqual(
            [(event["start_s"], event["end_s"]) for event in events],
            [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)],
        )
        self.assertTrue(all(event["ad_production_style"] == "produced" for event in events))

    def test_same_brand_merges_and_empty_brand_continues(self) -> None:
        segments, results = _segments(
            [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)],
            ["Alpha", "Alpha", ""],
            [False, False, False],
        )
        events = split_ad_events(_ad(), segments, results)
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start_s"], events[0]["end_s"]), (0.0, 60.0))

    def test_host_read_boundary_splits_and_tags_style(self) -> None:
        segments, results = _segments(
            [(0.0, 20.0), (20.0, 40.0)],
            ["Alpha", "Beta"],
            [False, True],
        )
        events = split_ad_events(_ad(), segments, results)
        self.assertEqual(
            [event["ad_production_style"] for event in events], ["produced", "host_read"]
        )

    def test_unknown_block_is_not_promoted(self) -> None:
        segments, results = _segments(
            [(0.0, 20.0), (20.0, 40.0)], ["Alpha", "Beta"], [False, False]
        )
        events = split_ad_events(_ad(style="unknown"), segments, results)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ad_production_style"], "unknown")

    def test_no_ad_segments_keeps_the_block(self) -> None:
        segments = [{"start_s": 0.0, "end_s": 20.0}]
        events = split_ad_events(_ad(), segments, {})
        self.assertEqual(len(events), 1)
        self.assertEqual((events[0]["start_s"], events[0]["end_s"]), (0.0, 60.0))

    def test_tiny_group_is_dropped(self) -> None:
        segments, results = _segments(
            [(0.0, 30.0), (30.0, 32.0), (32.0, 60.0)],
            ["Alpha", "Beta", "Gamma"],
            [False, False, False],
        )
        events = split_ad_events(_ad(), segments, results, min_duration_s=5.0)
        self.assertEqual(
            [(event["start_s"], event["end_s"]) for event in events],
            [(0.0, 30.0), (32.0, 60.0)],
        )

    def test_row_split_counts_ads(self) -> None:
        segments, results = _segments(
            [(0.0, 20.0), (20.0, 40.0)], ["Alpha", "Beta"], [False, False]
        )
        row = {"feed": "F", "episode": "E", "ads": [_ad(end=40.0)]}
        _, before, after = split_annotation_row(
            row, {"segments": segments, "results": results}
        )
        self.assertEqual((before, after), (1, 2))


if __name__ == "__main__":
    unittest.main()
