from __future__ import annotations

import unittest

from dai_detect.rss import parse_duration, parse_feed


RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Older episode</title>
      <guid>old-guid</guid>
      <pubDate>Mon, 02 Jan 2023 08:00:00 GMT</pubDate>
      <enclosure url="https://cdn.example.com/old.mp3" type="audio/mpeg" length="1"/>
      <itunes:duration>1:02:03</itunes:duration>
    </item>
    <item>
      <title>Newest episode</title>
      <guid>new-guid</guid>
      <pubDate>Wed, 04 Jan 2023 08:00:00 GMT</pubDate>
      <enclosure url="https://cdn.example.com/new.mp3" type="audio/mpeg" length="1"/>
      <itunes:duration>3723</itunes:duration>
    </item>
    <item>
      <title>Video item</title>
      <guid>video</guid>
      <pubDate>Tue, 03 Jan 2023 08:00:00 GMT</pubDate>
      <enclosure url="https://cdn.example.com/video.mp4" type="video/mp4" length="1"/>
    </item>
  </channel>
</rss>
"""


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Podcast</title>
  <entry>
    <id>tag:example,2024:1</id>
    <title>Atom Episode</title>
    <updated>2024-05-01T10:00:00Z</updated>
    <link rel="enclosure" type="audio/mpeg" href="https://cdn.example.com/atom.mp3"/>
  </entry>
</feed>
"""


class RssTest(unittest.TestCase):
    def test_parse_duration_formats(self) -> None:
        self.assertEqual(parse_duration("1:02:03"), 3723.0)
        self.assertEqual(parse_duration("02:03"), 123.0)
        self.assertEqual(parse_duration("45"), 45.0)
        self.assertIsNone(parse_duration("bad"))

    def test_rss_sorted_newest_first_and_audio_only(self) -> None:
        title, episodes = parse_feed(RSS.encode("utf-8"), "https://example.com/feed")
        self.assertEqual(title, "Test Podcast")
        self.assertEqual([episode.title for episode in episodes], ["Newest episode", "Older episode"])
        self.assertEqual(episodes[0].duration_s, 3723.0)
        self.assertEqual(episodes[1].duration_s, 3723.0)
        self.assertTrue(episodes[0].audio_url.startswith("https://cdn.example.com/"))

    def test_atom_enclosure(self) -> None:
        title, episodes = parse_feed(ATOM.encode("utf-8"), "https://example.com/atom")
        self.assertEqual(title, "Atom Podcast")
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].audio_url, "https://cdn.example.com/atom.mp3")
        self.assertEqual(episodes[0].published, "2024-05-01T10:00:00Z")
