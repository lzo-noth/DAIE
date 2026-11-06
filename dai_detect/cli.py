from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from .models import Config, Episode
from .pipeline import make_episode_from_path, run
from .platforms import verify_platforms
from .report import build_report, print_summary, write_report
from .rss import FeedError, fetch_feed, parse_feed


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", type=Path, help="将 JSON 报告写入此路径")
    parser.add_argument("--min-score", type=float, default=0.5, help="最低候选分数（默认 0.5）")
    parser.add_argument("--min-run", type=float, default=8.0, help="重复区间最短时长（默认 8 秒）")
    parser.add_argument("--cache-dir", type=Path, default=Path(".dai-cache"), help="特征缓存目录")
    parser.add_argument("--fresh", action="store_true", help="忽略缓存并重新拉流分析")
    parser.add_argument("--no-cache", action="store_true", help="不读写缓存")
    parser.add_argument("--proxy", help="HTTP(S) 代理，例如 http://127.0.0.1:7890")
    parser.add_argument("--quiet", action="store_true", help="不打印分析进度")
    return parser


def _episode_from_source(source: str) -> Episode:
    if source.startswith(("http://", "https://")):
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        return Episode(id=digest, title=source.rsplit("/", 1)[-1] or source, audio_url=source)
    return make_episode_from_path(source)


def _feed_episodes(args) -> tuple[str, str, list[Episode]]:
    data = fetch_feed(args.url, proxy=args.proxy)
    title, episodes = parse_feed(data, args.url)
    if args.episodes > 0:
        episodes = episodes[: args.episodes]
    return title or args.url, args.url, episodes


def _source_episodes(sources: list[str]) -> tuple[str, str, list[Episode]]:
    episodes = [_episode_from_source(source) for source in sources]
    missing = [
        episode.audio_url
        for episode in episodes
        if not episode.audio_url.startswith(("http://", "https://"))
        and not Path(episode.audio_url).exists()
    ]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return "本地文件", "", episodes


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    _configure_output()
    parser = argparse.ArgumentParser(
        prog="dai-detect",
        description="从流式播客音频识别 DAI 动态插入广告（重复指纹 + 插入段检测）",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = _common_parser()

    feed_parser = subparsers.add_parser("feed", parents=[common], help="分析 RSS 源中的剧集")
    feed_parser.add_argument("url", help="播客 RSS 地址")
    feed_parser.add_argument("-n", "--episodes", type=int, default=3, help="分析最新几集，0 表示全部")

    files_parser = subparsers.add_parser("files", parents=[common], help="分析多个本地文件或音频 URL")
    files_parser.add_argument("sources", nargs="+")

    file_parser = subparsers.add_parser("file", parents=[common], help="分析单个文件或音频 URL（仅插入段检测）")
    file_parser.add_argument("source")

    verify_parser = subparsers.add_parser("verify", help="验证平台 feed 可用性（只抓 RSS，不下载音频）")
    verify_parser.add_argument("registry", nargs="?", default="platforms.json", help="平台注册表 JSON")
    verify_parser.add_argument("--report", type=Path, help="将验证结果写入 JSON")
    verify_parser.add_argument("--proxy", help="HTTP(S) 代理")
    verify_parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.mode == "verify":
        try:
            report = verify_platforms(args.registry, proxy=args.proxy)
        except (OSError, ValueError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 1
        if not args.quiet:
            print("=== Feed 验证 ===")
            for platform in report["platforms"]:
                for feed in platform["feeds"]:
                    if feed["status"] == "ok":
                        print(
                            f"[ok] {platform['platform']} / {feed['name']}: "
                            f"{feed.get('title')!r} episodes={feed.get('episodes')} "
                            f"latest={feed.get('latest_title')}"
                        )
                    else:
                        print(f"[error] {platform['platform']} / {feed['name']}: {feed['error']}")
        if args.report:
            write_report(args.report, report)
            print(f"验证结果已写入: {args.report}")
        feeds = [feed for platform in report["platforms"] for feed in platform["feeds"]]
        return 0 if any(feed["status"] == "ok" for feed in feeds) else 1

    cfg = Config(min_run_s=args.min_run, cache=not args.no_cache, proxy=args.proxy)
    progress = None if args.quiet else _progress

    try:
        if args.mode == "feed":
            feed_title, feed_url, episodes = _feed_episodes(args)
        else:
            sources = [args.source] if args.mode == "file" else args.sources
            feed_title, feed_url, episodes = _source_episodes(sources)
    except (FeedError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if not episodes:
        print("错误: 没有可分析的音频剧集", file=sys.stderr)
        return 1

    if progress:
        progress(f"待分析 {len(episodes)} 集：")
        for index, episode in enumerate(episodes, start=1):
            progress(f"  [{index}] {episode.title}")

    results = run(
        episodes,
        cfg,
        args.cache_dir,
        fresh=args.fresh,
        min_score=args.min_score,
        progress=progress,
    )
    report = build_report(feed_title, feed_url, results, cfg)
    print_summary(feed_title, results)
    if args.report:
        write_report(args.report, report)
        print(f"报告已写入: {args.report}")
    if results and all(result.error for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
