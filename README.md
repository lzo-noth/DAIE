# DAI 广告检测原型

从流式播客音频中识别 DAI（Dynamic Ad Insertion，动态广告插入）广告区间。
输入播客 RSS 地址，输出每集音频里疑似广告的起止时间、评分与证据，报告可作为 GitHub Actions artifact 下载。

当前是离线分析原型：音频经 ffmpeg 流式解码，特征与指纹按块处理，不落盘原始音频。

## 原理

DAI 广告由服务端在请求音频时插入，因此有两个可利用的稳定特征：

1. **跨集重复指纹（主信号）**：同一条广告创意通常会在同一播客的多集、不同位置重复出现，而节目正文不会逐字重复。对音频提取 Wang 风格频谱峰值指纹，做跨集偏移投票，重复区间即为强候选。
2. **声学突变（辅助信号）**：广告是单独文件拼接进来的，边界处常出现响度跳变、音乐床、说话人/频谱突变。对响度、频谱平坦度、频谱通量、MFCC 做变化点检测，得到声学候选。

两者合并后按证据打分：重复匹配置信度、边界强度、音乐性、响度偏差、位置先验（前贴片/后贴片）。
没有标注数据时，重复区间是客观证据；声学候选标记为“未复核”，需要人工确认。

## 安装

需要 Python 3.11+、NumPy、ffmpeg/ffprobe（需在 PATH 中）。

```bash
pip install numpy
ffmpeg -version   # 确认可用
```

## 用法

分析 RSS 最新 3 集（跨集指纹需要至少 2 集）：

```bash
python -m dai_detect feed "https://feeds.simplecast.com/hNaFxXpO" -n 3 --report report.json
```

分析本地文件（同样做跨集匹配）：

```bash
python -m dai_detect files episode1.mp3 episode2.mp3 --report report.json
```

单集只能做声学候选：

```bash
python -m dai_detect file episode.mp3 --report report.json
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-n, --episodes` | 3 | 分析最新几集，0 表示全部 |
| `--min-run` | 8 | 重复区间最短秒数 |
| `--min-score` | 0.5 | 最终候选最低分 |
| `--report` | 无 | JSON 报告路径 |
| `--cache-dir` | `.dai-cache` | 特征缓存目录（不含音频） |
| `--fresh` | 关 | 忽略缓存重新拉流（DAI 内容可能逐次变化） |
| `--no-cache` | 关 | 完全不读写缓存 |
| `--proxy` | 无 | 例如 `http://127.0.0.1:7890` |
| `--quiet` | 关 | 不打印进度 |

报告输出版本、配置、每集时长与候选列表。候选字段：

- `start_s` / `end_s`：广告区间（秒）；`start` / `end`：时钟格式
- `score`：0–1，跨集重复候选通常 ≥ 0.7
- `tier`：`repeated`（跨集指纹命中）或 `acoustic`（仅声学候选）
- `evidence`：音乐性、响度偏差、边界强度、位置先验、匹配票数
- `matches`：匹配到的其他剧集与时间区间

## GitHub Actions

仓库内置 `.github/workflows/detect.yml`：

- push 到主分支：跑单元测试（不访问网络）
- 手动触发 `dai-detect`：输入 RSS 地址和集数，在 US runner 上拉流分析，报告上传为 `dai-report-<run>` artifact

使用方式：Actions → dai-detect → Run workflow，填写 `feed_url`。运行结束后在该次 run 的 Artifacts 里下载 `report.json` 和 `run.log`。

## 局限

- 只出现一次、或只在同一集出现的广告无法用跨集指纹发现，只能依赖声学候选。
- 主播口播广告（host-read）不是预录音频，重复指纹无效；可后续加 ASR + 话术分类。
- 共享片头/片尾/主题音乐也会被标为重复，报告里需结合位置与时长判断。
- 同一广告在不同请求间可能被替换，`--fresh` 才能拿到当前版本；缓存的是特征，不是音频。
- 声学候选阈值是启发式，没有标注集时建议把 `acoustic` 当作人工复核清单。
- 目前只做离线分析；实时跳过需要把特征/匹配器接入播放链路并维护滚动索引。

## 术语

- **DAI**：Dynamic Ad Insertion，服务端按请求动态插入广告。
- **指纹（fingerprint）**：频谱峰值对哈希，用于判断两段音频是否同一录音。
- **偏移投票**：同一录音在不同集出现时间不同，匹配对的时间差会聚成一个峰。
- **前贴片 / 中插 / 后贴片**：pre-roll / mid-roll / post-roll。
- **候选**：带证据的疑似广告区间，不等同于已确认广告。

## 测试

```bash
python -m unittest discover -s tests -t . -v
```

测试包含合成音频端到端：两集在不同位置插入同一段 20 秒广告，断言两集都能检出、边界误差 ≤ 3 秒，且无广告的第三集不产生重复候选。
