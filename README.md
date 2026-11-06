# DAI 广告检测原型

从流式播客音频中识别 DAI（Dynamic Ad Insertion，动态广告插入）广告区间。
输入播客 RSS 地址，输出每集音频里疑似广告的起止时间、评分与证据，报告可作为 GitHub Actions artifact 下载。

当前是离线分析原型：音频经 ffmpeg 流式解码，特征与指纹按块处理，不落盘原始音频。

## 原理

DAI 广告由服务端在请求音频时插入，因此有两个可利用的稳定特征：

1. **跨集重复指纹**：同一条广告创意通常会在同一播客的多集、不同位置重复出现，而节目正文不会逐字重复。对音频提取 Wang 风格频谱峰值指纹，做跨集偏移投票，重复区间即为强候选。
2. **插入段检测（覆盖单次广告）**：广告是独立文件拼接进来的，边界处通常出现静音、硬切、响度阶跃和音色/房间底噪变化。在 100 Hz 细粒度特征上定位拼接点，再按广告先验（时长、前/中/后贴片位置、音乐床、响度偏差、两端音色变化）给片段打分。两端都没有拼接特征、且与前后语音同音色的连续口播会被降分，因此不把主播口播广告算作目标。

两者合并后按证据打分：重复匹配置信度、拼接边界强度、音乐性、响度偏差、位置先验。
没有标注数据时，`repeated` 是强证据；`inserted` 是声学候选，需要人工复核。

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

单集只能做插入段检测（没有跨集重复可比）：

```bash
python -m dai_detect file episode.mp3 --report report.json
```

验证平台 feed（只抓 RSS，不下载音频）：

```bash
python -m dai_detect verify platforms.json --report feed_verification.json
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-n, --episodes` | 3 | 分析最新几集，0 表示全部 |
| `--min-run` | 8 | 重复区间最短秒数 |
| `--min-score` | 0.5 | 最终候选最低分 |
| `--report` | 无 | JSON 报告路径 |
| `--cache-dir` | `.dai-cache` | 特征缓存目录（不含音频） |
| `--clips-dir` | 无 | 为每个候选生成 10 秒试听 MP3（会强制重新拉流） |
| `--fresh` | 关 | 忽略缓存重新拉流（DAI 内容可能逐次变化） |
| `--no-cache` | 关 | 完全不读写缓存 |
| `--proxy` | 无 | 例如 `http://127.0.0.1:7890` |
| `--quiet` | 关 | 不打印进度 |

报告输出版本、配置、每集时长与候选列表。候选字段：

- `start_s` / `end_s`：广告区间（秒）；`start` / `end`：时钟格式
- `score`：0–1，跨集重复候选通常 ≥ 0.7
- `tier`：`repeated`（跨集指纹命中）或 `inserted`（拼接边界 + 广告先验）
- `evidence`：边界强度/类型、音色变化、音乐性、响度偏差、位置先验、匹配票数；Actions 里还会附 `slot_matches`、`clip`、`asr_text`、`ad_language`、`llm`/`llm_verified`
- `matches`：匹配到的其他剧集与时间区间

`inserted` 只覆盖预录音频插入的广告；主播口播广告不会被刻意检出。跨集广告位一致（`slot_matches`）用于识别“同一广告位、不同 creative”。

Actions 配置了 `GEMINI_API_KEY`（或 `OPENAI_API_KEY`）后，会对候选 ASR 文本做 LLM 分类：第三方广告加权并标记 `llm_verified`，正片、自宣和主播口播默认从报告中移除；重复候选若被 LLM 高置信判为正片也会删除（保留可用 `--keep-repeated-content`）。每集最多生成 40 个试听片段，确保候选都能进入 ASR/LLM。未配置 key 时只保留重复指纹、slot 和关键词证据。

LLM 标注也会反馈到检测器：当前样本里 slot 命中主要出现在正片（21/27），timbre 在正片反而更高，因此 slot 只对已有音乐/响度证据的候选小幅加分，timbre 不再作为广告正证据。`scripts/calibrate.py` 可从带 `llm` 字段的报告重算这些统计，输出 `calibration.json`；声学门控 precision 约 0.23，LLM 仍是主要判别层。

### LLM 反馈闭环

1. 每次 LLM 分类会把**所有候选（包括被删除的负例）**写入 `report.llm_labels`，包含声学特征与 LLM 判定。
2. 离线运行 `python scripts/fit_calibration.py <报告目录>`：按 feed 做留一交叉验证，拟合 8 维 logistic 候选评分器；只有 CV 指标优于启发式评分时才写 `enabled: true` 的 `calibration.json`。
3. 检测器运行时自动读取仓库根目录的 `calibration.json`，对 `inserted` 候选计算 `learned_ad_score`，低于阈值的候选直接过滤；`repeated` 候选保留。
4. 循环：矩阵 run → 下载报告 → fit_calibration → 提交 calibration.json → 再跑一轮验证。

没有可用 calibration 或未启用时，检测器保持原有的启发式评分。

## GitHub Actions

仓库内置两个工作流：

- `.github/workflows/detect.yml`：push 跑单元测试；手动触发单个 RSS 分析，报告上传为 `dai-report-<run>` artifact。
- `.github/workflows/platforms.yml`：手动触发平台矩阵。按 `platforms.json` 的每个 feed 并行分析（每个 feed 2 集），上传：
  - `feed-verification`：每个平台 feed 的可达性、标题、集数、最新一集（只抓 RSS）
  - `dai-<slug>`：单个 feed 的 `report.json`、`run.log` 和候选 10 秒试听片段 `clips/*.mp3`
  - `dai-matrix-summary`：所有平台的汇总 JSON / Markdown（含 slot-confirmed、ASR ad-language 计数）

矩阵任务会在候选片段上跑 `faster-whisper`（`base.en`），把“sponsored by / use code / 优惠码”等话术写入 `evidence.asr_text` 与 `evidence.ad_language`；若配置了 `GEMINI_API_KEY` 或 `OPENAI_API_KEY`，再用 `scripts/classify_ads.py` 调用 LLM 做广告/正片分类，默认丢弃被判定为正片、自宣或主播口播的插入候选。

启用 LLM：GitHub 仓库 Settings → Secrets and variables → Actions → New repository secret，添加 `GEMINI_API_KEY`（默认使用 Gemini 的 OpenAI 兼容端点与 `gemini-3.5-flash-lite`，可用 `GEMINI_MODEL` 覆盖）。没有 key 时矩阵照常运行，只是不做 LLM 分类。

用法：Actions → platforms → Run workflow，可填 `platform` 只跑某个平台。所有联网验证都在 runner 上完成；本机网络结果不作数。

## 局限

- 只有拼接边界、且符合广告先验的单次广告才会以 `inserted` 出现；无音乐、无响度变化、与主播同音色的插入广告可能被当成连续口播而漏报。
- 片头/片尾/章节音乐也可能被标为 `inserted`，需要结合位置、时长和重复性复核。
- 主播口播广告（host-read）是节目音频的一部分，按目标不检出。
- Spotify SAI 等 App 内插入不在 RSS/下载文件里，feed 分析看不到；需要设备播放捕获。
- 共享片头/片尾/主题音乐会被标为重复，报告里需结合位置与时长判断。
- 同一广告在不同请求间可能被替换，`--fresh` 才能拿到当前版本；缓存的是特征，不是音频。
- 平台/地区定向会改变广告内容：Actions runner 在美国，拿到的是美国定向广告。
- LLM 分类依赖 ASR 文本质量，可能把品牌型广告误判成正片；`clips/*.mp3` 仍保留，供人工复核。LLM 调用会产生少量 API 费用。
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

测试包含合成音频端到端：两集在不同位置插入同一段 20 秒广告，断言两集都能检出、边界误差 ≤ 3 秒；另有单集插入广告（无重复）与连续口播负例。
