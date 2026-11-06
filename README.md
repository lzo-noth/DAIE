# DAI 广告检测原型

从流式播客音频中识别 DAI（Dynamic Ad Insertion，动态广告插入）广告区间。
主路线是单集音频盲检测（只读这一份音频）；RSS/跨集模式保留为辅助诊断。
输入播客 RSS 地址，输出每集音频里疑似广告的起止时间、评分与证据，报告可作为 GitHub Actions artifact 下载。

当前是离线分析原型：音频经 ffmpeg 流式解码，特征与指纹按块处理，不落盘原始音频。

## 单音频盲检测（主路线）

正式目标是 **Single-Recording, Transcript-Free Blind DAI Detection**：

> 输入一集已经过 DAI 动态广告插入的完整播客音频，仅凭这一份音频，检测第三方预录广告的开始、结束、时长和广告位类型。运行时不请求原始音频、其他剧集或任何文字稿，也不做全集 ASR 或逐集人工标注。

完整规格、架构和验收指标见 [SPEC.md](SPEC.md)。运行时链路：

```
48 kHz 立体声（取证） + 16 kHz 单声道（声学）
  -> 1.0 秒窗口 / 0.5 秒步长的窗口特征 + 2 秒 / 4 秒上下文特征
  -> 逐窗广告概率（学习式逻辑回归；无模型时启发式兜底）
  -> 时序解码（平滑、滞回、间隙合并、时长约束）
  -> onset / offset 双头边界模型 + 10 ms 边界精修
  -> 时间轴 JSON
```

训练产物可直接用于同一工作流的盲测：`train` job 上传模型 artifact，`blind-test` job 通过 `needs` 下载并使用。训练模式要求两个模型都存在，否则 job 失败。非训练模式的 `model_path` / `boundary_model_path` 默认为空：留空时不向检测器传模型参数，明确使用启发式兜底；填写时必须能成功加载，否则报错。分类器和边界模型各自记录 `model_version`、`data_version`（标注与音频身份的哈希）和 `git_commit`，时间轴会输出两份元信息；严格模式要求两者 `data_version` 非空且一致。盲测 artifact 同时包含归档的原始音频快照，便于之后人工标注和复现。

单集检测：

```bash
python -m dai_detect single episode.mp3 --report timeline.json
python -m dai_detect single "https://example.com/episode.mp3" --report timeline.json
```

输出示例：

```json
{
  "mode": "single_recording_blind",
  "duration_s": 3600.0,
  "ads": [
    {"start": 1200.42, "end": 1231.08, "confidence": 0.96,
     "type": "third_party_inserted", "slot": "mid_roll"}
  ]
}
```

训练（至少需要两个独立节目；交叉验证为按 feed 的留一验证）：

```bash
python scripts/train_single_episode.py \
  --annotations datasets/annotations.jsonl \
  --model-out models/ad_segment_model.json \
  --boundary-out models/boundary_model.json
```

训练输出广告分类模型和 onset/offset 双头边界模型。阈值在每折训练集内部再做分组交叉验证选择，留出节目只参与最终评估；内层测试折可以是全正、全负或混合，不会被跳过。阈值优化目标默认是**内层留出集解码后的事件 F1**（同时参考每小时误报），只有没有事件标注时才回退到窗口 F1。事件 F1 按 episode 独立匹配后汇总 TP/FP/FN，绝不把不同剧集的时间轴放进同一个匹配器。分类阈值与 onset/offset 阈值交替校准：每轮先用当前边界阈值选分类阈值，再用该分类阈值解码出的片段重选边界阈值，直到变化不超过 0.01 或达到 5 轮，历史写入 `calibration.history`。未收敛时会用最终边界阈值再校准一次分类阈值，并重跑完整链路生成 `final_calibration_metrics`，不会把最后一轮的中间结果直接当作部署参数。边界阈值按最终边界误差 P95 校准，但只在满足相对覆盖率（最优候选的 80%）、绝对最低事件召回（默认 0.80，可用 `--min-boundary-event-recall` 调整）和每小时误报上限（默认 15，可用 `--max-boundary-fp-per-hour` 调整，`<=0` 关闭）的候选里比较 P95；没有候选达标时标记 `boundary_calibration_valid=false`，不会把该 P95 当作有效校准。部署结果还给出组合级 `calibration_process_valid`、`converged`、`event_threshold_valid`、`onset_threshold_valid`、`offset_threshold_valid` 和 `final_pipeline_valid`；未收敛或任一项不达标时 `valid=false`，模型仍可写出，但不能宣称已通过校准验收。`performance_targets_met` 在训练阶段固定为 `null`，只能由独立盲测写入，避免把校准过程有效误当成产品指标达标。候选阈值先粗搜（0.2–0.8，步长 0.1）再在最优点附近细搜（步长 0.01）。窗口和边界头的 P/R/F1 使用各外层折自己的阈值；AUC 同时报告 fold-wise 宏平均和 pooled 值，主 AUC 没有有效混合折时为 null；最终部署阈值和对应校准结果单独放在 `deployment_calibration`。没有广告的留出集仍输出误报数和每小时误报；边界校准会同时记录 `positive_boundary_episodes` 和 `negative_control_episodes`，无广告集参与完整流水线覆盖和误报检查，但不进入边界 P95 分母。需要边界模型时，外层每折的学习型边界头会绑定到该折所有留出集（包括确认无广告的集）；无广告集产生误报时也使用学习型精修，不允许回退启发式。每折分别统计分类器、onset、offset 和完整流水线的有效折数；只有 3 个及以上独立 feed 且所有内层流水线折都完成（分类器与学习型边界头均训练并产生预测）时，`cv_quality` 才会标记为 `strict_nested_grouped_cv`，否则为 `limited_nested_cv`。

训练要求边界模型时，如果最终没有写出 `boundary_model.json`，脚本返回非零退出码；只有显式 `--skip-boundary` 或弱监督模式允许跳过边界模型。训练结束会打印 `artifacts` 摘要。

训练评估与实际推理共用 [decode_config.py](dai_detect/decode_config.py) 的 enter/exit 阈值规则。阈值选择在内层留出集上运行完整的 decode → refine → evaluate 流程；onset/offset 阈值按最终边界误差 P95 校准，而不是帧级 F1。训练模式下 workflow 强制要求分类模型和边界模型都成功下载；任一缺失都会让 job 失败，不会静默回退到启发式。`--require-models` 还要求两个模型都有非空 `data_version` 且完全一致。`--require-calibrated-models`（训练 workflow 默认启用）进一步要求两个模型的 `deployment_calibration.calibration_process_valid`/`final_pipeline_valid` 为 true、校准已收敛，且 onset/offset 阈值一致；调试时可用 workflow 输入 `allow_uncalibrated_models=true` 显式放行，但程序仍会打印校准无效警告。时间轴同时记录分类器和边界模型的 `version`、`data_version`、`git_commit`。

标注 JSONL 每行一集：

```json
{"audio": "episodes/ep001.mp3", "feed": "My Show", "episode": "Ep 1", "ads": [{"start_s": 1200.4, "end_s": 1231.1}]}
```

也可以用 LLM 候选弱监督（需要音频清单，格式 `{"feed": "...", "episode": "...", "audio": "..."}`）：

```bash
python scripts/train_single_episode.py \
  --labels feedback/labels.jsonl --audio-manifest datasets/audio_manifest.jsonl
```

弱监督只报告窗口级指标，不报告事件级指标；LLM 标签不是人工真值。

盲测（先检测、后读真值）：

```bash
python scripts/blind_test.py --audio episode.mp3 \
  --ground-truth ground_truth/episode.json \
  --out timeline.json --metrics metrics.json
```

盲测脚本只有在时间轴生成后才读取真值，并按事件 IoU ≥ 0.3 做最大基数匹配（再最大化总 IoU），而不是贪心匹配，输出事件 precision/recall、边界误差 P95 和每小时误报。空真值集仍统计误报数和每小时误报，此时 recall 记为未定义。验收目标：事件 precision ≥ 0.90、recall ≥ 0.90、边界 P95 ≤ 0.5 s。没有人工真值时检测器只输出时间轴，不宣称达标。

### 怎么保证标注音频和训练/评测音频是同一份

DAI 会在同一个 episode URL 背后按请求替换广告，因此 **URL 和文件名不能作为音频身份**。同一集两次下载可能是两个不同变体，标注时间轴只在被标注的那一份里有效。

本仓库用三级身份绑定：

| 标识 | 定义 | 作用 |
| --- | --- | --- |
| `source_sha256` | 原始字节的 SHA-256 | 证明是同一个文件（最严格） |
| `audio_id` | 解码后 16 kHz PCM（16-bit 量化）的 SHA-256 | 证明是同一条录音内容；DAI 变体不同则一定不同 |
| `audio_id_kind` | 生成 `audio_id` 的解码路径 | 防止跨解码路径误比 |

标注前必须先把这一份音频归档：

```bash
python scripts/snapshot_audio.py \
  --source "https://example.com/episode.mp3" \
  --feed "My Show" --episode "Ep 1" \
  --manifest datasets/audio_manifest.jsonl
```

脚本会把原始字节按 `<sha256>.<ext>` 存到 `audio_snapshots/`，并输出/追加一行清单：

```json
{"audio": "audio_snapshots/<sha256>.mp3", "source_sha256": "<sha256>",
 "audio_id": "<pcm sha256>", "audio_id_kind": "downmix_48k_to_16k", "duration_s": 3600.0}
```

之后所有标注、训练、评测都只针对这个快照文件；真值 JSON 里同时写入 `source_sha256` 和 `audio_id`。

强制校验行为：

- `scripts/train_single_episode.py` 默认拒绝缺少 `source_sha256`/`audio_id` 的标注；字节或 PCM 不匹配直接中止训练。调试时才用 `--allow-unverified-audio` / `--allow-audio-mismatch`。
- `scripts/collect_labels.py` 会把报告里的 `audio_id`、`audio_id_kind`、`source_sha256` 一起写入 `feedback/labels.jsonl`。弱监督训练按 `audio_id` 分组，旧的无身份标签会被跳过，避免把不同变体的标签混用。
- `scripts/blind_test.py` 在生成时间轴后才校验真值中的身份；真值缺少 `audio_id`/`audio_id_kind`（本地音频还缺 `source_sha256`）时同样拒绝评分并返回 exit 3。`--allow-audio-mismatch` 只覆盖“身份不一致”，不能覆盖“身份缺失”；后者只能用 `--allow-unverified-ground-truth` 在调试时显式跳过。
- 单集时间轴和检测报告都会输出 `audio.id` / `audio.source_sha256`。
- 旧版 `platforms` 循环是流式分析 URL，没有归档被标注的那一份音频；因此它产出的 LLM 标签没有可验证的音频身份，不能直接用于新模型。要使用弱监督，必须先用 `snapshot_audio.py` 归档音频，再在该快照上产生标签。

训练期音频增强（多次重编码、响度归一化、混响/底噪、音乐床、交叉淡化）可以通过在标注文件里为同一节目追加增强副本实现；这些副本必须与原始集放在同一个 feed 分组，避免泄漏。

## 原理（旧版跨集分析，保留）

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
| `--clips-dir` | 无 | 为每个候选生成试听 MP3（会强制重新拉流） |
| `--clip-seconds` | 10 | 每个试听片段的秒数；Actions 用 30 秒以便 ASR 抓到完整广告口播 |
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

### 自我改进闭环（GitHub Actions）

`platforms.yml` 是完整的反馈闭环，每天自动跑一轮，也可以手动触发：

1. **检测**：按 `platforms.json` 矩阵分析每个 feed 的最新集，生成候选和试听片段。
2. **标注**：`faster-whisper` 转写候选片段；配置 `GEMINI_API_KEY`（或 `OPENAI_API_KEY`）时，用 LLM 判定广告/正片。
3. **回流**：`scripts/collect_labels.py` 把所有 `llm_labels`（包括被删除的负例）去重写入 `feedback/labels.jsonl`；`scripts/mine_keywords.py` 从广告转写和 LLM 品牌字段中挖出赞助话术/品牌词，写入 `feedback/keywords.json`。Actions 生成 30 秒候选片段，ASR 能覆盖完整广告口播；关键词按词边界匹配，避免 `revolut` 命中 `revolution` 这类子串误报。
4. **优化**：`scripts/fit_calibration.py` 按 feed 做留一交叉验证，拟合候选评分器，输出 `calibration.json` 和指标；下一次检测直接加载。`scripts/asr_clips.py` 也会加载 `feedback/keywords.json`，把学到的品牌词作为确定性广告话术特征。
5. **循环**：`learn` job 自动提交 `feedback/` 与 `calibration.json`（提交信息带 `[skip ci]`），下一轮（每天定时或手动）使用新模型。`feedback/state.json` 记录轮数和指标；达到目标后写入 `feedback/COMPLETE`，`gate` job 之后不再执行分析。默认最多 10 轮，可用 `max_rounds` 调整。

默认目标（手动 dispatch 时可覆盖）：`recall@40 ≥ 0.95`、`candidate recall ≥ 0.8`、`AUC ≥ 0.8`、送复核候选 ≤ 30/小时、预复核误报 ≤ 15/小时。此外，**必须有一份独立人工真值**，事件 precision ≥ 0.9、事件 recall ≥ 0.9、起止误差 P95 ≤ 0.5 秒，才会写 `COMPLETE`。没有 `ground_truth/*.json` 时循环不会宣布完成——LLM 标签只用于候选覆盖和成本控制，不作为真值。

LLM 在这里承担**推理时复核**和离线标注：检测候选生成仍走声学/指纹，最终把候选片段交给 LLM 判定第三方广告并剔除正片、自宣和主播口播。关键词和校准模型是确定性产物，用于保证候选召回并控制 LLM 调用量。

人工真值格式（`ground_truth/<episode>.json`，同一集的多个下载版本必须放在同一数据分组）：

```json
{
  "episode": "Episode 001",
  "episode_id": "abc123",
  "ads": [
    {"start_s": 1200.4, "end_s": 1231.1, "type": "third_party_inserted"},
    {"start_s": 2450.0, "end_s": 2480.0, "type": "third_party_inserted"}
  ]
}
```

`scripts/evaluate_ground_truth.py` 会按候选区间与真值区间的重叠做事件匹配，输出事件 precision/recall、起止误差 P95 和 IoU；`update_loop_state.py` 用这些指标判定 `COMPLETE`。

反馈文件：

- `feedback/labels.jsonl`：去重后的标注样本，包含声学特征、tier、ASR 文本、关键词、LLM 判定。
- `feedback/keywords.json`：从广告样本中挖出的赞助话术与品牌词。
- `feedback/state.json`：当前轮数、目标、指标、是否完成。
- `feedback/COMPLETE`：达到目标后的停止标记。
- `feedback/STOPPED`：达到最大轮数仍未达标时的停止标记。

首次启用：Actions → platforms → Run workflow；确保仓库 Secrets 里有 `GEMINI_API_KEY`。`learn` job 需要 `contents: write` 权限；如果分支保护禁止 Actions 直接 push，校准和标签仍会作为 `dai-learn` artifact 上传，可手动下载后提交。删除 `feedback/COMPLETE` 或 `feedback/STOPPED` 并 push 会自动触发新一轮（workflow 的 push paths 包含这两个停止标记）；`round` 计数会重置为 0。`platforms.json` 里标了 `optional: true` 的 feed（当前是 Substack）失败不会把整轮 run 标红。

### LLM 反馈闭环（手动方式）

1. 每次 LLM 分类会把**所有候选（包括被删除的负例）**写入 `report.llm_labels`，包含声学特征与 LLM 判定。
2. 离线运行 `python scripts/fit_calibration.py <报告目录>`：按 feed 做留一交叉验证，拟合候选评分器。
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

- 单音频模式没有训练模型时只能走启发式兜底，结果不满足验收指标；必须先积累按节目分组的标注。
- 如果广告与正文的制作风格、房间声学和响度完全一致，纯声学方法无法保证识别；这是信息上限。
- 边界精修依赖可检测的制作变化；平滑交叉淡入淡出的广告边界可能偏差更大，0.5 s 的 P95 目标需要真实标注集验收。
- 模型在某个地区/节目分布上训练后，换到差异很大的节目或定向广告时可能迁移不足，需要重新评估。
- 旧版跨集分析：只有拼接边界、且符合广告先验的单次广告才会以 `inserted` 出现；无音乐、无响度变化、与主播同音色的插入广告可能被当成连续口播而漏报。
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

测试包含合成音频端到端：两集在不同位置插入同一段 20 秒广告，断言两集都能检出、边界误差 ≤ 3 秒；另有单集插入广告（无重复）与连续口播负例。单音频盲检测测试使用 48 kHz 立体声合成节目，断言仅凭单文件检出插入广告、纯语音不误报，并验证时序解码、模型加载和事件级评测。

硬样本测试还覆盖：同响度广告由学习模型检出、正文音乐床不误报、普通剪辑不误报、淡入淡出广告可检出（测试容差 2.5 秒）、MP3 重编码后仍可检出，以及 onset/offset 双头边界模型和精修的单元测试。
