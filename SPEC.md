# DAIE 规格：单音频盲检测（Single-Recording Blind DAI Detection）

## 1. 目标定义

**输入**：一集已经过 DAI（Dynamic Ad Insertion）处理的完整播客音频。

**输出**：该集内第三方预录广告的起止时间、时长、置信度和广告位类型。

**运行时不依赖任何外部信息**：

- 不请求该集的原始/无广告版本；
- 不请求同一节目的其他剧集；
- 不读取任何文字稿（Transcript），不做全集 ASR；
- 不依赖逐集人工提供的参考素材、关键词表或品牌库；
- 不联网查询广告库。

唯一允许的运行时输入是这一份音频文件本身（本地路径或音频 URL）。

训练期可以使用其他音频、人工标注、LLM 弱标注和曝光增强数据；但这些内容不得进入部署时的推理路径。模型文件（`models/*.json`）属于程序的一部分，可以随仓库分发。

## 2. 非目标

- 不检测主播口播（host-read）广告：它与正文同源，按当前产品目标排除。
- 不检测 App 内插入（如 Spotify SAI）：这些广告不在 RSS/下载文件里。
- 不做实时跳过，仅做离线分析。
- 不追求识别广告主品牌；只判断“这一段是否是第三方预录广告插入”。

## 3. 技术路线

核心判断：**DAI 广告检测 ≠ 音频拼接检测**。广告可以被平滑混入，节目正文也存在正常剪辑点。因此系统同时使用四类信息，并由学习模型融合：

| 信息 | 作用 |
| --- | --- |
| 拼接/编码取证 | 定位独立插入音频的进入/退出点，识别重编码、噪声底和立体声变化 |
| 声学环境 | 录音房间、混音、响度、背景音乐和声道结构的变化 |
| 广告制作风格 | 音乐床、压缩/限幅、音色、语速与节目正文的生产方式差异 |
| 时间上下文 | 广告通常成段出现，有典型时长和位置先验，普通剪辑点不具备同样的持续结构 |

架构（全部运行时可离线执行，纯 NumPy）：

```
单集音频（MP3/AAC/WAV/URL）
  └─ audio.py        48 kHz 立体声解码（取证） + 3:1 下混 16 kHz 单声道（声学模型）
      ├─ forensic.py        拼接/编码取证帧（10 ms）
      └─ features.py        16 kHz 声学帧（10 ms，Log-Mel/MFCC/响度/频谱平坦度/通量）
          └─ single_features.py   1.0 s 窗口 / 0.5 s 步长 + 2 s / 4 s 上下文特征
              ├─ ad_classifier.py      逐窗广告概率（学习式；无模型时启发式兜底）
              ├─ temporal_decoder.py   时序解码：平滑、滞回、合并、时长约束
              ├─ boundary_model.py     帧级 onset / offset 双头边界模型
              └─ refinement.py         在 ±2.5 s 内把起止精修到帧级
                  └─ 时间轴 JSON
```

### 3.1 音频层

- 取证层：保留原始双声道，48 kHz 解码，保留 12–20 kHz 高频残差、立体声相关性和编码截止特征。
- 模型层：由同一解码流下混为 16 kHz 单声道，供声学特征使用。
- 只解码一次，两个视图由同一 PCM 流派生，避免两次拉流导致内容不一致。

### 3.2 取证与声学特征

- 帧级：响度、频谱平坦度、频谱通量、MFCC、频谱熵、95% 滚降频率、高频能量占比、高频底噪、立体声 L/R 相关、Side/Mid 能量比、削波比例、相位跳变、16 个对数频带能量。
- 窗口级：上述统计量的均值/标准差/低分位、相对全集基线的偏移（响度、音乐性、高频、相关、宽度、噪声底、频带轮廓距离）、相邻窗口差分、MFCC 偏移。
- 上下文级：同一时间点的 2 s 与 4 s 窗口统计，用于判断持续制作风格差异，而不是只看 1 s 局部变化。
- 明确不以“有音乐/音量变大/换说话人”单独判定广告，这些只是特征。

### 3.3 学习模型（第一阶段为纯 NumPy）

- 广告分类器：标准化 + 逻辑回归（`models/ad_segment_model.json`），特征名与顺序写入模型文件，运行时按名对齐。
- 边界模型：onset 与 offset 两个独立逻辑回归头，特征同时包含绝对变化量和带方向的有符号变化量（`models/boundary_model.json`）；旧版单头模型仍可加载。
- 边界精修：用 onset 头搜开始、offset 头搜结束；先对约 0.2 s 的取证分数做平滑以覆盖淡入淡出，再在 ±0.35 s 内回到原始峰值；内部变化强度显著高于两端时拒绝这组边界。
- 时间解码：滑动中值/均值平滑后做滞回状态机，合并短间隙、过滤过短事件、拆分超长事件。
- 模型文件记录 `model_version`、`data_version`（标注与音频身份的哈希）和 `git_commit`；时间轴分别输出分类器和边界模型的元信息。严格模式要求两者 `data_version` 非空且完全一致。
- 运行模式显式区分：`trained`（两个模型都已加载并通过版本校验）、`heuristic`（调试兜底）、`invalid`（模型缺失/损坏/版本冲突时报错退出）。`--require-models` 或工作流训练模式必须走 `trained`，不得静默降级。
- 预留升级位：可用预训练音频编码器（PANNs/AST/WavLM/HuBERT）+ TCN/BiGRU 替换分类器，只要保持窗口特征接口和“单音频输入”约束即可。

### 3.4 训练与弱监督

- 训练数据来自人工标注（`datasets/annotations.jsonl`）和 LLM 候选标注（`feedback/labels.jsonl` + 音频清单）。
- 数据划分必须按节目（feed）分组，禁止同一节目跨训练/测试集。分组交叉验证为留一节目：每折的测试集是该节目的全部剧集，训练集是其余节目；不足两个节目时训练直接拒绝，避免过于乐观的单节目评测。
- 阈值选择采用嵌套分组交叉验证：外层留出节目只参与最终评估，阈值只用外层训练集内部的内层折选择；内层训练折必须同时包含正负样本，但内层测试折允许全正、全负或混合。外层训练集只有一个 feed 时退化为按剧集拆分，并标记为 `within_feed_fallback`，不报告成严格分组验证。
- 阈值优化目标为内层留出剧集**完整 decode → refine → evaluate** 后的事件 F1（并参考每小时误报），而不是窗口 F1；onset/offset 阈值按最终边界误差 P95 校准，帧级 F1 只作辅助诊断。没有事件标注时才回退窗口 F1。
- 分类阈值与边界阈值必须交替校准，不能各自独立选择：每轮先用当前边界阈值选分类阈值，再用该分类阈值解码出的片段选 onset/offset 阈值，如此循环，直到各阈值变化不超过 `THRESHOLD_TOLERANCE`（0.01）或达到 `MAX_CALIBRATION_ROUNDS`（5）；每轮结果写入 `calibration.history`。达到最大轮数仍未收敛时，必须用最终 onset/offset 阈值再校准一次分类阈值。无论是否收敛，最终三阈值都要固定后重新执行一次完整的内层 OOF `decode → refine → evaluate`，`final_calibration_metrics` 和部署验收只使用这次联合结果；最后一轮中间指标不能代替它。
- 边界阈值坐标下降结束后，也必须用最终 onset/offset 组合重新计算一次 P95、Recall、FP/hour，不能把分别在不同另一侧阈值下测得的 onset/offset 结果拼成最终校准记录。
- 事件分类阈值同样必须有独立的 FP/hour 硬约束（默认 15，可用 `--max-event-fp-per-hour` 配置）；最终三阈值联合验证时超过上限即 `final_pipeline_valid=false`。
- 边界搜索先粗后细；P95 只在事件匹配覆盖率不低于最优候选 `min_recall_ratio`（默认 0.8）、达到绝对最低事件召回 `min_event_recall`（默认 0.80，可用 `--min-boundary-event-recall` 配置）且每小时误报不超过 `max_false_positives_per_hour`（默认 15，可用 `--max-boundary-fp-per-hour` 配置，`<=0` 关闭）的候选之间比较。没有候选达到绝对召回或误报上限时分别标记 `recall_valid=false` / `false_positive_rate_valid=false`，整体 `boundary_calibration_valid=false`，不能把该 P95 当作有效校准。解码不出事件的留出集仍计入真值事件分母。
- 训练评估和实际推理共用同一 enter/exit 阈值函数。交叉验证的窗口与边界 P/R/F1 使用各外层折自己的阈值；AUC 同时报告 fold-wise 宏平均与 pooled 值；主 AUC 使用宏平均，没有有效混合折时为 null，pooled 仅作诊断。最终部署阈值及对应校准结果单独记录为 `deployment_calibration`，不得用于回算交叉验证指标。
- 校准覆盖信息必须分别记录 `classifier_folds_completed`、`onset_folds_completed`、`offset_folds_completed`、`pipeline_folds_completed` 与 `folds_attempted`；跳过折不能算作已完成折。完整流水线折指分类器与 onset/offset 头均训练成功，并对该折**全部有标签留出集**（包括 `ads=[]` 的无广告节目）完成预测。另需分别记录 `positive_boundary_episodes` 与 `negative_control_episodes`：无广告节目没有真实边界误差，不能进入 P95 分母，但必须参与边界精修和误报检查。需要边界模型时，学习型边界头缺失不得静默回退启发式精修，该折不计入完整流水线覆盖。
- 最终部署参数必须有一个组合级验收记录，至少包含 `valid`、`calibration_process_valid`、`deployment_eligible`、`cv_quality`、`converged`、`event_threshold_valid`、`event_false_positive_rate_valid`、`final_coverage_valid`、`onset_threshold_valid`、`offset_threshold_valid`、`final_pipeline_valid` 和 `final_recalibrated`。这些字段必须基于最终返回的参数组合重新运行完整检测链路，而不是拼接各参数单独选择时的结果；`deployment_eligible` 还要求 `cv_quality=strict_nested_grouped_cv`。`valid=false` 时不禁止写出模型，但禁止宣称校准验收通过。
- `calibration_process_valid` 只表示阈值校准流程有效；产品性能目标必须单独由独立盲测判定。训练输出 `performance_targets_met=null`、`performance_targets_source="independent_blind_test_required"`；盲测脚本按事件 precision ≥0.90、recall ≥0.90、边界 P95 ≤0.50 s 写入 `performance_targets_met`。
- 严格部署模式必须支持 `--require-calibrated-models`：分类模型和边界模型都要求 `calibration_process_valid`、`deployment_eligible`、`final_pipeline_valid`、`converged` 为 true，`cv_quality=strict_nested_grouped_cv`，且两个模型的 onset/offset 阈值一致；任一条件失败时拒绝部署。训练 workflow 默认启用该检查，只有显式 `allow_uncalibrated_models=true` 的调试运行可以放行，并必须保留警告。
- 模型文件必须保存 `training_identity`（训练用过的 `audio_ids`、`source_sha256`、`feeds`）。盲测必须用 `--feed` 检查输入 feed，并把 `same_feed_holdout`、`unseen_feed_blind_test` 和 `unknown_feed` 区分开；输入音频的 audio_id/source_sha256 命中训练集，或 feed 命中训练集时，默认拒绝独立验收。`performance_targets_met` 只允许在未见 feed 且音频未参与训练时为 true。
- 训练入口必须拒绝同一 audio_id/source_sha256 跨多个 feed、重复的 feed/episode 标识、广告区间 start≥end、start<0、end 超出音频时长以及区间重叠；调试覆盖必须显式使用 `--allow-cross-feed-duplicates` 或 `--allow-invalid-annotations`。
- 完整监督标注可以携带 `type`/`ad_type`；启用 `--require-ad-type` 时只接受 `third_party_inserted`。没有该字段时，输出中的 `third_party_inserted` 只是未经验证的假设，不能当作已确认标签。
- 当前特征使用整集统计基线和未来上下文，属于离线整集检测架构；若未来支持边播边检测，必须单独建立因果特征与有限 look-ahead 流程，不能直接复用当前离线模型。
- 训练数据少于 3 个独立 feed，或任一内层折的完整流水线未完成时，结果等级标记为 `limited_nested_cv`；只有 3 个及以上 feed 且内层分类器、onset、offset 全部有效才可报告 `strict_nested_grouped_cv`。
- 要求训练边界模型时，未生成边界模型必须返回非零退出码；只有 `--skip-boundary` 或弱监督模式允许缺少边界模型。
- 每个外层训练折必须同时包含正负窗口，否则中止训练；边界 onset/offset 训练折同样独立检查类别。
- 事件级评估必须使用该折模型对**完整窗口序列**的预测，不能把筛选后的窗口概率与完整时间轴混用。不同 episode 的时间戳都从 0 开始，因此**必须逐集调用事件匹配后再汇总 TP/FP/FN 和边界误差 P95**，禁止把多集预测与真值放进同一个匹配器。没有广告的留出集仍统计误报和每小时误报，recall 记为未定义。
- 需要边界模型时，外层留出集的每个 episode（包括确认无广告、但可能产生误报的节目）都必须绑定该折学习型 onset/offset 头；无广告节目出现待精修事件时同样禁止回退启发式。
- 事件匹配使用最大基数二分图匹配，再在最大基数下最大化总 IoU；不使用贪心 IoU 排序。
- 边界评估使用同一折的 onset/offset 边界头对留出集精修，并分别报告两个头的帧级 P/R/AUC 与事件边界误差 P95，而不是只看启发式变化分数。
- 音频增强：多次 MP3/AAC 重编码、响度归一化、混响与底噪叠加、音乐床混合、交叉淡入淡出、不同拼接边界类型。
- 训练脚本输出模型 JSON 与分组交叉验证指标；不把 LLM 标签当人工真值。

## 4. 输出格式

```json
{
  "version": 1,
  "mode": "single_recording_blind",
  "source": "episode.mp3",
  "duration_s": 3600.0,
  "ads": [
    {
      "start": 1200.42,
      "end": 1231.08,
      "start_s": 1200.42,
      "end_s": 1231.08,
      "duration_s": 30.66,
      "confidence": 0.96,
      "type": "third_party_inserted",
      "slot": "mid_roll",
      "evidence": {"mean_probability": 0.94, "boundary_start": 0.81}
    }
  ]
}
```

`slot` 取值：`pre_roll`（开始 < 120 s）、`post_roll`（结束 > 时长 − 150 s）、其余 `mid_roll`。

## 4.5 音频同一性（标注 / 训练 / 评测必须绑定同一份音频）

DAI 会在同一 URL 背后按请求替换广告，因此 URL、文件名、RSS GUID 都不能作为音频身份。同一集两次请求可能是不同变体，广告时间轴只对其中一份有效。

系统用三个标识绑定音频：

| 标识 | 计算方式 | 保证 |
| --- | --- | --- |
| `source_sha256` | 原始字节 SHA-256 | 同一个文件 |
| `audio_id` | 解码后 16 kHz PCM（16-bit 量化）SHA-256 | 同一条录音内容；广告变体不同则必然不同 |
| `audio_id_kind` | 生成 `audio_id` 的解码路径 | 禁止跨路径误比 |

规则：

1. 标注前必须用 `scripts/snapshot_audio.py` 把被标注的那一次音频按字节归档，快照文件名即 `source_sha256`。
2. 标注、训练、盲测只使用快照文件；真值 JSON 必须包含 `source_sha256`、`audio_id` 与 `audio_id_kind`。
3. 训练默认强制校验，缺失或冲突即中止；`--allow-unverified-audio` / `--allow-audio-mismatch` 只用于调试。
4. 盲测在生成时间轴后校验身份：真值必须提供 `audio_id` 与 `audio_id_kind`，本地输入还必须提供 `source_sha256`；身份缺失或冲突都返回非零退出码且不评分。`--allow-audio-mismatch` 不能绕过身份缺失，只能绕过值不一致。
5. 不能从 URL 重新下载替代快照；重新下载得到的可能是另一个广告变体。

## 5. 验收标准

以下为研发目标，不是当前系统已经达到的性能。验收必须使用独立人工真值，且检测器在评测时看不到标注。

| 指标 | 目标 | 说明 |
| --- | --- | --- |
| 广告事件 Precision | ≥ 0.90 | 事件级匹配（IoU ≥ 0.3） |
| 广告事件 Recall | ≥ 0.90 | 同上 |
| 边界误差 P95 | ≤ 0.5 s | 起止边界绝对误差合并统计 |
| 每小时误报 | 单独验收 | 越低越好，不与召回互相抵消 |
| 外部音频依赖 | 0 | 运行时只读目标音频 |
| Transcript 依赖 | 0 | 不调用 ASR/LLM |
| 逐集 ASR | 0 | 同上 |

盲测协议：把完整一集音频交给检测器，隐藏所有人工广告标注；检测器输出时间轴后再用真值评估。盲测脚本 `scripts/blind_test.py` 强制先检测、后评估。

## 6. 已知风险

- 如果广告与正文的制作风格、房间声学和响度完全一致，纯声学方法无法保证 100% 识别；这是问题的信息论上限，不是实现缺陷。
- 无训练数据时只能使用启发式兜底，指标显著低于验收目标；必须先积累分组标注。
- 共享片头/片尾、节目自宣和章节音乐可能产生误报，需由训练数据学习区分。
- 平台/地区定向会改变广告内容；评测集的采集地区必须与使用场景一致。

## 7. 代码地图

| 文件 | 职责 |
| --- | --- |
| `dai_detect/audio.py` | 单次解码、48 kHz 立体声、16 kHz 单声道下混 |
| `dai_detect/features.py` | 16 kHz 帧级声学特征 |
| `dai_detect/forensic.py` | 48 kHz 立体声取证帧 |
| `dai_detect/single_features.py` | 多尺度窗口特征与训练样本构造 |
| `dai_detect/ad_classifier.py` | 窗口广告概率模型 |
| `dai_detect/boundary_model.py` | 帧级边界变化特征与模型 |
| `dai_detect/temporal_decoder.py` | 时序解码 |
| `dai_detect/refinement.py` | 边界精修 |
| `dai_detect/decode_config.py` | 训练评估与推理共用的 enter/exit 阈值 |
| `dai_detect/single_episode.py` | 单音频端到端检测 |
| `dai_detect/evaluation.py` | 事件级最大匹配指标 |
| `scripts/train_single_episode.py` | 训练广告/边界模型 |
| `scripts/blind_test.py` | 盲测与真值评估 |
| `scripts/snapshot_audio.py` | 按字节归档音频快照，产出身份清单 |
| `models/*.json` | 运行时模型 |
| `datasets/annotations.jsonl` | 人工标注清单 |
