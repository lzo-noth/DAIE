# 播客 DAI 地区差异探测报告

- 日期：2026-11-10
- 范围：15 个 DAI 厂商 / 节目；重点对比中国大陆与澳大利亚；辅以美国、德国、法国、日本、韩国、印度、瑞典、意大利、新加坡、香港、俄罗斯等。
- 状态：澳洲、中国大陆、美国等已实测；新西兰、加拿大、英国、爱尔兰、北欧、拉美、东南亚等因缺少可用的媒体出口，尚未验证。

## 1. 结论摘要

- **澳大利亚（AU）是目前唯一确认的 Omny 无 DAI 市场。** Omny Dan Patrick 为 `39,424,560 B / 2452 s / 128.6 kbps`，正好是母带码率；同集国际版为 `52,382,960 B / 170.9 kbps`，美国版约 `52,805,101 B / 172.3 kbps`。
- **中国大陆（CN）不是无 DAI 地区。** Omny 两个节目都拿到国际广告版；Megaphone 也有广告；但 **ART19 在中国大陆是干净版**。
- **ART19 的广告只投美国。** CN / AU / DE 均为 `115,557,687 B / 128.1 kbps`，美国为 `118,161,572 B / 131.0 kbps`。
- **Megaphone（Morning Wire / Start Here / Joe Rogan）** 所有可测地区都有 DAI，只是广告版本不同。
- **静态 feed 与地区无关**：Acast、Audioboom、RedCircle、Simplecast、Spreaker、Libsyn、Podbean、Captivate 在不同国家总长一致。
- **原音频**：Omny 公开 API 确认母带为 `39,285,812 B / 2452.257 s`，但公开 GET 一律被 `media-session` 广告拼接层接管；AU 干净版就是母带（多 138,748 B 的封装/ID3 开销）。目前拿不到可用的 AU 媒体出口，整集干净版尚未下载完成。

## 2. 方法

1. 通过代理出口访问同一集音频，用 `Range: 0-0` 读取 `Content-Range` 的总长度。
2. 用 Cloudflare `cdn-cgi/trace` 的 `loc` 确认代理实际出口国家。
3. 用 RSS 的 `<itunes:duration>` 折算有效码率：`kbps = 总字节 × 8 / 时长 / 1000`。
4. 判定标准：
   - Omny 母带约 **128.6 kbps**（≈39.4 MB）；国际广告版 **170.9 kbps**（52,382,960 B）；美国广告版约 **172.3 kbps**。
   - ART19 母带约 **128.1 kbps**（115,557,687 B）；美国广告版 **131.0 kbps**（118,161,572 B）。
   - 静态 feed：不同国家总长完全一致。
5. 中国大陆的 Omny 数据：`traffic.omny.fm` 被阻断，改为直连 Triton 源站（`iheartmedia.mc.tritondigital.com` / `*.mc.tritondigital.com`）测量；广告决策仍按 CN IP。

## 3. 15 个 DAI 厂商 / 节目对比

| # | DAI 厂商 · 节目 | 时长 | 中国大陆 CN | 澳大利亚 AU | US 参考 | 判定 |
|---|---|---|---|---|---|---|
| 1 | Megaphone · Morning Wire | 1150 s | 47,697,831 | 46,439,775 | 47,878,599 | 两国都有 DAI；AU 少 1.26 MB |
| 2 | Omny · Dan Patrick | 2452 s | 52,382,960 | **39,424,560** | 52,805,101 | **CN 有广告，AU 干净母带** |
| 3 | Omny · Breakfast Club | 2752 s | 50,679,386 | **44,248,252** | 51,957,923 | **CN 有广告，AU 干净母带** |
| 4 | Megaphone · Joe Rogan | 8671 s | 116,420,555 | 未采到 | 121,204,097 | CN 比 US 少 4.78 MB；AU 待补 |
| 5 | Acast · Casefile | 5811 s | 不可达 | 93,621,931 | 93,621,931 | AU 静态；CN 被阻断 |
| 6 | ART19 · Armchair Expert | 7214 s | **115,557,687** | **115,557,687** | 118,161,572 | **CN/AU 都干净**；US 有广告 |
| 7 | Audioboom · Tim Dillon | 4035 s | 不可达 | 64,692,183 | 64,692,183 | AU 静态；CN 被阻断 |
| 8 | RedCircle · Bourbon | 7755 s | 93,068,861 | 93,068,861 | 93,068,861 | 静态，无 DAI |
| 9 | Simplecast · The Daily | 1874 s | 29,984,547 | 29,984,547 | 29,984,547 | 静态，无 DAI |
| 10 | Spreaker · Sliced Golf | 3563 s | 不可达 | 56,999,645 | 56,999,645 | AU 静态；CN 被阻断 |
| 11 | Libsyn · The Feed | 3744 s | 30,278,335 | 未采到 | 30,278,335 | CN=US 静态；AU 按静态推断 |
| 12 | Podbean · Grumpy Economist | 2709 s | 不可达 | 33,597,059 | 33,597,059 | AU=US 静态；CN 被阻断 |
| 13 | Captivate · Insider | 3973 s | 63,575,456 | 63,575,456 | 63,575,456 | 静态，无 DAI |
| 14 | RSS.com · 示例 feed | 44 s | 不可达 | 403 | 738,199 | 样本太短；CN/AU 不可达 |
| 15 | SoundStack | — | 未采 | 未采 | — | 无公开 RSS |

> 表中数值单位为字节。CN 的“不可达”指大陆直连 RSS/音频被阻断；AU 的“未采到”指 AU 代理无法访问对应 CDN，不代表没有 DAI。

## 4. Omny Dan Patrick 地区判定

| 地区 | 总长 | 有效码率 | 判定 |
|---|---|---|---|
| **澳大利亚 AU** | **39,424,560** | **128.6 kbps** | **干净母带（无 DAI）** |
| 中国大陆 CN | 52,382,960 | 170.9 kbps | 国际广告版 |
| 德国 DE | 52,382,960 | 170.9 kbps | 国际广告版 |
| 法国 FR | 52,382,960 | 170.9 kbps | 国际广告版 |
| 日本 JP | 52,382,960 | 170.9 kbps | 国际广告版 |
| 韩国 KR | 52,382,960 | 170.9 kbps | 国际广告版 |
| 印度 IN | 52,382,960 | 170.9 kbps | 国际广告版 |
| 瑞典 SE | 52,382,960 | 170.9 kbps | 国际广告版 |
| 意大利 IT | 52,382,960 | 170.9 kbps | 国际广告版 |
| 新加坡 SG | 52,382,960 | 170.9 kbps | 国际广告版 |
| 香港 HK | 52,382,960 | 170.9 kbps | 国际广告版 |
| 俄罗斯 RU | 52,382,960 | 170.9 kbps | 国际广告版 |
| 美国 US | 52,805,101（另一时段 53,261,509） | 172.3 kbps | 美国广告版 |
| 新西兰 NZ | 未采到 | — | 未验证 |
| CA / GB / IE / NO / DK / FI / ZA / MX / BR / AR / CL / TW / MY / TH / AE / IL | 未采到 | — | 未验证（免费代理无法取媒体） |

> 较早一集的补充：NL、RO、VN、ID、IE、JP 也均为广告版（52.99–54.5 MB），没有干净版。

## 5. 访问层面：中国大陆 vs 澳大利亚

| 服务 / Feed | 中国大陆 | 澳大利亚 |
|---|---|---|
| Omny RSS | 可下载 | 可下载 |
| Omny 音频 | 直连卡在 `traffic.omny.fm`；Triton 源站可绕过（有广告） | 可下（干净版） |
| ART19 音频 | 可下（干净版） | 可下（干净版） |
| Megaphone 音频 | 可下（广告版） | 可下（广告版） |
| Acast / Audioboom / Spreaker | 不可达 | 可下（静态） |
| RedCircle / Simplecast | 可下（静态） | 可下（静态） |
| Libsyn / Captivate | 可下 | 可下 |
| Podbean | 不可达 | 可下（静态） |
| RSS.com / SoundStack | 不可达 / 无 feed | 403 / 无 feed |

## 6. 原音频获取

### 6.1 已确认的母带信息

Omny 公开 API：

```text
GET https://api.omny.fm/orgs/{org}/clips/{clip}
```

返回（Dan Patrick 当前集）：

| 字段 | 值 |
|---|---|
| `PublishedAudioSizeInBytes` | 39,285,812 B |
| `DurationSeconds` | 2452.257 s |
| `AudioUrl` / `MediaUrls.AudioUrl` | podtrac / traffic（DAI） |
| `AdMarkers` | PreRoll 0:00、MidRoll 17:07.28、MidRoll 28:24.25、PostRoll 40:52.26 |

对 Triton 源站发 `HEAD` 也返回 `200`、`39,285,812 B`、`accept-ranges: bytes`；但任何 `GET` 都会被 302 到 `media-session`，返回广告拼接版。

### 6.2 已排除的直取方式

- 去掉 `starship-episode-id`、去掉 query、`download=true`、`dl=1`；
- `noAds / ads=0 / dai=0 / raw=1 / type=original` 等 20 多个参数；
- `X-Forwarded-For` / `CF-Connecting-IP` 伪装澳洲 IP；
- Omny CDN 的 `original.mp3 / source.mp3 / master.mp3`、API 的 `original / media / download` 端点；
- 非零 Range、尾字节 Range。

全部仍返回拼接版或 404/400。

### 6.3 可行路径（按可靠性排序）

| 方法 | 结果 | 说明 |
|---|---|---|
| 1. 用 AU 出口下载 | **真正的母带（推荐）** | AU 返回 `39,424,560 B / 2452 s / 128.6 kbps`，比 API 母带仅多 138,748 B 的封装/ID3 开销 |
| 2. Omny 后台 / CMS | 直接下载原始文件 | 需要组织账号/API 权限 |
| 3. 从拼接版重建 | 内容完整、非字节级原文件 | 用 `AdMarkers` + 音频/ASR 对齐切掉 4 段广告（约 886 s） |
| 4. 静态 feed | RSS 直链就是原文件 | Acast/Audioboom/RedCircle/Simplecast/Spreaker 等无需处理 |
| 5. 官方/镜像站 | 不确定 | 可能仍是同一 Omny DAI 流 |

AU 下载与校验命令：

```bash
curl -L --proxy socks5h://<AU_PROXY> -o danpatrick_clean.mp3 "<RSS enclosure URL>"
ffprobe -v error -show_entries format=duration,bit_rate -of default=nw=1 danpatrick_clean.mp3
```

判定：约 39.4 MB、约 2452 s、128.6 kbps，开头直接是节目内容。

### 6.4 当前阻塞

免费 AU 出口（privasto / proxifly / GFP / spys.one 等）要么已死，要么是只放行 geo 查询的白名单代理；拼接版已完整下载并拿到 AdMarkers，但还没有可用的 AU 媒体出口完成整集干净版下载。

## 7. 未验证国家与限制

- **未验证**：新西兰、加拿大、英国、爱尔兰、挪威、丹麦、芬兰、南非、墨西哥、巴西、阿根廷、智利、台湾、马来西亚、泰国、阿联酋、以色列等。
- **原因**：免费代理只能过 Cloudflare trace，无法访问 Omny/ART19 媒体 CDN（302/403/超时）；GFP 的 50 万条 SOCKS5 抽样、proxifly、privasto、spys.one、proxylister 均未找到可用的目标国家媒体出口。
- “不可达 / 未采到”不等于“无 DAI”；只表示没有拿到可判定的音频长度。
- 广告模板会随时间/会话轮换，绝对字节数可能变化；本报告是 2026-11-09/10 的快照。
- 国家归因以 Cloudflare `loc` 为准；个别出口 IP 的地理库存在分歧（例如 `85.192.38.201`：Cloudflare/ipinfo 说 DE，ip-api/ipwho/db-ip 说 GB，ifconfig 说 RU），此类结果不应作为国家结论。
- 中国大陆的 Omny 数据是绕过 `traffic.omny.fm`、直连 Triton 源站测到的；广告决策仍按 CN IP。

## 8. 复现步骤

```bash
# 1) 确认代理出口国家
curl -s -x socks5h://<proxy> https://www.cloudflare.com/cdn-cgi/trace | grep -E '^(ip|loc|colo)='

# 2) 读取实际下发总长（不下载整集）
curl -sL -x socks5h://<proxy> -r 0-0 -D - -o /dev/null "<audio URL>" \
  | grep -i '^content-range'
```

- Omny 干净版：约 `39.4 MB` / `128.6 kbps`。
- Omny 国际广告版：`52,382,960 B` / `170.9 kbps`。
- Omny 美国广告版：约 `52.8 MB` / `172.3 kbps`。
- ART19 干净版：`115,557,687 B` / `128.1 kbps`；美国广告版：`118,161,572 B` / `131.0 kbps`。

## 9. 证据文件（本地，未提交）

- `geo_probe/episodes_all.json`：10 个 feed 的最新集与 enclosure。
- `geo_probe/episodes_extra.json`：JRE / Libsyn / Podbean / Captivate / RSS.com。
- `geo_probe/omny_clip.json`：Omny 公开 API 的母带信息与 AdMarkers。
- `geo_probe/omny_stitched.mp3`：已下载的拼接版（53,463,803 B / 3338.5 s）。
- `geo_probe/headers_chain.txt`、`geo_probe/variant_urls.txt`：Omny 跳转链与 URL 变体。
- `geo_probe/s2/`、`geo_probe/au_0.bin`、`geo_probe/au_5m.bin`、`geo_probe/de_0.bin`、`geo_probe/de_5m.bin`：首段/前 5MB 样本。
- `geo_probe/gfp_targets.json`、`geo_probe/spys_nz_decoded.txt`：免费代理的国家映射与 spys.one 端口解码结果。
