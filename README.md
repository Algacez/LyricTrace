# LyricTrace

逐字歌词同步系统：上传音频，自动产出每个字（或每个词）的演唱时间轴，可在网页编辑器里校对、导出。

- 自动识别歌名/歌手，从歌词源（网易云 → LRCLIB → LrcApi）获取原始歌词（也可手动粘贴）。
- 语音识别定位每一行，并给出逐字时间。
- 按读音把歌词行对齐到识别结果，LLM 校验并判断现场加唱。
- 可选「逐字对齐」（外部强制对齐器，作为识别词流的第二个意见做融合）进一步提升精度。
- LLM 按语意把字合并成词（可选）。
- 网页播放校对与编辑，导出 JSON / LRC / ASS。

**核心原则**：逐字时间只能来自真实的对齐结果——识别器直接听到的字、强制对齐器按行给出的时间，或歌词源自带、且通过 ASR 核对的逐字时间。两个后端都没给出的字按前后插值并标记 `interpolated`，整行没对上的标记 `unmatched`，绝不伪造逐字时间戳。

---

## 目录结构

```
.
├── backend/                    # FastAPI 应用 + pytest 测试
│   ├── app/
│   │   ├── main.py             # 应用、lifespan、CORS
│   │   ├── config.py           # 环境变量配置
│   │   ├── db.py               # SQLite 建表与访问
│   │   ├── schemas.py          # 全部 API 的 Pydantic 模型
│   │   ├── audio.py            # ffprobe 探测时长/标签
│   │   ├── tasks.py            # 单进程持久化后台任务队列
│   │   ├── routers/            # projects / debug / system 接口
│   │   └── services/           # 流水线、各 provider、对齐、导出
│   └── tests/
├── frontend/                   # React + TypeScript + Vite + wavesurfer.js
└── .env.example                # 全部配置项（含说明，默认值内置于代码）
```

---

## 环境要求

- Python 3.11+
- Node.js 18+
- `ffmpeg` / `ffprobe`（需在 PATH 中）
- 三个可选项 HTTP 服务（内网即可）：
  - Qwen3-ASR（带时间的词流：定位每一行并给出逐字时间）
  - 强制对齐器（可选；项目打开「逐字对齐」时参与逐字时间）
  - OpenAI 兼容 LLM（行校验、加唱判定、语意合并）
- BSRoformer.cpp 人声分离（默认开启，面板上可取消勾选）

逐字时间默认来自**识别器自带的逐字 token**（不请求强制对齐器）；项目开关 `hfa_chars` 打开后两个后端都跑再融合（`LYRICTRACE_CHAR_HFA` 改全局默认，`LYRICTRACE_CHAR_FUSE` 控制融合本身，默认开）。配置 `LYRICTRACE_CHAR_ALIGNER_URL` 才有强制对齐器可用。

---

## 快速开始

### 后端

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # 可选
pip install -r requirements.txt

# 可选：加载配置（后端不会自动读取 .env，请用 source 或进程管理器注入）
# set -a; source ../.env; set +a

uvicorn app.main:app --reload --port 8000
```

- OpenAPI 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/api/health>
- 数据默认写到仓库根目录的 `data/`（SQLite + 上传的音频）。

### 前端

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite 会把 `/api` 代理到 `http://127.0.0.1:8000`。后端不在默认端口时：

```bash
LYRICTRACE_API_URL=http://127.0.0.1:8765 npm run dev
```

生产构建：`npm run build`（产物在 `frontend/dist/`，需自行用静态服务器托管，并把 `/api` 反代到后端）。不同源部署时：`VITE_API_BASE=https://api.example.com npm run build`。

### 测试与检查

```bash
cd backend && python3 -m pytest -q
cd frontend && npm run typecheck && npm run build
```

测试使用临时数据目录，不访问任何模型服务。

---

## 使用流程

1. 在首页拖入或选择音频文件（mp3 / flac / wav / m4a / aac / ogg / opus / wma / aiff / webm）。
2. 后端创建项目并异步探测音频、提取歌曲信息（内嵌标签优先，其次 LLM 解析文件名，最后退回规则解析），再用识别出的歌名/歌手去歌词源取原始歌词。
3. 歌词**默认来自歌词源**：匹配度够高就自动填入「原始歌词」；没查到或想用别的版本，可搜候选替换或直接粘贴——手填/手改的歌词优先，永不被 API 覆盖（带 `[mm:ss.xx]` 也行，时间戳会被忽略，只取文本）。然后点「开始逐字打轴」，后端依次：
   - **人声分离（默认开启）**：先取出人声更稳；干净人声可取消勾选直接识别原始混音。
   - **分段识别（只负责定位）**：按 20 秒一段送识别，拿回带时间戳的字/词。ASR 自带的强制对齐默认关闭（唱歌上更差），关掉后词流由识别服务的 `/transcribe` 给出。整首一次识别会漂移，分段后行级准确率更高。识别语言由 LLM 判断（可手动指定）。
   - **逐字歌词核对**：歌词源若自带逐字时间（LRCLIB `lyricsfile` / 网易云 `yrc`），先和 ASR 核对，对得上就直接用并平移，否则走本地流程。
   - **歌词对齐**：歌词与识别结果转成假名读音（pykakasi），做全局字符 DP，得到每行对应的识别片段。
   - **参考定位**：歌词源条目若带 LRC 时间轴，用它找回 DP 没对上的行。
   - **LLM 校验**：只标记不改时间，可疑行打 `可能错行` / `部分吻合` 附原因。
   - **识别现场加唱**（项目开关 `detect_extras`，默认关）：默认只输出原始歌词行；打开后额外内容交 LLM 判断是否真实人声并保留为 `asr-extra` 行。
   - **漏识别片段处理**：一行只唱对一部分时，紧贴某行的短识别片段并回该行；够长/够远的片段保留为完整新句。
   - **识别间隔检查**：句中空档过大说明漏字，整句（含前后 0.4s）重新识别，填不满则保留原时间，绝不编造。
   - **逐字时间精修**：默认直接取识别器逐字 token；打开「逐字对齐」（`hfa_chars`）后两个后端融合（相差 ≤ `FUSE_TOLERANCE` 取识别词流，否则取对齐器）。
   - **LLM 按语意合并成词**（`llm_merge`，默认关）：只问在哪里断词，时间取合并单元首字起点到末字终点。
4. 结果回到时间轴编辑器：波形播放/暂停、点击跳到行、卡拉 OK 高亮（按「当前唱到哪个字」，字与字之间不熄灭）；跟随播放、行标签（`部分吻合`/`可能错行`/`参考定位`/`未识别到`/`现场加唱`）、插入/编辑歌词行、整行与逐字平移、逐字滑块、多选字均分、逐字打轴、参考时间轴跳转、自动保存与撤销/重做。
5. 右上角导出 JSON / LRC / ASS。

歌词来源记在项目的 `lyrics_source`（`lrclib`/`lrcapi`/`netease`/`manual`）与 `lyrics_reference` 上；只有被 ASR 核对的歌词源自带逐字时间会被采用，其余逐字时间来自识别器逐字 token（或融合的强制对齐结果）。

**翻唱匹配**：歌手对不上时不直接丢弃，只要歌名足够接近且时长相差 ≤5 秒就当作同一首歌的另一演唱版本。

时间轴状态：`none` / `untimed` / `line_only` / `partial` / `complete`（所有行、所有字都有 start/end）。

---

## 数据模型（节选）

- `words[].source`：`asr`（识别词流）、`aligner`（强制对齐结果）、`merged`（语意合并单元）、`interpolated`（插值）、`word-sync`（歌词源逐字）、`manual`（编辑器人工改过）。
- `lines[].source`：`asr` / `asr-partial` / `asr-bad` / `asr-extra` / `unmatched` / `word-sync(-shifted|-missing)`。
- 中间结果存 `debug_steps` 表，每个项目保留最近 10 次运行（见调试页 `/projects/<id>/debug`）。
- 规范单位为**秒（float）**；LRC/ASS 导出精度为厘秒（10 ms）。

---

## API 示例

完整交互式文档在 `/docs`。后端默认 `http://127.0.0.1:8000`。

```bash
# 健康检查
curl http://127.0.0.1:8000/api/health

# 上传音频，创建项目（异步任务）
curl -X POST http://127.0.0.1:8000/api/projects -F "file=@song.mp3"

# 写入原始歌词并开始逐字打轴
curl -X PUT  http://127.0.0.1:8000/api/projects/prj_xxx/lyrics \
  -H 'Content-Type: application/json' \
  -d '{"original_lyrics":"第一行\n第二行","language":"Japanese"}'
curl -X POST http://127.0.0.1:8000/api/projects/prj_xxx/align \
  -H 'Content-Type: application/json' -d '{"reset_cache":false}'

# 一次性对齐 API（不建项目、不开网页，音频 → 逐字时间戳）
curl -X POST http://127.0.0.1:8000/api/align -F "file=@song.mp3" -F "language=Japanese"
curl        http://127.0.0.1:8000/api/align/job_xxx

# 导出
curl -o lyrics.json "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=json"
curl -o lyrics.lrc  "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=lrc"
curl -o lyrics.ass  "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=ass&karaoke=true"
```

`GET /api/providers`、`/api/formats`、`/api/config`、`/api/storage`、`POST /api/storage/cleanup` 提供运行时信息与缓存管理。

---

## Provider 接口

`backend/app/services/providers/base.py` 定义全部可替换组件，注册表按环境变量选择实现：

| 类别 | 接口 | 当前状态 | 环境变量 |
| --- | --- | --- | --- |
| 元数据提取 | `MetadataExtractor.extract` | `llm` / `filename` | `LYRICTRACE_METADATA_EXTRACTOR` |
| 语音识别 | `ASRProvider.transcribe` | `qwen-asr-http`（Qwen3-ASR，自带对齐默认关） | `LYRICTRACE_ASR_PROVIDER` / `LYRICTRACE_ASR_ALIGN` |
| 歌词对齐 | `AlignerProvider.align` | `llm-aligner`（读音 DP + LLM） | `LYRICTRACE_ALIGNER_PROVIDER` |
| 强制对齐器 | 逐字时间第二意见 | 可选 | `LYRICTRACE_CHAR_ALIGNER_URL` / `LYRICTRACE_CHAR_HFA` |
| 人声分离 | `SeparatorProvider.separate` | `bs-roformer` | `LYRICTRACE_SEPARATOR_PROVIDER` |
| 歌词纠错 / 音频导入 | 未实现 | `unavailable` | `LYRICTRACE_CORRECTOR_PROVIDER` / `LYRICTRACE_IMPORTER_PROVIDER` |

接入新模型：实现对应抽象类并在注册表登记即可；未实现/未配置的实现抛 `ProviderUnavailable`，由路由转成 503，其他功能不受影响。

---

## License

[MIT](LICENSE)
