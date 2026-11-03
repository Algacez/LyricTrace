# LyricTrace

<p align="center">
  <a href="README_CN.md">简体中文</a>
</p>

A word-level lyric synchronization system. Upload audio to automatically generate a timestamp for every character or word in the performance. Review and edit the results in a web editor, then export them.

- Automatically identifies song titles and artists, then retrieves original lyrics from lyric providers (NetEase Cloud Music → LRCLIB → LrcApi). Lyrics can also be pasted manually.
- Uses speech recognition to locate each lyric line and provide word-level timestamps.
- Aligns lyric lines with recognition results based on pronunciation. An LLM validates the alignment and detects live performance additions.
- Optional **character-level alignment** uses an external forced aligner as a second opinion to improve timestamp accuracy through fusion.
- Optionally merges characters into words based on semantic meaning using an LLM.
- Provides web-based playback, review, and editing, with JSON, LRC, and ASS export.

**Core principle:** Character-level timestamps must come from genuine alignment results: characters directly recognized by the ASR system, timestamps assigned line-by-line by a forced aligner, or character-level timestamps provided by a lyric source and verified against ASR. Characters without timestamps from either backend are interpolated from neighboring timestamps and marked `interpolated`. Entire unmatched lines are marked `unmatched`. Timestamps are never fabricated.

---

## Directory Structure

```
.
├── backend/                    # FastAPI application + pytest tests
│   ├── app/
│   │   ├── main.py             # Application, lifespan, CORS
│   │   ├── config.py           # Environment variable configuration
│   │   ├── db.py               # SQLite schema and access
│   │   ├── schemas.py           # Pydantic models for all APIs
│   │   ├── audio.py             # ffprobe duration and metadata extraction
│   │   ├── tasks.py              # Single-process persistent background task queue
│   │   ├── routers/              # Projects / debug / system endpoints
│   │   └── services/             # Pipeline, providers, alignment, export
│   └── tests/
├── frontend/                    # React + TypeScript + Vite + wavesurfer.js
└── .env.example                 # All configuration options and descriptions
```

---

## Requirements

- Python 3.11+
- Node.js 18+
- `ffmpeg` / `ffprobe` (must be available in `PATH`)
- Three optional HTTP services (can run on an internal network):
  - Qwen3-ASR (timestamped token stream for locating lines and providing character-level timestamps)
  - Forced aligner (optional; used for character timestamps when character-level alignment is enabled)
  - OpenAI-compatible LLM (line validation, extra-lyrics detection, semantic merging)
- BSRoformer.cpp vocal separation (enabled by default; can be disabled in the UI)

Character timestamps come from **the ASR system's own character-level tokens by default** (the forced aligner is not called). When the `hfa_chars` project option is enabled, both backends run and their results are fused. `LYRICTRACE_CHAR_HFA` controls the global default, while `LYRICTRACE_CHAR_FUSE` controls fusion itself (enabled by default). The forced aligner is available only when `LYRICTRACE_CHAR_ALIGNER_URL` is configured.

---

## Quick Start

### Backend

```
cd backend
python3 -m venv .venv && source .venv/bin/activate  # Optional
pip install -r requirements.txt

# Optional: load configuration (the backend does not read .env automatically)
# Use source or inject variables through your process manager:
# set -a; source ../.env; set +a

uvicorn app.main:app --reload --port 8000
```

- OpenAPI documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- Data is stored in `data/` at the repository root by default (SQLite database and uploaded audio).

### Frontend

```
cd frontend
npm install
npm run dev  # http://localhost:5173
```

Vite proxies `/api` to `http://127.0.0.1:8000`. If the backend uses a different port:

```
LYRICTRACE_API_URL=http://127.0.0.1:8765 npm run dev
```

Production build:

```
npm run build
```

The output is in `frontend/dist/`. Serve it with a static web server and reverse-proxy `/api` to the backend. For cross-origin deployments:

```
VITE_API_BASE=https://api.example.com npm run build
```

### Tests and Checks

```
cd backend && python3 -m pytest -q
cd frontend && npm run typecheck && npm run build
```

Tests use temporary data directories and do not access model services.

---

## Workflow

1. **Upload audio.** Drag and drop or select an audio file on the home page (mp3 / flac / wav / m4a / aac / ogg / opus / wma / aiff / webm).
2. **Create a project.** The backend creates a project, asynchronously probes the audio, and extracts song metadata. Embedded tags take priority, followed by LLM-based filename parsing, then rule-based parsing. It then retrieves original lyrics using the detected title and artist.
3. **Review the lyrics.** Lyrics are **retrieved from lyric providers by default**. If a sufficiently good match is found, the lyrics are populated automatically. If no match is found or you want another version, search alternatives or paste lyrics manually. Manually entered or edited lyrics always take priority and are never overwritten by API results. Timestamps such as `[mm:ss.xx]` are accepted but ignored; only the text is used.

   Click **Start Character-Level Sync**. The backend runs the following pipeline:
   - **Vocal separation (enabled by default):** Extracts vocals for more reliable recognition. Disable it to recognize the original mix directly when the vocals are already clean.
   - **Segmented recognition (localization only):** Audio is sent to the recognizer in 20-second segments, returning timestamped characters or words. ASR's built-in forced alignment is disabled by default because it performs worse on singing. With it disabled, the token stream comes from the recognition service's `/transcribe` endpoint. Recognizing the entire song at once can cause drift; segmentation improves line-level accuracy. The LLM detects the language, which can also be specified manually.
   - **Character-level lyric verification:** If the lyric source provides character-level timestamps (LRCLIB `lyricsfile` or NetEase Cloud Music `yrc`), they are checked against ASR. Matching timestamps are used directly, with an offset applied when necessary; otherwise, the local pipeline is used.
   - **Lyric alignment:** Lyrics and recognition results are converted to phonetic readings (using pykakasi), then aligned with global character-level dynamic programming to identify the corresponding recognition segment for each line.
   - **Reference localization:** If a lyric source provides an LRC timeline, it is used to recover lines that the dynamic programming alignment could not match.
   - **LLM validation:** The LLM flags suspicious lines without changing timestamps, labeling them `Possible Misalignment` or `Partial Match` with an explanation.
   - **Live extra-lyrics detection:** Controlled by the `detect_extras` project option (disabled by default). Only original lyric lines are output by default. When enabled, the LLM determines whether additional recognized content is actual singing and retains it as `asr-extra` lines.
   - **Missing-fragment handling:** Short recognized fragments close to a lyric line are merged back into that line when only part of it was recognized. Longer or more distant fragments are retained as complete new lines.
   - **Recognition-gap checks:** Large gaps within a line may indicate missed characters. The entire line is re-recognized, including 0.4 seconds before and after it. If the gap cannot be filled, the original timestamps are retained; timestamps are never invented.
   - **Character timestamp refinement:** By default, timestamps come directly from ASR character-level tokens. When **Character-Level Alignment** (`hfa_chars`) is enabled, both backends are fused. If their timestamps differ by no more than `FUSE_TOLERANCE`, the ASR token stream is used; otherwise, the aligner's result is used.
   - **LLM semantic word merging** (`llm_merge`, disabled by default): The LLM identifies word boundaries only. Each merged unit's timestamp spans from the first character's start to the last character's end.
4. **Review and edit the results.** The timeline editor supports waveform playback and pause, clicking to jump to a line, karaoke highlighting (the current character remains highlighted between characters), playback following, line labels (`Partial Match`, `Possible Misalignment`, `Reference Localized`, `Unrecognized`, `Live Extra`), inserting and editing lyric lines, shifting entire lines or individual characters, per-character sliders, distributing timestamps evenly across selected characters, character-level timestamping, jumping to the reference timeline, autosave, and undo/redo.
5. **Export.** Use the top-right menu to export JSON, LRC, or ASS.

The project's `lyrics_source` (`lrclib` / `lrcapi` / `netease` / `manual`) and `lyrics_reference` record the lyric source. Only source-provided character timestamps verified against ASR are used; all other character timestamps come from ASR tokens or fused forced-alignment results.

**Cover matching:** If the artist does not match, the result is not immediately discarded. If the title is sufficiently similar and the duration differs by no more than 5 seconds, it is treated as another version of the same song.

Timeline states: `none` / `untimed` / `line_only` / `partial` / `complete` (every line and every character has a start and end timestamp).

---

## Data Model (Excerpt)

- `words[].source`: `asr` (recognized token stream), `aligner` (forced alignment result), `merged` (semantic merge unit), `interpolated` (interpolation), `word-sync` (source-provided character timestamps), `manual` (edited in the editor).
- `lines[].source`: `asr` / `asr-partial` / `asr-bad` / `asr-extra` / `unmatched` / `word-sync(-shifted|-missing)`.
- Intermediate results are stored in the `debug_steps` table. Each project retains its 10 most recent runs (see the debug page at `/projects/<id>/debug`).
- The canonical unit is **seconds (float)**. LRC/ASS exports use centisecond precision (10 ms).

---

## API Examples

Full interactive documentation is available at `/docs`. The backend defaults to `http://127.0.0.1:8000`.

```
# Health check
curl http://127.0.0.1:8000/api/health

# Upload audio and create a project (asynchronous task)
curl -X POST http://127.0.0.1:8000/api/projects -F "file=@song.mp3"

# Set original lyrics and start character-level synchronization
curl -X PUT http://127.0.0.1:8000/api/projects/prj_xxx/lyrics \
  -H 'Content-Type: application/json' \
  -d '{"original_lyrics":"First line\nSecond line","language":"Japanese"}'

curl -X POST http://127.0.0.1:8000/api/projects/prj_xxx/align \
  -H 'Content-Type: application/json' -d '{"reset_cache":false}'

# One-shot alignment API (no project or web UI; audio -> character timestamps)
curl -X POST http://127.0.0.1:8000/api/align \
  -F "file=@song.mp3" -F "language=Japanese"

curl http://127.0.0.1:8000/api/align/job_xxx

# Export
curl -o lyrics.json "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=json"
curl -o lyrics.lrc "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=lrc"
curl -o lyrics.ass "http://127.0.0.1:8000/api/projects/prj_xxx/export?format=ass&karaoke=true"
```

`GET /api/providers`, `/api/formats`, `/api/config`, `/api/storage`, and `POST /api/storage/cleanup` provide runtime information and cache management.

---

## Provider Interfaces

`backend/app/services/providers/base.py` defines the replaceable components. The registry selects implementations through environment variables:

| Category                        | Interface                               | Current Implementation                                              | Environment Variable                                             |
| ------------------------------- | --------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Metadata extraction             | `MetadataExtractor.extract`             | `llm` / `filename`                                                  | `LYRICTRACE_METADATA_EXTRACTOR`                                  |
| Speech recognition              | `ASRProvider.transcribe`                | `qwen-asr-http` (Qwen3-ASR; built-in alignment disabled by default) | `LYRICTRACE_ASR_PROVIDER` / `LYRICTRACE_ASR_ALIGN`               |
| Lyric alignment                 | `AlignerProvider.align`                 | `llm-aligner` (phonetic DP + LLM)                                   | `LYRICTRACE_ALIGNER_PROVIDER`                                    |
| Forced aligner                  | Second opinion for character timestamps | Optional                                                            | `LYRICTRACE_CHAR_ALIGNER_URL` / `LYRICTRACE_CHAR_HFA`            |
| Vocal separation                | `SeparatorProvider.separate`            | `bs-roformer`                                                       | `LYRICTRACE_SEPARATOR_PROVIDER`                                  |
| Lyric correction / audio import | Not implemented                         | `unavailable`                                                       | `LYRICTRACE_CORRECTOR_PROVIDER` / `LYRICTRACE_IMPORTER_PROVIDER` |

To integrate a new model, implement the corresponding abstract class and register it. Unimplemented or unconfigured providers raise `ProviderUnavailable`, which the router converts to HTTP 503 without affecting other features.

---

## License

MIT