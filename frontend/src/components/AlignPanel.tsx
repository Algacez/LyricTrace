import { useEffect, useMemo, useState } from "react";
import { api, errorMessage } from "../api";
import { useAutosave } from "../hooks/useAutosave";
import type { LyricsCandidate, Project, Task } from "../types";
import { formatTime } from "../utils";

interface Props {
  project: Project;
  task: Task | null;
  busy: boolean;
  onProjectChange: (project: Project) => void;
  onRun: (resetCache: boolean) => void;
}

const STAGE_LABEL: Record<string, string> = {
  separate: "分离人声",
  asr: "语音识别",
  language: "判断语言",
  align: "对齐",
  merge: "按语意合并",
  done: "完成",
};

const SOURCE_NAME: Record<string, string> = {
  lrclib: "LRCLIB",
  lrcapi: "LrcApi",
  netease: "网易云",
};

/**
 * The one input that matters before alignment: the original lyrics.
 * Everything else (metadata, LRCLIB) was deliberately dropped.
 */
export function AlignPanel({ project, task, busy, onProjectChange, onRun }: Props) {
  const [lyrics, setLyrics] = useState(project.original_lyrics);
  const [language, setLanguage] = useState(project.lyrics_language ?? "auto");
  const [separate, setSeparate] = useState(project.separate_vocals ?? true);
  const [hfaChars, setHfaChars] = useState(project.hfa_chars ?? false);
  const [wordSyncSearch, setWordSyncSearch] = useState(project.word_sync_search ?? true);
  const [detectExtras, setDetectExtras] = useState(project.detect_extras ?? false);
  const [llmMerge, setLlmMerge] = useState(project.llm_merge ?? false);

  useEffect(() => {
    setLyrics(project.original_lyrics);
    setLanguage(project.lyrics_language ?? "auto");
    setSeparate(project.separate_vocals ?? true);
    setHfaChars(project.hfa_chars ?? false);
    setWordSyncSearch(project.word_sync_search ?? true);
    setDetectExtras(project.detect_extras ?? false);
    setLlmMerge(project.llm_merge ?? false);
    setCandidates(null);
    setSearchError(null);
    setQuery("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id]);

  const save = useMemo(
    () => async (value: string) => {
      // Only send the language when it actually differs from what the server
      // has. The initial value is the LLM's own guess; echoing it back would
      // record it as a *manual* choice and freeze a wrong guess forever.
      const stored = project.lyrics_language ?? "auto";
      const updated = await api.updateLyrics(project.id, {
        original_lyrics: value,
        ...(language !== stored ? { language } : {}),
      });
      onProjectChange(updated);
    },
    [project.id, project.lyrics_language, language, onProjectChange],
  );
  const autosave = useAutosave(lyrics, save, { delay: 1000, signature: lyrics });

  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<LyricsCandidate[] | null>(null);
  const [query, setQuery] = useState("");

  const search = async (text?: string) => {
    setSearching(true);
    setSearchError(null);
    try {
      const result = await api.searchLyrics(project.id, text || undefined);
      setCandidates(result.candidates);
    } catch (err) {
      setSearchError(errorMessage(err));
      setCandidates([]);
    } finally {
      setSearching(false);
    }
  };

  const pick = async (candidate: LyricsCandidate) => {
    const externalId = candidate.external_id ?? (candidate.lrclib_id !== null ? String(candidate.lrclib_id) : null);
    if (externalId === null) return;
    setSearching(true);
    setSearchError(null);
    try {
      const updated = await api.selectLyrics(project.id, {
        source: candidate.source,
        external_id: externalId,
        ...(candidate.lrclib_id !== null ? { lrclib_id: candidate.lrclib_id } : {}),
      });
      setLyrics(updated.original_lyrics);
      onProjectChange(updated);
      setCandidates(null);
    } catch (err) {
      setSearchError(errorMessage(err));
    } finally {
      setSearching(false);
    }
  };

  const lineCount = lyrics.split("\n").filter((line) => line.trim()).length;
  const hasTimeline = (project.timeline?.lines.length ?? 0) > 0;
  const sourceLabel =
    project.lyrics_source === "manual"
      ? "手动填写"
      : project.lyrics_source
        ? `来自 ${SOURCE_NAME[project.lyrics_source] ?? project.lyrics_source}${
            project.lyrics_lrclib_id ? ` #${project.lyrics_lrclib_id}` : ""
          }`
        : null;
  const referenceLabel = project.lyrics_reference ? "歌词源时间轴仅作参考" : null;
  const languageLabel =
    project.lyrics_language && project.language_origin === "llm"
      ? `识别语言：${project.lyrics_language}（LLM 判断）`
      : project.lyrics_language && project.language_origin === "manual"
        ? `识别语言：${project.lyrics_language}（手动指定）`
        : project.language_origin === "auto"
          ? "语言：自动识别"
          : null;

  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
        <h2 style={{ margin: 0 }}>
          原始歌词
          {sourceLabel && <span className="badge info">{sourceLabel}</span>}
          {referenceLabel && (
            <span className="badge" title="歌词源自带的时间轴只用来估计每行大致位置；最终的行位置来自识别词流，逐字时间来自识别器自带的逐字 token（打开「强制对齐逐字」时再与对齐器融合）">
              {referenceLabel}
            </span>
          )}
          {project.lyrics_word_sync_edition && (
            <span
              className="badge info"
              title="本项目用的歌词条目没有逐字歌词，打轴时采用了同一首歌另一个版本的逐字歌词（仍会先和这段音频核对是否吻合）"
            >
              逐字歌词来自另一个版本：{project.lyrics_word_sync_edition}
            </span>
          )}
          {languageLabel && (
            <span
              className="badge"
              title="判断不了语言时会交给 ASR 自动识别；这里可以手动改"
            >
              {languageLabel}
            </span>
          )}
        </h2>
        <div className="row small">
          <a
            href={api.lyricsExportUrl(project.id, project.lyrics_reference ? "lrc" : "txt")}
            download
            title={
              project.lyrics_reference
                ? "下载原始歌词，带歌词源自带的时间轴（LRC）"
                : "歌词源没有提供时间轴，下载的是纯文本歌词"
            }
          >
            <button disabled={!project.original_lyrics.trim()}>
              下载原始歌词{project.lyrics_reference ? "（含时间轴）" : "（纯文本）"}
            </button>
          </a>
          <span className={autosave.status === "error" ? "error-text small" : "muted small"}>
            {autosave.status === "saving" && "保存中…"}
            {autosave.status === "saved" && "已保存"}
            {autosave.status === "error" && `保存失败：${autosave.error}`}
            {(autosave.status === "idle" || lyrics === project.original_lyrics) &&
              !autosave.error &&
              "编辑后自动保存"}
          </span>
        </div>
      </div>

      <div className="row" style={{ marginBottom: 8, gap: 6 }}>
        <input
          type="search"
          value={query}
          placeholder="搜索歌词（留空则用歌名 / 歌手）"
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void search(query);
          }}
          style={{ flex: 1, minWidth: 180 }}
        />
        <button disabled={searching} onClick={() => void search(query)}>
          {searching ? "查询中…" : "查找歌词"}
        </button>
      </div>
      <p className="muted small" style={{ margin: "0 0 8px" }}>
        依次查找 网易云 → LRCLIB → 内网 LrcApi，取第一个带逐字歌词的可信结果（没有逐字歌词时退回文本）。
      </p>

      {searchError && <p className="error-text small">歌词查询失败：{searchError}</p>}

      {candidates && candidates.length === 0 && !searching && (
        <p className="muted small">网易云 / LRCLIB / LrcApi 都没有找到歌词，可以手动粘贴。</p>
      )}

      {candidates && candidates.length > 0 && (
        <ul className="lyrics-candidates">
          {candidates.map((candidate) => (
            <li key={`${candidate.source}:${candidate.external_id ?? candidate.lrclib_id}`}>
              <button className="lyrics-candidate" onClick={() => void pick(candidate)}>
                <span className="title">
                  {candidate.track_name || "（无标题）"}
                  <span className="muted"> — {candidate.artist_name || "未知歌手"}</span>
                  <span className="badge info">{SOURCE_NAME[candidate.source] ?? candidate.source}</span>
                </span>
                <span className="meta small muted">
                  {candidate.line_count} 行 · 匹配度 {Math.round(candidate.score * 100)}%
                  {candidate.duration ? ` · ${formatTime(candidate.duration)}` : ""}
                  {candidate.word_sync
                    ? " · 带逐字歌词（与音频吻合时采用）"
                    : candidate.synced
                      ? " · 带时间轴（作定位参考）"
                      : ""}
                </span>
                <span className="preview small">{candidate.preview}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <textarea
        rows={10}
        value={lyrics}
        onChange={(event) => setLyrics(event.target.value)}
        placeholder={"每行一句歌词。上传后会自动查找；也可以直接粘贴，带 [mm:ss.xx] 时间戳的 LRC 也行（时间戳只用来估计每行位置，不会被写进结果）。"}
      />

      <div className="row" style={{ marginTop: 10, gap: 8 }}>
        <label className="muted small">语言</label>
        <select
          value={language}
          onChange={(event) => {
            const next = event.target.value;
            setLanguage(next);
            // Save immediately: the language is read when the run starts.
            void api
              .updateLyrics(project.id, { language: next })
              .then(onProjectChange)
              .catch((err: unknown) => setSearchError(errorMessage(err)));
          }}
          title="唱的是什么语言。默认自动：由 LLM 根据歌名/歌手/歌词判断，判断不了就交给识别服务自动识别。这个值同时决定识别语言和逐字对齐用的语言，选错会让整句对不上。"
        >
          {["auto", "Japanese", "Chinese", "English", "Korean", "Cantonese"].map((item) => (
            <option key={item} value={item}>
              {item === "auto" ? "自动（LLM 判断）" : item}
            </option>
          ))}
        </select>
        <span className="muted small">{lineCount} 行</span>
      </div>

      <div className="align-options">
        <label className="align-option">
          <input
            type="checkbox"
            checked={separate}
            disabled={busy}
            onChange={(event) => {
              setSeparate(event.target.checked);
              // Saved at once: the align run reads it when it starts, and
              // switching invalidates the cached ASR result server-side.
              void api
                .updateProject(project.id, { separate_vocals: event.target.checked })
                .then(onProjectChange)
                .catch((err: unknown) => setSearchError(errorMessage(err)));
            }}
          />
          <span>
            <b>分离人声</b>
            <span className="muted small">
              先用 BSRoformer 把伴奏去掉再识别。默认开：伴奏很响时不容易被误听成歌词，代价是多花 1–2 分钟。
              关掉就直接识别原始混音，更快，但伴奏可能被听成词。切换后需要重新识别。
            </span>
          </span>
        </label>

        <label className="align-option">
          <input
            type="checkbox"
            checked={hfaChars}
            disabled={busy}
            onChange={(event) => {
              setHfaChars(event.target.checked);
              void api
                .updateProject(project.id, { hfa_chars: event.target.checked })
                .then(onProjectChange)
                .catch((err: unknown) => setSearchError(errorMessage(err)));
            }}
          />
          <span>
            <b>强制对齐逐字</b>
            <span className="muted small">
              默认关：每个字的时间只来自识别词流，一次对齐跑得最快。打开后每句再让强制对齐器
              对齐一遍，两个后端一致时取识别器给的边界、分歧时用对齐器——对照官方逐字歌词，
              字误差从 0.14 秒降到 0.085 秒，代价是每句多一次请求（四分钟的歌约 30–60 秒）。
            </span>
          </span>
        </label>

        <label className="align-option">
          <input
            type="checkbox"
            checked={wordSyncSearch}
            disabled={busy}
            onChange={(event) => {
              setWordSyncSearch(event.target.checked);
              void api
                .updateProject(project.id, { word_sync_search: event.target.checked })
                .then(onProjectChange)
                .catch((err: unknown) => setSearchError(errorMessage(err)));
            }}
          />
          <span>
            <b>查找其他版本的逐字歌词</b>
            <span className="muted small">
              本项目这条歌词没有逐字歌词时，去同一首歌的其他版本（例如现场版）里找。找到的版本要先和这段音频核对
              吻合才采用，采用后它的逐字时间直接作为结果。默认开；关掉则只用本项目歌词条目自带的数据。
            </span>
          </span>
        </label>

        <label className="align-option">
          <input
            type="checkbox"
            checked={detectExtras}
            disabled={busy}
            onChange={(event) => {
              setDetectExtras(event.target.checked);
              void api
                .updateProject(project.id, { detect_extras: event.target.checked })
                .then(onProjectChange)
                .catch((err: unknown) => setSearchError(errorMessage(err)));
            }}
          />
          <span>
            <b>识别现场加唱</b>
            <span className="muted small">
              默认关：结果只有原始歌词的行，识别到但歌词里没有的内容（开场白、MC、即兴加唱、观众合唱）一律不输出。
              关着时如果某一段其实是识别器把某句歌词听错了，LLM 会把它对应到那句歌词并重新识别那段音频，
              让这一行拿到时间——识别错误照样修，只是不会变成新行。打开后这些内容会作为「现场加唱」行插入时间轴。
            </span>
          </span>
        </label>

        <label className="align-option">
          <input
            type="checkbox"
            checked={llmMerge}
            disabled={busy}
            onChange={(event) => {
              setLlmMerge(event.target.checked);
              void api
                .updateProject(project.id, { llm_merge: event.target.checked })
                .then(onProjectChange)
                .catch((err: unknown) => setSearchError(errorMessage(err)));
            }}
          />
          <span>
            <b>按语意合并成词</b>
            <span className="muted small">
              默认关：每个字有自己独立的时间段，高亮一次亮一个字，这就是测量到的粒度。打开后整份结果再交给
              LLM 一次，由它判断每个词在哪里断开（君の / 心が / 見えない），每个词的时间取它覆盖的第一个字的起点
              到最后一个字的终点——LLM 只断词、不产生任何时间；返回的单元拼不回原行就整行保留逐字结果。
              代价是多一次 LLM 请求，且高亮一次亮一个词而不是一个字。
            </span>
          </span>
        </label>
      </div>

      <div className="row" style={{ marginTop: 12, gap: 8 }}>
        <button
          className="primary"
          disabled={busy}
          onClick={() => onRun(false)}
          title={
            lineCount === 0
              ? "还没有歌词：将直接用 ASR 识别结果按停顿断句（文本是识别出来的，不是歌词）"
              : "识别定位每一行 → 逐字时间（打开「按语意合并成词」时再由 LLM 断词）"
          }
        >
          {busy ? "处理中…" : hasTimeline ? "重新逐字打轴" : "开始逐字打轴"}
        </button>
        {hasTimeline && (
          <button
            disabled={busy}
            onClick={() => {
              if (window.confirm("重新做一次语音识别（忽略缓存的 ASR 结果）？")) onRun(true);
            }}
            title="忽略缓存的识别结果，重新识别一遍（歌词、设置都不变；只在怀疑识别本身出错时才需要）"
          >
            重新识别
          </button>
        )}
        <div className="spacer" />
      </div>

      {busy && task && (
        <div style={{ marginTop: 12 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="small">
              {STAGE_LABEL[task.stage] || task.stage || "准备中"}
              {task.message ? ` · ${task.message}` : ""}
            </span>
            <span className="mono small">{Math.round(task.progress * 100)}%</span>
          </div>
          <div className="progress" style={{ marginTop: 6 }}>
            <div style={{ width: `${Math.round(task.progress * 100)}%` }} />
          </div>
        </div>
      )}

      {!busy && task?.status === "failed" && (
        <p className="error-text small" style={{ marginTop: 10 }}>
          对齐失败：{task.error || project.error}
        </p>
      )}

      {!busy && task?.status === "succeeded" && task.message && (
        <p className="ok-text small" style={{ marginTop: 10 }}>
          {task.message}
        </p>
      )}

      <p className="muted small" style={{ marginTop: 10, marginBottom: 0 }}>
        逐字时间以识别词流为准（识别器直接听到的字就是一次测量），开了「强制对齐逐字」再由它核对并兜住识别没听到的字。
        默认每个字独立高亮；打开「按语意合并成词」才把整份结果交给 LLM，由它判断每个词在哪里断开，
        合并后的单元时间取它第一个字的起点和最后一个字的终点——LLM 只决定断词，不产生任何时间。歌词源自带的时间轴只用来估计每行大致位置（标为「参考定位」）；
        识别听到但歌词里没有的内容默认丢弃（只输出歌词行）；打开「识别现场加唱」才保留为「现场加唱」行。
        翻唱只唱了整首歌的一部分时（TV-size、只唱副歌），这段录音没唱到的那几段歌词会直接从时间轴去掉，
        不会留成「没有时间」的行；只丢一行的情况不算，那多半是识别漏了，仍然保留并标记。
        识别没听到的行保持无时间，补出来的字标记为「插值」。
        结果按设计需要人工试听确认（时间轴里的「待确认」）。
      </p>
    </div>
  );
}
