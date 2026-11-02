import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errorMessage, type OneShotJob, type OneShotResult } from "../api";
import { formatTime } from "../utils";

/** A finished job's lines, in the shape the karaoke strip reads. */
interface CachedLine {
  text: string;
  start: number | null;
  end: number | null;
  words: { text: string; start: number | null; end: number | null }[];
}

function linesOf(result: OneShotResult | null): CachedLine[] {
  if (!result) return [];
  return result.lines.map((line) => ({
    text: line.text,
    start: line.start,
    end: line.end,
    words: line.words.length
      ? line.words
      : [{ text: line.text, start: line.start, end: line.end }],
  }));
}

/**
 * The API's answer cache: the newest few `/api/align` requests keep their
 * source file and their finished 逐字 result, so a result can be listened to
 * again here instead of being re-aligned. The server does the keeping (see
 * `services/housekeeping`); this page only reads it.
 */
export function ApiCachePage() {
  const [jobs, setJobs] = useState<OneShotJob[]>([]);
  const [maxJobs, setMaxJobs] = useState(5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.listAlignJobs();
      setJobs(data.items);
      setMaxJobs(data.max_jobs);
      setSelected((current) =>
        current && data.items.some((job) => job.job_id === current) ? current : data.items[0]?.job_id ?? null,
      );
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // A job still running will change state; poll while one is.
  useEffect(() => {
    const pending = jobs.some((job) => job.status === "queued" || job.status === "running");
    if (!pending) return;
    const timer = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(timer);
  }, [jobs, load]);

  const job = jobs.find((item) => item.job_id === selected) ?? null;
  const lines = useMemo(() => linesOf(job?.result ?? null), [job]);
  const duration = useMemo(() => {
    let end = 0;
    for (const line of lines) end = Math.max(end, line.end ?? line.start ?? 0);
    return end;
  }, [lines]);

  // The strip follows the same rule as the timeline editor: a character is lit
  // from its own start until its own end, and the line stays on the last
  // character that started.
  const active = (() => {
    let line = -1;
    let start = -1;
    lines.forEach((item, lineIndex) => {
      for (const word of item.words) {
        if (word.start === null || word.start > playhead + 0.001) continue;
        if (word.start >= start) {
          start = word.start;
          line = lineIndex;
        }
      }
    });
    return line;
  })();

  const seek = (time: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = time;
    setPlayhead(time);
  };

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div className="row">
          <h1 style={{ fontSize: 20, margin: 0 }}>API 缓存</h1>
          <span className="muted small">
            最近 {maxJobs} 次 <code>/api/align</code> 请求的源文件与逐字结果
          </span>
        </div>
        <button onClick={() => void load()}>刷新</button>
      </div>

      {error && <p className="error-text">{error}</p>}

      {loading ? (
        <p className="muted">加载中…</p>
      ) : jobs.length === 0 ? (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            还没有 API 请求。调用 <code>POST /api/align</code> 之后，最近的 {maxJobs} 次会连同源文件和逐字结果留在这里，
            可以直接播放核对；更早的会被清理线程删掉。
          </p>
        </div>
      ) : (
        <div className="editor-grid">
          <div className="card">
            <h2 style={{ marginTop: 0 }}>缓存列表</h2>
            <ul className="cache-list">
              {jobs.map((item) => (
                <li key={item.job_id}>
                  <button
                    className={`cache-item ${item.job_id === selected ? "selected" : ""}`}
                    onClick={() => {
                      setSelected(item.job_id);
                      setPlayhead(0);
                      setPlaying(false);
                    }}
                  >
                    <span className="title">
                      {item.filename || "（未命名）"}
                      <span className={`badge ${item.status === "succeeded" ? "ok" : item.status === "failed" ? "err" : "info"}`}>
                        {STATUS_TEXT[item.status]}
                      </span>
                    </span>
                    <span className="meta small muted">
                      {(item.created_at ?? "").replace("T", " ").slice(0, 19)}
                      {item.result ? ` · ${item.result.lines.length} 行 / ${item.result.words.length} 个字` : ""}
                      {item.with_lyrics ? " · 带歌词" : " · 文本来自识别"}
                      {item.language ? ` · ${item.language}` : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <p className="muted small" style={{ marginBottom: 0 }}>
              只保留最近 {maxJobs} 次：更早的源文件和结果会被自动清理，接口调用本身不受影响。
            </p>
          </div>

          <div className="card">
            {!job ? (
              <p className="muted">选择左边一条缓存。</p>
            ) : (
              <>
                <h2 style={{ marginTop: 0 }}>
                  {job.filename || job.job_id}
                  <span className={`badge ${job.status === "succeeded" ? "ok" : job.status === "failed" ? "err" : "info"}`}>
                    {STATUS_TEXT[job.status]}
                  </span>
                  {job.result && <span className="badge info">{job.result.text_source === "lyrics" ? "歌词文本" : "识别文本"}</span>}
                </h2>

                {job.status === "failed" && <p className="error-text small">失败：{job.error}</p>}
                {(job.status === "queued" || job.status === "running") && (
                  <p className="muted small">
                    {job.message || "处理中"} · {Math.round((job.progress ?? 0) * 100)}%
                  </p>
                )}

                {job.result && (
                  <>
                    <audio
                      ref={audioRef}
                      src={api.alignJobAudioUrl(job.job_id)}
                      controls
                      style={{ width: "100%" }}
                      onTimeUpdate={(event) => setPlayhead(event.currentTarget.currentTime)}
                      onPlay={() => setPlaying(true)}
                      onPause={() => setPlaying(false)}
                      onEnded={() => setPlaying(false)}
                    />
                    <div className="row small" style={{ justifyContent: "space-between", marginTop: 6 }}>
                      <span className="mono">
                        {formatTime(playhead)} / {formatTime(duration)}
                      </span>
                      <span className="muted">
                        {job.result.lines.length} 行 · {job.result.words.length} 个字
                        {playing ? " · 播放中" : ""}
                      </span>
                    </div>

                    {active >= 0 && (
                      <div className="karaoke-line" style={{ marginTop: 10 }}>
                        {lines[active].words.map((word, index) => {
                          const started = word.start !== null && word.start <= playhead + 0.001;
                          const lit = started && (word.end === null || playhead < word.end);
                          return (
                            <span
                              key={index}
                              className={`karaoke-word ${lit ? "active" : ""} ${started && !lit ? "sung" : ""}`}
                            >
                              {word.text}
                            </span>
                          );
                        })}
                      </div>
                    )}

                    {job.result.notes.length > 0 && (
                      <ul className="run-notes" style={{ marginTop: 10 }}>
                        {job.result.notes.map((note, index) => (
                          <li key={index}>{note}</li>
                        ))}
                      </ul>
                    )}

                    <div className="cache-lines">
                      {lines.map((line, index) => (
                        <div
                          key={index}
                          className={`line-row ${index === active ? "active" : ""}`}
                          onClick={() => seek(line.start ?? 0)}
                          title="点击跳到这一行"
                        >
                          <div className="row small" style={{ justifyContent: "space-between" }}>
                            <span className="mono muted">
                              {formatTime(line.start)} – {formatTime(line.end)}
                            </span>
                          </div>
                          <div className="cache-words">
                            {line.words.map((word, wordIndex) => {
                              const started = word.start !== null && word.start <= playhead + 0.001;
                              const lit = started && (word.end === null || playhead < word.end);
                              return (
                                <span
                                  key={wordIndex}
                                  className={`karaoke-word ${lit ? "active" : ""} ${started && !lit ? "sung" : ""}`}
                                >
                                  {word.text}
                                </span>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const STATUS_TEXT: Record<OneShotJob["status"], string> = {
  queued: "排队中",
  running: "处理中",
  succeeded: "已完成",
  failed: "失败",
};
