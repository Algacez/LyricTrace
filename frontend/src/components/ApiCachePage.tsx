import { useCallback, useEffect, useMemo, useState } from "react";
import { api, errorMessage, type OneShotJob, type OneShotResult } from "../api";
import { computeStatus } from "../utils";
import type { Project, Timeline } from "../types";
import { TimelineEditor } from "./TimelineEditor";

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

/** A cached result as a ``Timeline``, so the same editor can correct it. */
function timelineOf(result: OneShotResult | null): Timeline | null {
  if (!result) return null;
  const timeline: Timeline = {
    status: "partial",
    lines: result.lines.map((line) => ({
      text: line.text,
      start: line.start,
      end: line.end,
      words: line.words.map((word) => ({ ...word, source: word.source ?? undefined })),
      source: line.source ?? undefined,
      note: line.note ?? undefined,
    })),
    notes: result.notes,
  };
  return { ...timeline, status: computeStatus(timeline) };
}

/** Back from the editor to the job's stored result shape. */
function resultOf(job: OneShotJob, timeline: Timeline): OneShotResult {
  const base = job.result;
  return {
    language: base?.language ?? null,
    time_unit: base?.time_unit ?? "seconds",
    text_source: base?.text_source ?? "asr",
    notes: timeline.notes ?? base?.notes ?? [],
    // The flat list is every timed token in line order, so a correction that
    // moved a character inside its line moves it here too.
    words: timeline.lines.flatMap((line) =>
      line.words.map((word) => ({
        text: word.text,
        start: word.start,
        end: word.end,
        source: word.source ?? null,
      })),
    ),
    lines: timeline.lines.map((line) => ({
      text: line.text,
      start: line.start,
      end: line.end,
      source: line.source ?? null,
      note: line.note ?? null,
      words: line.words.map((word) => ({
        text: word.text,
        start: word.start,
        end: word.end,
        source: word.source ?? null,
      })),
    })),
  };
}

/** How many cached jobs a page holds. Every list in the app pages. */
const PAGE_SIZE = 20;

/**
 * The API's history: every `/api/align` request keeps its finished 逐字 result
 * here, so a result can be listened to again instead of being re-aligned. The
 * server does the keeping (see `services/housekeeping`) — job records are never
 * swept, and a job's uploaded source is only dropped once the request is older
 * than `oneshot_max_age_hours` (the result stays readable either way).
 */
export function ApiCachePage() {
  const [jobs, setJobs] = useState<OneShotJob[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  // How long a finished job keeps its uploaded source; the result outlives it.
  const [maxAgeHours, setMaxAgeHours] = useState(24);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  // Which jobs are ticked in the left column's checkboxes.
  const [checked, setChecked] = useState<string[]>([]);
  // Whether a correction is on its way to the server, and whether it landed.
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await api.listAlignJobs({ limit: PAGE_SIZE, offset });
      setJobs(data.items);
      setTotal(data.total);
      setMaxAgeHours(data.max_age_hours);
      setSelected((current) =>
        current && data.items.some((job) => job.job_id === current) ? current : data.items[0]?.job_id ?? null,
      );
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [offset]);

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
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const lines = useMemo(() => linesOf(job?.result ?? null), [job]);
  const duration = useMemo(() => {
    let end = 0;
    for (const line of lines) end = Math.max(end, line.end ?? line.start ?? 0);
    return end;
  }, [lines]);

  // The editor takes a ``Project`` — it reads the timeline, the reference lines
  // and the audio url. A cached job has no lyrics source, so the reference
  // lines are empty and the audio comes from the job's own upload (``null``
  // once the sweep has dropped it). Everything else here is a label.
  const jobEditorProject: Project | null = useMemo(() => {
    if (!job) return null;
    return {
      id: job.job_id,
      title: job.filename || job.job_id,
      artist: "",
      album: "",
      duration: duration || null,
      status: "ready",
      progress: 1,
      error: null,
      metadata: { title: job.filename || "", artist: "", album: "", duration: duration || null, source: "manual" },
      lyrics_source: null,
      timeline_status: "none",
      audio_url: api.alignJobAudioUrl(job.job_id),
      created_at: job.created_at ?? "",
      updated_at: job.finished_at ?? job.created_at ?? "",
      original_filename: job.filename || "",
      audio_filename: job.filename || "",
      audio_size: 0,
      audio_mime: "",
      metadata_source: "manual",
      original_lyrics: "",
      revised_lyrics: "",
      lyrics_lrclib_id: null,
      lyrics_reference: false,
      timeline: timelineOf(job.result),
      reference_lines: [],
    };
  }, [job, duration]);

  const toggle = (jobId: string) =>
    setChecked((current) =>
      current.includes(jobId) ? current.filter((id) => id !== jobId) : [...current, jobId],
    );

  const removeChecked = async () => {
    if (checked.length === 0) return;
    if (!window.confirm(`删除选中的 ${checked.length} 条 API 记录？记录与上传的音频都会被删掉，此操作不可撤销。`))
      return;
    try {
      await api.deleteAlignJobs(checked);
      setChecked([]);
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const removeOne = async (jobId: string) => {
    if (!window.confirm("删除这条 API 记录？记录与上传的音频都会被删掉，此操作不可撤销。")) return;
    try {
      await api.deleteAlignJob(jobId);
      setChecked((current) => current.filter((id) => id !== jobId));
      await load();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  return (
    <div className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div className="row">
          <h1 style={{ fontSize: 20, margin: 0 }}>API 缓存</h1>
          <span className="muted small">
            全部 {total} 次 <code>/api/align</code> 请求的记录与逐字结果
            {total > PAGE_SIZE && `（第 ${Math.floor(offset / PAGE_SIZE) + 1} / ${pages} 页）`}
          </span>
        </div>
        <div className="row">
          <button onClick={() => void load()}>刷新</button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      {loading ? (
        <p className="muted">加载中…</p>
      ) : jobs.length === 0 ? (
        <div className="card">
          <p className="muted" style={{ margin: 0 }}>
            还没有 API 请求。调用 <code>POST /api/align</code> 之后，每次请求的记录和逐字结果都会留在这里，
            可以直接播放核对；只有超过 {maxAgeHours} 小时的请求会被清掉上传的音频，结果本身一直可以读。
          </p>
        </div>
      ) : (
        <div className="editor-grid">
          <div className="card">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
              <h2 style={{ marginTop: 0, marginBottom: 0 }}>缓存列表</h2>
              <div className="row small" style={{ gap: 6 }}>
                <label className="row small muted" style={{ gap: 4 }}>
                  <input
                    type="checkbox"
                    checked={jobs.length > 0 && checked.length === jobs.length}
                    onChange={(event) => setChecked(event.target.checked ? jobs.map((item) => item.job_id) : [])}
                  />
                  全选本页
                </label>
                <button className="danger ghost" disabled={checked.length === 0} onClick={() => void removeChecked()}>
                  删除选中{checked.length > 0 ? `（${checked.length}）` : ""}
                </button>
              </div>
            </div>
            <ul className="cache-list">
              {jobs.map((item) => (
                <li key={item.job_id} className="cache-row">
                  <input
                    type="checkbox"
                    className="cache-check"
                    checked={checked.includes(item.job_id)}
                    onChange={() => toggle(item.job_id)}
                    title="选中后批量删除"
                  />
                  <button
                    className={`cache-item ${item.job_id === selected ? "selected" : ""}`}
                    onClick={() => setSelected(item.job_id)}
                  >
                    <span className="title">
                      {item.filename || "（未命名）"}
                      <span className={`badge ${item.status === "succeeded" ? "ok" : item.status === "failed" ? "err" : "info"}`}>
                        {STATUS_TEXT[item.status]}
                      </span>
                      {item.edited_at && <span className="badge info">已修正</span>}
                    </span>
                    <span className="meta small muted">
                      {(item.created_at ?? "").replace("T", " ").slice(0, 19)}
                      {item.result ? ` · ${item.result.lines.length} 行 / ${item.result.words.length} 个字` : ""}
                      {item.with_lyrics ? " · 带歌词" : " · 文本来自识别"}
                      {item.language ? ` · ${item.language}` : ""}
                      {item.with_audio ? "" : " · 音频已清理"}
                    </span>
                  </button>
                  <button
                    className="cache-delete"
                    title="删除这条记录与它上传的音频"
                    onClick={() => void removeOne(item.job_id)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
            {total > PAGE_SIZE && (
              <div className="row small" style={{ gap: 8, marginBottom: 8 }}>
                <button disabled={page <= 1} onClick={() => setOffset(Math.max(offset - PAGE_SIZE, 0))}>
                  上一页
                </button>
                <span className="muted">
                  第 {page} / {pages} 页
                </span>
                <button
                  disabled={page >= pages}
                  onClick={() => setOffset(Math.min(offset + PAGE_SIZE, (pages - 1) * PAGE_SIZE))}
                >
                  下一页
                </button>
              </div>
            )}
            <p className="muted small" style={{ marginBottom: 0 }}>
              记录和结果永久保留，可以随时重新读取；上传的音频只保留最近 {maxAgeHours}{" "}
              小时，更早的条目仍能看结果、只是不能播放。选中的记录连记录一起删掉，
              上传的音频也随之删除。
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

                {/* The player is outside the ``job.result`` guard: a job that died
                    before it could align still keeps the audio it was given, and
                    listening to it is how one tells a bad file from a bad run. */}
                {job.with_audio && (
                  <audio
                    src={api.alignJobAudioUrl(job.job_id)}
                    controls
                    style={{ width: "100%" }}
                  />
                )}

                {job.result && jobEditorProject && (
                  <>
                    <p className="muted small" style={{ marginTop: 0 }}>
                      {job.result.lines.length} 行 · {job.result.words.length} 个字
                      {saving ? " · 保存中…" : saved ? " · 已保存" : ""}
                      {" · 改完会自动写回这条记录"}
                    </p>
                    {!job.with_audio && (
                      <p className="muted small">
                        这条请求上传的音频已按保留时限清理，逐字结果仍可查看和修正；重新提交同一文件即可再次播放。
                      </p>
                    )}

                    {/* The same editor the project page uses: per-character times,
                        the slider, the keyboard stamps. Saving writes the corrected
                        result back into the job record, so a correction survives a
                        reload instead of living only in this tab. */}
                    <TimelineEditor
                      key={`${job.job_id}-${job.edited_at ?? ""}`}
                      project={jobEditorProject}
                      onProjectChange={() => {}}
                      audioUrl={job.with_audio ? api.alignJobAudioUrl(job.job_id) : null}
                      save={async (timeline) => {
                        setSaving(true);
                        try {
                          await api.saveAlignResult(job.job_id, resultOf(job, timeline));
                          setSaved(true);
                          window.setTimeout(() => setSaved(false), 2000);
                          await load();
                        } finally {
                          setSaving(false);
                        }
                      }}
                    />
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
