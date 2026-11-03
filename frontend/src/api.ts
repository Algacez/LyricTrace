import type {
  DebugTrace,
  Health,
  LyricsSearchResult,
  Project,
  ProjectList,
  Task,
  Timeline,
} from "./types";
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail: unknown = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail ?? payload;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export interface OneShotJob {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  created_at: string | null;
  finished_at: string | null;
  with_lyrics: boolean;
  /** The name the caller uploaded it under. */
  filename: string | null;
  language: string | null;
  result: OneShotResult | null;
  /** Whether the uploaded source is still on disk (playable here). The result
   * outlives it: past ``oneshot_max_age_hours`` only the upload is swept. */
  with_audio: boolean;
  /** What this request asked of the no-lyrics line splitter. */
  line_split: boolean;
  /** Its own characters-per-line ceiling, or ``null`` for the per-language one. */
  line_split_max_chars: number | null;
  /** When the result was last corrected in the browser, if it was. */
  edited_at?: string | null;
}

export interface OneShotWord {
  text: string;
  start: number | null;
  end: number | null;
  source: string | null;
}

export interface OneShotLine {
  text: string;
  start: number | null;
  end: number | null;
  source: string | null;
  note: string | null;
  words: OneShotWord[];
}

export interface OneShotResult {
  language: string | null;
  time_unit: string;
  text_source: "lyrics" | "asr";
  notes: string[];
  words: OneShotWord[];
  lines: OneShotLine[];
}

export const api = {
  health: () => request<Health>("/api/health"),

  listProjects: (params: { q?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.q) query.set("q", params.q);
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    if (params.offset !== undefined) query.set("offset", String(params.offset));
    const suffix = query.toString();
    return request<ProjectList>(`/api/projects${suffix ? `?${suffix}` : ""}`);
  },

  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Project>("/api/projects", { method: "POST", body: form });
  },

  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  /** Delete several projects at once, with their audio and work dirs. */
  deleteProjects: (ids: string[]) =>
    request<void>(`/api/projects/delete`, {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  deleteAlignJob: (jobId: string) => request<void>(`/api/align/${jobId}`, { method: "DELETE" }),
  /** Delete several cached jobs at once, records and audio alike. */
  deleteAlignJobs: (ids: string[]) =>
    request<void>(`/api/align/delete`, {
      method: "POST",
      body: JSON.stringify({ ids }),
    }),
  /** Write a hand-edited result back into a cached job. */
  saveAlignResult: (jobId: string, result: OneShotResult) =>
    request<OneShotJob>(`/api/align/${jobId}/result`, {
      method: "PUT",
      body: JSON.stringify(result),
    }),
  updateProject: (
    id: string,
    patch: Partial<
      Pick<
        Project,
        | "title"
        | "artist"
        | "album"
        | "separate_vocals"
        | "hfa_chars"
        | "asr_align"
        | "hfa_timing"
        | "detect_extras"
        | "llm_merge"
        | "word_sync_search"
        | "fetch_lyrics"
        | "line_split"
        | "line_split_max_chars"
      >
    >,
  ) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),

  /** The project's most recent tasks, newest first. Only the latest is read (to
   * follow a run), so the page keeps the response small. */
  tasks: (id: string, limit = 5) =>
    request<Task[]>(`/api/projects/${id}/tasks?limit=${limit}`),

  updateLyrics: (id: string, body: { original_lyrics?: string; language?: string }) =>
    request<Project>(`/api/projects/${id}/lyrics`, { method: "PUT", body: JSON.stringify(body) }),

  /** Search NetEase → LRCLIB → LrcApi for the original lyrics. */
  searchLyrics: (id: string, query?: string) =>
    request<LyricsSearchResult>(`/api/projects/${id}/lyrics/search`, {
      method: "POST",
      body: JSON.stringify(query ? { query } : {}),
    }),

  selectLyrics: (id: string, pick: { lrclib_id?: number; external_id?: string; source: string }) =>
    request<Project>(`/api/projects/${id}/lyrics/select`, {
      method: "POST",
      body: JSON.stringify(pick),
    }),

  align: (id: string, resetCache = false) =>
    request<Task>(`/api/projects/${id}/align`, {
      method: "POST",
      body: JSON.stringify({ reset_cache: resetCache }),
    }),

  saveTimeline: (id: string, timeline: Timeline) =>
    request<Project>(`/api/projects/${id}/timeline`, {
      method: "PUT",
      body: JSON.stringify({ timeline }),
    }),

  exportUrl: (id: string, format: "json" | "lrc" | "ass") => `${BASE}/api/projects/${id}/export?format=${format}`,
  /** Original lyrics; `lrc` keeps the source's own timeline when it has one. */
  lyricsExportUrl: (id: string, format: "lrc" | "txt" = "lrc") =>
    `${BASE}/api/projects/${id}/lyrics/export?format=${format}`,
  audioUrl: (id: string) => `${BASE}/api/projects/${id}/audio`,

  /** Every kept alignment run's intermediate results (newest first). */
  debugTrace: (id: string) => request<DebugTrace>(`/api/projects/${id}/debug`),

  /** One-shot API: audio (+ optional lyrics) in, word timestamps out. */
  startAlignJob: (
    file: File,
    options: {
      lyrics?: string;
      language?: string;
      separate?: boolean;
      word_sync_search?: boolean;
      detect_extras?: boolean;
      llm_merge?: boolean;
      fetch_lyrics?: boolean;
      line_split?: boolean;
      line_split_max_chars?: number | null;
    } = {},
  ) => {
    const form = new FormData();
    form.append("file", file);
    if (options.lyrics) form.append("lyrics", options.lyrics);
    if (options.language) form.append("language", options.language);
    form.append("separate", String(options.separate ?? true));
    if (options.word_sync_search !== undefined)
      form.append("word_sync_search", String(options.word_sync_search));
    if (options.detect_extras !== undefined)
      form.append("detect_extras", String(options.detect_extras));
    if (options.llm_merge !== undefined) form.append("llm_merge", String(options.llm_merge));
    if (options.fetch_lyrics !== undefined) form.append("fetch_lyrics", String(options.fetch_lyrics));
    if (options.line_split !== undefined) form.append("line_split", String(options.line_split));
    if (options.line_split_max_chars !== undefined && options.line_split_max_chars !== null)
      form.append("line_split_max_chars", String(options.line_split_max_chars));
    return request<OneShotJob>("/api/align", { method: "POST", body: form });
  },
  getAlignJob: (jobId: string) => request<OneShotJob>(`/api/align/${jobId}`),
  /** One page of the request history, newest first. Job records are never
   * swept; only the source file of a job older than the age limit goes
   * (``with_audio``). ``total`` counts everything, for the pager. */
  listAlignJobs: (params: { limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.limit !== undefined) query.set("limit", String(params.limit));
    if (params.offset !== undefined) query.set("offset", String(params.offset));
    const suffix = query.toString();
    return request<{ items: OneShotJob[]; total: number; limit: number; offset: number; max_jobs: number; max_age_hours: number }>(
      `/api/align${suffix ? `?${suffix}` : ""}`,
    );
  },
  /** The source file of a cached job, for playing its result back. */
  alignJobAudioUrl: (jobId: string) => `${BASE}/api/align/${jobId}/audio`,
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (typeof error.detail === "string") return error.detail;
    if (error.detail && typeof error.detail === "object" && "message" in error.detail) {
      return String((error.detail as { message: unknown }).message);
    }
    return JSON.stringify(error.detail);
  }
  if (error instanceof Error) return error.message;
  return String(error);
}
