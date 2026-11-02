export type TimelineStatus = "none" | "untimed" | "line_only" | "partial" | "complete";
export type TaskStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type ProjectStatus = "queued" | "processing" | "ready" | "needs_review" | "failed";

export interface Word {
  text: string;
  start: number | null;
  end: number | null;
  /**
   * `asr` = the recogniser's own character time — what a normal run produces:
   * the ASR word stream is the per-character source, and a character it timed
   * itself is kept whenever the forced aligner agrees within 0.3 s. `aligner` =
   * placed by the forced aligner, which keeps the characters the recogniser only
   * interpolated, or placed somewhere else in the song. `interpolated` = filled
   * between timed neighbours,
   * `word-sync` = from the lyrics source's own word timings, `manual` = edited
   * by hand in the timeline editor. `merged` = a unit the LLM grouped out of
   * several characters; its time is the span of the characters it covers.
   */
  source?: string | null;
}

export interface LyricLine {
  text: string;
  start: number | null;
  end: number | null;
  words: Word[];
  /** `asr` / `asr-partial` / `asr-bad` (LLM verdict) or `unmatched`. */
  source?: string | null;
  /** Why the line was flagged for a human listen. */
  note?: string | null;
  /** What the ASR heard on a flagged line, for comparison with the lyric. */
  heard?: string | null;
  /** Set once the user has listened to a flagged line. */
  reviewed?: boolean;
}

export interface Timeline {
  status: TimelineStatus;
  lines: LyricLine[];
  updated_at?: string | null;
  /** Run-level observations, e.g. ASR heard material the lyrics do not cover. */
  notes?: string[];
}

export interface SongMetadata {
  title: string;
  artist: string;
  album: string;
  duration: number | null;
  source: string;
}

export interface ProjectSummary {
  id: string;
  title: string;
  artist: string;
  album: string;
  duration: number | null;
  status: ProjectStatus;
  progress: number;
  error: string | null;
  metadata: SongMetadata;
  lyrics_source: string | null;
  timeline_status: TimelineStatus;
  audio_url: string;
  created_at: string;
  updated_at: string;
}

export interface Project extends ProjectSummary {
  original_filename: string;
  audio_filename: string;
  audio_size: number;
  audio_mime: string;
  metadata_source: string;
  original_lyrics: string;
  revised_lyrics: string;
  /** Id of the entry in `lyrics_source` (LRCLIB id, LrcApi/NetEase id). */
  lyrics_lrclib_id: number | null;
  lyrics_external_id?: string | null;
  /** Language handed to the ASR; null means it auto-detects. */
  lyrics_language?: string | null;
  /** `manual` / `llm` / `script` / `auto`: how the language was decided. */
  language_origin?: string | null;
  /** Whether the next run separates the vocals before recognising them. */
  separate_vocals?: boolean;
  /**
   * Whether the ASR service's own forced aligner produces the word stream.
   * Off by default: the recogniser's plain transcription is re-tokenised per
   * character; the aligner is weaker on singing (upstream: 80% zero-width
   * Japanese characters).
   */
  asr_align?: boolean;
  /**
   * Whether the external forced aligner decides every time instead of the
   * recogniser. Off by default: the recogniser places each line (measured more
   * accurately against the official 逐字 timings), and the aligner re-times its
   * characters.
   */
  hfa_timing?: boolean;
  /**
   * Whether the forced aligner also places every character. Off by default: the
   * per-character table is the recogniser's own and no aligner request is made.
   * On, both are measured and fused where they agree (more accurate, one request
   * per line).
   */
  hfa_chars?: boolean;
  /**
   * Whether the aligner-first flow runs the recogniser to look for live extras.
   * Only read when `hfa_timing` is on.
   */
  detect_extras?: boolean;
  /**
   * Whether the finished timeline is grouped into words by the LLM.
   * Off by default: every character keeps its own time and highlight. On, the
   * LLM only decides where words end; each unit takes the span of the
   * characters it covers (never a time of its own).
   */
  llm_merge?: boolean;
  /**
   * Whether the run looks for 逐字 lyrics in other editions of the same song.
   * On by default: the studio cut often has no 逐字 data, the live one does.
   */
  word_sync_search?: boolean;
  /** The source also supplied a timeline; it is a positional hint, never output. */
  lyrics_reference: boolean;
  /** That source LRC verbatim (studio timings included), for download. */
  lyrics_reference_lrc?: string | null;
  /** The lyrics source supplied word-level (逐字) timings for this song. */
  lyrics_word_sync?: boolean;
  /** When those timings came from another edition of the song, which one. */
  lyrics_word_sync_edition?: string | null;
  /** The same timeline, matched onto `original_lyrics` line by line. */
  reference_lines?: ReferenceLine[];
  timeline: Timeline | null;
}

/** One line of the lyrics source's own timeline, as a hint next to our line. */
export interface ReferenceLine {
  index: number;
  text: string;
  start: number | null;
  end: number | null;
}

export interface LyricsCandidate {
  lrclib_id: number | null;
  external_id?: string | null;
  /** `lrclib` / `lrcapi` / `netease`. */
  source: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  duration: number | null;
  score: number;
  line_count: number;
  preview: string;
  synced: boolean;
  /** The entry carries word-level (逐字) timings; used only if they match the audio. */
  word_sync?: boolean;
}

export interface LyricsSearchResult {
  source: string;
  sources?: string[];
  query: Record<string, string | null>;
  candidates: LyricsCandidate[];
}

export interface ProjectList {
  items: ProjectSummary[];
  total: number;
}

export interface Task {
  id: string;
  project_id: string;
  kind: string;
  status: TaskStatus;
  stage: string;
  progress: number;
  message: string;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Health {
  status: string;
  version: string;
  database: string;
  lrclib_url: string;
}

/** One intermediate result of an alignment run (see the debug page). */
export interface DebugStep {
  step: string;
  payload: unknown;
  created_at: string;
}

export interface DebugRun {
  run_id: string;
  created_at: string;
  finished_at: string | null;
  steps: DebugStep[];
}

export interface DebugTrace {
  project_id: string;
  /** Canonical step order; runs render their steps in this order. */
  order: string[];
  /** Newest run first. */
  runs: DebugRun[];
}
