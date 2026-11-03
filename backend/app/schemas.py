"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TimelineStatus = Literal["none", "untimed", "line_only", "partial", "complete"]
TaskStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
ProjectStatus = Literal["queued", "processing", "ready", "needs_review", "failed"]


class Word(BaseModel):
    text: str = ""
    start: float | None = None
    end: float | None = None
    source: str | None = None


class Line(BaseModel):
    text: str = ""
    start: float | None = None
    end: float | None = None
    words: list[Word] = Field(default_factory=list)
    source: str | None = None
    note: str | None = None
    """Why the line was flagged for a human listen; never a rewritten lyric."""
    heard: str | None = None
    """What the ASR actually heard for this line, kept for comparison.

    The displayed text stays the original lyric (misheard kanji are noise), but
    on a live version the sung words can genuinely differ, and this is the
    evidence a human needs to decide.
    """
    reviewed: bool = False
    """Set by the user once they have listened to a flagged line."""


class Timeline(BaseModel):
    """The canonical editable representation of timed lyrics.

    All times are seconds (float) with millisecond precision in JSON.
    """

    status: TimelineStatus = "none"
    lines: list[Line] = Field(default_factory=list)
    updated_at: str | None = None
    notes: list[str] = Field(default_factory=list)
    """Run-level observations (e.g. ASR heard words no lyric line covers)."""

    @field_validator("lines")
    @classmethod
    def _check_order(cls, lines: list[Line]) -> list[Line]:
        return lines


class SongMetadata(BaseModel):
    title: str = ""
    artist: str = ""
    album: str = ""
    duration: float | None = None
    source: str = "manual"


class TaskOut(BaseModel):
    id: str
    project_id: str
    kind: str
    status: TaskStatus
    stage: str = ""
    progress: float = 0
    message: str = ""
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class CandidateOut(BaseModel):
    id: str
    project_id: str
    lrclib_id: int | None = None
    track_name: str = ""
    artist_name: str = ""
    album_name: str = ""
    duration: float | None = None
    instrumental: bool = False
    score: float = 0
    selected: bool = False
    has_synced_lyrics: bool = False
    has_plain_lyrics: bool = False
    synced_lyrics: str | None = None
    plain_lyrics: str | None = None


class ReferenceLine(BaseModel):
    """One line of the lyrics source's own timeline (e.g. the studio LRC).

    Shown next to the aligned line as a positional hint, never written into the
    output timings: the recording may be a live version that runs at another pace.
    """

    index: int
    """Index into ``original_lyrics`` lines, or -1 when the line did not match."""
    text: str = ""
    start: float | None = None
    end: float | None = None


class ProjectSummary(BaseModel):
    id: str
    title: str
    artist: str
    album: str
    duration: float | None = None
    status: ProjectStatus
    progress: float = 0
    error: str | None = None
    metadata: SongMetadata
    lyrics_source: str | None = None
    timeline_status: TimelineStatus = "none"
    audio_url: str
    created_at: str
    updated_at: str


class ProjectOut(ProjectSummary):
    original_filename: str = ""
    audio_filename: str = ""
    audio_size: int = 0
    audio_mime: str = ""
    metadata_source: str = "manual"
    search_query: str | None = None
    selected_candidate_id: str | None = None
    original_lyrics: str = ""
    revised_lyrics: str = ""
    lyrics_lrclib_id: int | None = None
    """LRCLIB entry the original lyrics came from, if any."""
    lyrics_external_id: str | None = None
    """Id in ``lyrics_source`` when that source is not LRCLIB."""
    lyrics_language: str | None = None
    """Language sent to the ASR, or ``None`` when it auto-detects."""
    language_origin: str | None = None
    """``manual`` / ``llm`` / ``script`` / ``auto``: how the language was chosen."""
    separate_vocals: bool = True
    """Whether the next run separates the vocals before recognising them."""
    asr_align: bool = False
    """Whether the ASR service's own forced aligner produces the word stream.

    Off by default: the recogniser's plain transcription is re-tokenised per
    character. This only decides *which* recogniser positions the lines; the
    per-character times come from ``hfa_chars``.
    """
    hfa_timing: bool = False
    """Whether the external forced aligner decides every time, instead of the recogniser.

    Off by default: the recogniser places each line, which measured more
    accurately against the official 逐字 timings. On, the aligner-first path cuts
    the vocals into sung stretches and aligns one lyric line per stretch; the ASR
    then only says whether this take sang anything the lyrics do not contain.
    """
    hfa_chars: bool = False
    """Whether the forced aligner takes part in the per-character times.

    Off by default: the per-character table is the recogniser's own and no
    aligner request is made (0.140 s against the official 逐字 times on the six
    songs in ``/root/yrc``). On, the aligner also places every character line by
    line and the two are fused where they agree (``char_align.FUSE_TOLERANCE``),
    which measures 0.085 s — at the cost of one request per line.
    """
    detect_extras: bool = False
    """Whether the aligner-first flow runs the recogniser to look for live extras.

    Only read when ``hfa_timing`` is on. Off means: no ASR, no transcript, and
    the timeline is the original lyrics placed by the aligner. On means: the
    transcript is compared with the lyrics, and content the lyrics do not contain
    (a repeated chorus, an ad-lib) is added back and re-aligned by the aligner.
    The ASR-timed flow always transcribes, since its timings are the point.
    """
    llm_merge: bool = False
    """Whether the finished timeline is grouped into words by the LLM.

    Off by default: the per-character times are what was measured, and one
    time per character is the finest highlight there is. On, the LLM says where
    each word ends and every unit takes the span of the characters it covers —
    it is never asked for a time (see ``llm_merge``).
    """
    word_sync_search: bool = False
    """Whether to look for 逐字 lyrics in other editions of the same song.

    Off by default: the search costs a second chain of source queries, and it
    only pays off when the entry the lyrics came from is the studio cut with no
    ``yrc`` while a live edition has one. On keeps the run to the one entry the
    lyrics came from.
    """
    fetch_lyrics: bool = True
    """Whether a run with no pasted lyrics looks the song up in the source chain.

    On by default: the matcher fetches the original lyrics from NetEase / LRCLIB /
    LrcApi automatically. Off, the run never asks the sources — it aligns the ASR
    transcript (or the lyrics the user did paste) and nothing is fetched behind
    the user's back.
    """
    line_split: bool = True
    """Whether a run with no lyrics asks the LLM where the lines break.

    On by default: the recogniser's transcript is cut into sung lines at phrase
    boundaries instead of only at pauses. The LLM decides **cuts only** — it is
    never asked for text or a time, and an answer that does not reproduce the
    transcript exactly is discarded, so turning it on cannot cost a run (see
    ``services/line_split``). Off keeps the pause-based split.
    """
    line_split_max_chars: int | None = None
    """The characters per line when there are no lyrics, or ``None`` for per language.

    ``None`` (the default) uses the configured table: 12 for Chinese and
    Cantonese, 18 for Japanese and Korean, 24 for everything else. A number
    overrides the table for this project. It is a ceiling, not a target — the
    LLM may cut earlier, and a cut that would exceed it is rejected.
    """
    lyrics_reference: bool = False
    """LRCLIB supplied a synced timeline; used as a positional hint only."""
    lyrics_reference_lrc: str | None = None
    """The source's own LRC (studio timings included), kept verbatim for reference."""
    lyrics_word_sync: bool = False
    """The lyrics source supplied word-level (逐字) timings for this song."""
    lyrics_word_sync_edition: str | None = None
    """When the timings came from another edition of the song, which one."""
    reference_lines: list[ReferenceLine] = Field(default_factory=list)
    """That timeline matched onto ``original_lyrics`` line by line."""
    timeline: Timeline | None = None


class ProjectList(BaseModel):
    items: list[ProjectSummary]
    total: int


class ProjectDelete(BaseModel):
    """A batch of project ids to delete, with everything each one owns."""

    ids: list[str] = Field(min_length=1)


class JobDelete(BaseModel):
    """A batch of one-shot job ids to delete, records and audio alike."""

    ids: list[str] = Field(min_length=1)


class OneShotResult(BaseModel):
    """A cached job's result, as the browser may have edited it.

    The shape is what ``OneShotJobs._build_result`` writes; the editor works on
    the same per-character lines, so a correction round-trips without any
    translation layer. ``words`` is the flat list of every timed token and
    ``lines`` carries the per-character spans.
    """

    language: str | None = None
    time_unit: str = "seconds"
    text_source: Literal["lyrics", "asr"] = "asr"
    notes: list[str] = Field(default_factory=list)
    words: list[Word] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)


class ProjectUpdate(BaseModel):
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration: float | None = None
    metadata_source: str | None = None
    separate_vocals: bool | None = None
    asr_align: bool | None = None
    hfa_timing: bool | None = None
    hfa_chars: bool | None = None
    detect_extras: bool | None = None
    llm_merge: bool | None = None
    word_sync_search: bool | None = None
    fetch_lyrics: bool | None = None
    line_split: bool | None = None
    line_split_max_chars: int | None = None

    @field_validator("line_split_max_chars")
    @classmethod
    def _sane_char_limit(cls, value: int | None) -> int | None:
        # 0 or less would reject every answer; the splitter floors it anyway,
        # but saying so here gives a 422 instead of a silently odd run.
        if value is not None and value < 1:
            raise ValueError("line_split_max_chars must be at least 1")
        return value


class LyricsUpdate(BaseModel):
    original_lyrics: str | None = None
    revised_lyrics: str | None = None
    language: str | None = None
    lyrics_source: str | None = None
    lyrics_lrclib_id: int | None = None


class LyricsCandidateOut(BaseModel):
    lrclib_id: int | None = None
    external_id: str | None = None
    """Provider-specific id, used to select a non-LRCLIB entry."""
    source: str = "lrclib"
    track_name: str = ""
    artist_name: str = ""
    album_name: str = ""
    duration: float | None = None
    score: float = 0
    line_count: int = 0
    preview: str = ""
    synced: bool = False
    """The API entry has timestamps; they are kept only as a positional hint."""
    word_sync: bool = False
    """The entry carries word-level (逐字) timings, usable if they match the audio."""


class LyricsSearchRequest(BaseModel):
    query: str | None = None
    """Free-text override; by default the search uses the project's metadata."""


class LyricsSearchResponse(BaseModel):
    source: str = "lrclib"
    """Primary source; candidates also carry their own ``source``."""
    sources: list[str] = Field(default_factory=list)
    query: dict[str, str | None] = Field(default_factory=dict)
    candidates: list[LyricsCandidateOut] = Field(default_factory=list)


class LyricsSelectRequest(BaseModel):
    lrclib_id: int | None = None
    external_id: str | None = None
    source: str = "lrclib"
    """``lrclib``, ``lrcapi`` or ``netease``."""


class SearchRequest(BaseModel):
    track_name: str | None = None
    artist_name: str | None = None
    album_name: str | None = None
    query: str | None = None
    auto_select: bool | None = None


class SearchResponse(BaseModel):
    query: dict[str, str | None]
    candidates: list[CandidateOut]
    selected_candidate_id: str | None = None


class SelectCandidateRequest(BaseModel):
    candidate_id: str | None = None
    """Select a stored candidate. Mutually exclusive with ``candidate``."""
    candidate: dict[str, Any] | None = None
    """Ad-hoc candidate payload (e.g. pasted LRCLIB result) to select directly."""


class TimelineUpdate(BaseModel):
    timeline: Timeline
    """Full replacement. Send the whole timeline so autosave is idempotent."""


class TimelineInitRequest(BaseModel):
    use_revised: bool = True


class AlignRequest(BaseModel):
    """Kick off ASR + LLM alignment + forced-aligner per-character alignment."""

    reset_cache: bool = False
    """Ignore the cached ASR result and transcribe the audio again."""

    separate: bool | None = None
    """Separate the vocals before ASR.

    ``None`` keeps the project's current setting, which defaults to on.
    Turning it off recognises the raw mix directly: faster, but a loud backing
    track can be misheard as words. Changing it invalidates the cached ASR
    result, which belongs to the other audio.
    """

    asr_align: bool | None = None
    """Use the ASR service's own forced aligner for the word stream.

    ``None`` keeps the project's current setting, which defaults to off: the
    recogniser's plain transcription is re-tokenised per character. This only
    decides which recogniser positions the lines; ``hfa_chars`` decides the
    character times.
    """

    hfa_chars: bool | None = None
    """Whether the forced aligner also places every character (off by default).

    ``None`` keeps the project's current setting. Off, per-character times are
    the recogniser's own table; on, the aligner places them too and the two are
    fused where they agree.
    """

    hfa_timing: bool | None = None
    """Let the forced aligner decide every time instead of the recogniser.

    ``None`` keeps the project's current setting, which defaults to off (the
    recogniser places each line; measured more accurate against the official
    逐字 timings). On, the whole song goes to the aligner block by block.
    """

    word_sync_search: bool | None = None
    """Look for 逐字 lyrics in other editions of the same song.

    ``None`` keeps the project's current setting, which defaults to off. Off
    restricts the run to the entry the lyrics came from: a payload already
    cached from another edition is dropped, since it would otherwise be reused
    without searching.
    """
    fetch_lyrics: bool | None = None
    """Look the song up in the source chain when no lyrics were pasted.

    ``None`` keeps the project's current setting, which defaults to on. Off, the
    run never asks NetEase / LRCLIB / LrcApi and aligns the ASR transcript (or
    the lyrics the user did paste) instead.
    """

    line_split: bool | None = None
    """Ask the LLM where the lines break when there are no lyrics.

    ``None`` keeps the project's current setting, which defaults to on. The LLM
    only decides cuts; an answer that does not reproduce the transcript is
    discarded, so this cannot cost a run (see ``services/line_split``).
    """

    line_split_max_chars: int | None = None
    """Ceiling on the characters per line when there are no lyrics.

    ``None`` keeps the per-language table (Chinese 12, Japanese 18, else 24);
    a number overrides it for this run. Only read when ``line_split`` is on.
    """


class ExportFormat(BaseModel):
    format: Literal["json", "lrc", "ass"]
    precision_ms: float
    notes: str


class DebugStep(BaseModel):
    """One intermediate result of an alignment run."""

    step: str
    payload: Any = None
    created_at: str


class DebugRun(BaseModel):
    run_id: str
    created_at: str
    """Timestamp of the run's first step."""
    finished_at: str | None = None
    """Timestamp of the run's last step, if it recorded more than one."""
    steps: list[DebugStep] = Field(default_factory=list)


class DebugOut(BaseModel):
    project_id: str
    order: list[str] = Field(default_factory=list)
    """The canonical step order; runs are rendered in this order."""
    runs: list[DebugRun] = Field(default_factory=list)
    """Newest run first."""


class ProviderInfo(BaseModel):
    name: str
    kind: str
    available: bool
    detail: str = ""


class ProvidersOut(BaseModel):
    providers: list[ProviderInfo]


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    lrclib_url: str
