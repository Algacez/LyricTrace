"""The processing pipeline: audio -> vocals -> ASR -> lyric alignment (+ extras check).

Two task kinds:

* ``pipeline`` (on upload) probes the audio, fills in metadata and fetches the
  original lyrics from the source chain (NetEase, then LRCLIB, then LrcApi).
  The user can still paste their own lyrics; those always win over the API.
* ``align`` runs the real work. Qwen3-ASR hears the audio, the kana DP maps
  every lyric line onto the tokens it heard, and the LLM verifies that mapping.
  The recogniser's per-character tokens are the timings (``char_align`` re-slices
  each line from the tokens heard over it, fills the gaps and spreads stacked
  characters); on singing the recogniser's own alignment is what the output is
  built on, so no external forced-aligner service is required.

Neither flow invents a time. A line nothing could place stays untimed and is
marked; a line the recogniser could not time says so in the run notes rather than
passing the word stream's split spans off as an alignment. The one source whose
own timings can reach the output is the lyrics source's 逐字 data, and only
through ``word_sync``'s gate (the ASR must have heard the same lines in the
same places).

A song with no lyrics anywhere is not exempt from any of that: its text is the
transcript, split at the recogniser's pauses, and the same forced aligner
re-times its characters (see ``_align_from_asr``). The recogniser's tokens are
word-sized, so without that pass ``君の`` would be one highlight block for two
characters.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from .. import db
from ..audio import cut_audio, probe_audio
from ..config import settings
from ..schemas import Line, Timeline, Word
from ..tasks import update_progress
from . import char_align, line_split, llm_merge, lyrics_fetch, word_stream, word_sync
from .asr_lines import (
    LINE_GAP_SEC,
    group_words,
    lines_from_groups,
    lines_from_words,
    repair_word_times,
)
from .asr_repair import (
    prepared_audio,
    repair_gaps,
    repair_note,
    resolve_extra_segments,
    retranscribe_span,
)
from .language import resolve as language_detect
from .llm import LLMClient
from .lyrics import parse_lrc, parse_plain_lyrics
from .projects import compute_timeline_status
from .reference import match_reference, reference_lines
from .providers.base import ASRResult, ProviderUnavailable, registry
from .providers.bs_roformer import to_asr_wav
from .trace import Trace

logger = logging.getLogger("lyrictrace.pipeline")


def pipeline_handler(task_id: str, task: dict[str, Any]) -> None:
    """Upload-time work: audio info + metadata. No network, no lyric guessing.

    Deliberately stops before the lyrics: uploading a file must not go looking
    for a song on the internet. Whether the sources are asked at all is decided
    by the 「自动获取歌词」 switch when the run starts (``align_handler``), so a
    file can be uploaded, listened to, and described without any lookup.
    """
    project_id = task["project_id"]
    with db.connect() as conn:
        project = db.fetch_project(conn, project_id)
    if project is None:
        raise RuntimeError("Project disappeared while processing.")

    audio_path = settings.audio_dir / project["audio_path"]
    update_progress(task_id, stage="audio", progress=0.2, message="Reading audio information")
    info = probe_audio(audio_path)

    fields: dict[str, Any] = {}
    if project["duration"] is None and info.duration:
        fields["duration"] = info.duration

    update_progress(task_id, stage="metadata", progress=0.6, message="Extracting song metadata")
    extractor = registry.metadata_extractor()
    embedded = dict(info.tags)
    if project["duration"]:
        embedded.setdefault("duration", project["duration"])
    metadata = extractor.extract(project["original_filename"], embedded)

    # Do not clobber values the user already edited.
    if project.get("metadata_source") in {"filename", "embedded", "llm"} or not project["title"]:
        if metadata.title:
            fields["title"] = metadata.title
        if metadata.artist:
            fields["artist"] = metadata.artist
        if metadata.album:
            fields["album"] = metadata.album
        fields["metadata_source"] = metadata.source
        fields["metadata_json"] = metadata.model_dump_json()

    fields.update(status="needs_review", error=None, progress=1)
    with db.connect() as conn:
        current = db.fetch_project(conn, project_id) or {}
        # The user may have pasted their own lyrics while this ran; theirs win.
        if (current.get("original_lyrics") or "").strip() and fields.get("original_lyrics"):
            for key in LYRIC_FIELDS:
                fields.pop(key, None)
        # Same for a title/artist they edited by hand: the extraction that was
        # already running must not overwrite it a second later.
        if current.get("metadata_source") == "manual":
            for key in ("title", "artist", "album", "metadata_source", "metadata_json"):
                fields.pop(key, None)
        db.update_project(conn, project_id, **fields)
    update_progress(task_id, stage="done", progress=1, message="已读取音频信息")


LYRIC_FIELDS = (
    "original_lyrics",
    "lyrics_source",
    "lyrics_lrclib_id",
    "lyrics_external_id",
    "lyrics_synced",
    "lyrics_word_sync",
    "lyrics_word_sync_format",
    "lyrics_word_sync_source",
    "lyrics_word_sync_edition",
    "timeline_json",
)


def _search_fields(project: dict[str, Any]) -> dict[str, Any]:
    """Song title/artist to search with, extracted by the LLM if still missing.

    The upload handler already extracts metadata before a run looks anything up,
    so this only matters when a project reached the align step without a title
    (an old project, a failed extraction). Searching on the raw filename would
    query the sources with the uploader's channel name, which finds the wrong
    song.
    """
    fields = {"title": project.get("title") or "", "artist": project.get("artist") or ""}
    if fields["title"] or not project.get("original_filename"):
        return fields
    try:
        metadata = registry.metadata_extractor().extract(
            project["original_filename"],
            {"duration": project.get("duration")} if project.get("duration") else {},
        )
    except Exception as exc:  # noqa: BLE001 - the lookup is best-effort
        logger.warning("metadata extraction failed before the lyrics lookup: %s", exc)
        return fields
    return {
        "title": metadata.title or "",
        "artist": metadata.artist or "",
        "album": metadata.album or "",
        "metadata_source": metadata.source,
        "metadata_json": metadata.model_dump_json(),
    }


def _fetch_lyrics_fields(
    project: dict[str, Any], *, task_id: str | None = None, trace: "Trace | None" = None
) -> dict[str, Any]:
    """Look the lyrics up in the source chain unless the user already supplied some.

    Only the *text* is taken from the API for alignment; the source's word-level
    (逐字) payload is stored alongside so the align handler can check it against
    this audio before deciding to use it.

    Title and artist come first — by LLM where they are still unknown — because
    every source is queried by name, and a wrong name finds a wrong song.

    ``fetch_lyrics`` on the project (default on) is the switch: off means the run
    never asks the sources and aligns the ASR transcript (or pasted lyrics)
    instead. Only the align handler calls this — an upload never does.
    """
    if (project.get("original_lyrics") or "").strip():
        return {}
    if not settings.auto_search or not bool(project.get("fetch_lyrics", True)):
        return {}
    search_fields = _search_fields(project)
    metadata_fields = {
        key: search_fields[key]
        for key in ("title", "artist", "album", "metadata_source", "metadata_json")
        if search_fields.get(key)
    }
    if search_fields["title"] != (project.get("title") or ""):
        # Nothing was known about the song before the LLM read the filename:
        # keep what it found, the debug page and the manual search need it too.
        project = {**project, **search_fields}
        if trace is not None:
            trace.step("metadata", {**search_fields, "note": "搜索歌词前先用 LLM 提取歌名和歌手"})
    match = lyrics_fetch.auto_fetch(project)
    if match is None:
        logger.info("no lyrics for %r in any source", project.get("title"))
        if trace is not None:
            trace.step("lyrics_search", {"query": project.get("title"), "result": "none"})
        if task_id:
            update_progress(
                task_id, stage="lyrics", message="网易云 / LRCLIB / LrcApi 都没有找到匹配的歌词，请手动粘贴"
            )
        return metadata_fields
    timeline = build_timeline_from_text(match.lyrics)
    if task_id:
        update_progress(
            task_id,
            stage="lyrics",
            message=f"歌词来自 {match.source_label}：{match.candidate.track_name} — {match.candidate.artist_name}",
        )
    if trace is not None:
        trace.step(
            "lyrics_search",
            {
                "query": {"title": project.get("title"), "artist": project.get("artist")},
                "source": match.source,
                "source_label": match.source_label,
                "external_id": match.external_id,
                "lrclib_id": match.lrclib_id,
                "score": round(match.score, 3),
                "track_name": match.candidate.track_name,
                "artist_name": match.candidate.artist_name,
                "synced": bool(match.synced),
                "word_sync": match.has_word_sync,
                "line_count": len(match.lyrics.splitlines()),
            },
        )
    fields = {
        "original_lyrics": match.lyrics,
        "lyrics_source": match.source,
        "lyrics_lrclib_id": match.lrclib_id,
        "lyrics_external_id": match.external_id,
        "lyrics_synced": match.synced or None,
        "lyrics_word_sync": match.candidate.word_sync_raw or None,
        "lyrics_word_sync_format": match.candidate.word_sync_format,
        "lyrics_word_sync_source": match.source if match.candidate.word_sync_raw else None,
        "lyrics_word_sync_edition": None,
        "timeline_json": timeline.model_dump_json(),
    }
    fields.update(metadata_fields)
    return fields


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


def align_handler(task_id: str, task: dict[str, Any]) -> None:
    """Align one project. Two flows, and the default one takes its times from the ASR.

    * **ASR timings** (default). Qwen3-ASR hears the audio
      in 20 s windows, the kana DP maps every lyric line onto the tokens it
      heard, and the LLM verifies that mapping. A line's position therefore
      comes from what was sung in *this* recording, and each character's time
      comes from the recogniser's own per-character tokens (``char_align``
      re-slices them per line, fills the gaps and spreads stacked characters).

    A line nothing could place stays untimed and is
    marked, and a line the recogniser could not time says so in the run notes
    rather than passing the word stream's split spans off as an alignment.
    """
    project_id = task["project_id"]
    with db.connect() as conn:
        project = db.fetch_project(conn, project_id)
    if project is None:
        raise RuntimeError("Project not found.")

    trace = Trace(project_id, task_id)
    trace.step(
        "metadata",
        {
            "title": project.get("title"),
            "artist": project.get("artist"),
            "album": project.get("album"),
            "duration": project.get("duration"),
            "source": project.get("metadata_source"),
            "original_filename": project.get("original_filename"),
        },
    )

    lines = _lines_for_alignment(project)
    if not (project.get("original_lyrics") or "").strip():
        # No real lyrics yet. Two cases:
        # - fetch_lyrics is on (default): ask the sources now. This is the only
        #   place the lookup happens — uploading a file never goes looking for a
        #   song, so the run starts with whatever the user pasted. Lyrics the
        #   user pasted are in ``original_lyrics`` and never reach this branch.
        # - fetch_lyrics is off: never ask the sources. The text stays whatever
        #   was pasted (or empty → the ASR transcript below).
        if bool(project.get("fetch_lyrics", True)):
            update_progress(
                task_id, stage="lyrics", progress=0.05, message="从 网易云 / LRCLIB / LrcApi 获取原始歌词"
            )
            fields = _fetch_lyrics_fields(project, task_id=task_id, trace=trace)
            if fields:
                with db.connect() as conn:
                    db.update_project(conn, project_id, **fields)
                project = {**project, **fields}
                lines = _lines_for_alignment(project)
        elif not (project.get("revised_lyrics") or "").strip():
            update_progress(
                task_id, stage="lyrics", progress=0.05, message="未获取歌词：直接以 ASR 识别结果按停顿断句"
            )
    trace.step(
        "lyrics_selected",
        {
            "source": project.get("lyrics_source"),
            "external_id": project.get("lyrics_external_id"),
            "line_count": len(lines),
            "lines": [line.text for line in lines],
            "has_reference_timeline": bool((project.get("lyrics_synced") or "").strip()),
            "word_sync": word_sync.describe(
                word_sync.from_stored(
                    project.get("lyrics_word_sync"), project.get("lyrics_word_sync_format")
                )
            ),
        },
    )
    if not lines:
        logger.info("no lyrics for %s; aligning the ASR transcript instead", project_id)

    # Which language is being sung? The LLM reads the metadata and lyrics and
    # names it; with no usable answer the ASR auto-detects. The aligner needs
    # the same answer, so this happens before anything is aligned.
    language, origin = language_detect(project, lyrics=project.get("original_lyrics") or "")
    with db.connect() as conn:
        db.update_project(conn, project_id, lyrics_language=language, language_origin=origin)
    project = {**project, "lyrics_language": language, "language_origin": origin}
    logger.info("language for %s: %s (%s)", project_id, language or "auto", origin)
    trace.step(
        "language",
        {"language": language, "origin": origin, "asr_language": language or "(自动识别)"},
    )

    audio_path = settings.audio_dir / project["audio_path"]
    work_dir = settings.data_dir / "work" / project_id
    work_dir.mkdir(parents=True, exist_ok=True)

    run_notes: list[str] = []
    if origin == "llm" and language:
        run_notes.append(f"识别语言：{language}（由 LLM 根据歌曲信息判断）")
    separate = project.get("separate_vocals")
    separate = True if separate is None else bool(separate)

    common = {
        "task_id": task_id,
        "project": project,
        "project_id": project_id,
        "lines": lines,
        "language": language,
        "audio_path": audio_path,
        "work_dir": work_dir,
        "separate": separate,
        "run_notes": run_notes,
        "trace": trace,
    }
    trace.step("align_flow", {"flow": "asr", "note": "识别定位每一行，逐字时间来自识别器自带的逐字 token"})
    timeline = _align_from_asr(**common)

    # The per-character times are settled. The last step hands the finished
    # timeline to the LLM once, to say where the words end and the next one
    # begins; the units it names keep the span of the characters they cover.
    # Times are never asked of it, and a line whose answer does not tile the
    # text keeps its per-character times.
    wants_merge = bool(project.get("llm_merge"))
    if timeline.lines and llm_merge.available(wants_merge):
        update_progress(task_id, stage="merge", progress=0.985, message="LLM 正在按语意合并逐字结果")
        try:
            merge_notes = llm_merge.apply(
                timeline,
                language=language,
                trace=trace,
                enabled=wants_merge,
                on_progress=lambda fraction, message: update_progress(
                    task_id, stage="merge", progress=min(0.995, 0.985 + 0.01 * fraction), message=message
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a merge must not sink a timed run
            logger.warning("semantic merge failed: %s", exc)
            merge_notes = [f"按语意合并逐字结果失败（{exc}），逐字结果仍按每个字给出时间"]
        run_notes.extend(merge_notes)
    elif timeline.lines:
        trace.step(
            "llm_merge",
            {"skipped": True, "reason": llm_merge.unavailable_detail(wants_merge)},
        )

    timeline.status = compute_timeline_status(timeline)
    timeline.updated_at = db.now_iso()
    if run_notes:
        timeline.notes = run_notes + list(timeline.notes)

    timed = sum(1 for line in timeline.lines if line.start is not None)
    with db.connect() as conn:
        db.update_project(
            conn,
            project_id,
            timeline_json=timeline.model_dump_json(),
            revised_lyrics="\n".join(line.text for line in timeline.lines),
            status="needs_review",
            error=None,
            progress=1,
        )
    update_progress(
        task_id,
        stage="done",
        progress=1,
        message=f"完成：{timed}/{len(timeline.lines)} 行已对齐",
    )
    # Drop the oldest traces once the run is complete; the debug page only
    # needs the recent ones and the ASR payload is the biggest thing we store.
    trace.finish()
    # The work dir holds the prepared vocals and every per-line slice of this
    # run; a re-run regenerates them, and the sizes add up to about one song
    # each. The audio itself and the timeline live elsewhere.
    _drop_work_dir(work_dir)


def _drop_work_dir(work_dir: Path) -> None:
    """Drop the big slices of a finished run, keeping the small caches.

    The WAVs (the prepared 16 kHz vocals plus one slice per ASR window, per
    line and per search window) are about one song's worth of audio and are
    regenerated on the next run. The JSON caches next to them are kilobytes
    and are what makes a re-run cheaper, so they stay.
    """
    if settings.keep_work_dirs or not work_dir.exists():
        return
    for path in work_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".wav":
            try:
                path.unlink()
            except OSError:
                continue


def _uses_hfa_chars(project: dict[str, Any]) -> bool:
    """Whether the optional per-character fusion runs; off is the default.

    The recogniser's own per-character tokens are the source of record. When
    this is on, ``char_align`` additionally asks the recogniser's aligner for a
    second per-character table and keeps the ASR's boundary wherever the two
    agree (``char_align.FUSE_TOLERANCE``) — a small accuracy gain for one extra
    recognition pass per run.
    """
    value = project.get("hfa_chars")
    return False if value is None else bool(value)


def _align_from_asr(
    *,
    task_id: str,
    project: dict[str, Any],
    project_id: str,
    lines: list[Line],
    language: str | None,
    audio_path: Path,
    work_dir: Path,
    separate: bool,
    run_notes: list[str],
    trace: Trace,
) -> Timeline:
    """Default flow: the recogniser places each line and times each character.

    Every line position comes from the ASR word stream (or, when the lyrics
    source's own 逐字 timings survive the ``word_sync`` gate, from those);
    every per-character time inside a line comes from the recogniser's own
    per-character tokens (``char_align`` re-slices them per line, fills gaps and
    spreads stacked characters). Lines the ASR did not hear stay untimed and are
    marked.
    """
    hfa_chars = _uses_hfa_chars(project)
    aligner = registry.aligner()
    stream = word_stream.for_project(language, asr_align=project.get("asr_align"))
    if not stream.available():
        raise ProviderUnavailable(stream.unavailable_detail())
    asr_provider = stream
    run_notes.extend(stream.notes)

    words = _cached_words(project)
    cached = bool(words)
    if words and not _cache_matches_audio(project, language, separate):
        # The tokens were recognised under another language (a Japanese song
        # heard as Chinese comes back as wrong Han characters) or from the other
        # audio (separated vocals vs raw mix). Reuse would freeze that mistake.
        logger.info(
            "cached ASR for %s was recognised as %s, now %s; transcribing again",
            project_id,
            project.get("asr_language"),
            language or "(自动识别)",
        )
        if _cache_matches_language(project, language):
            run_notes.append(
                f"这次{'先分离人声再识别' if separate else '直接识别原始混音'}，"
                "和上次缓存的识别结果对应的音频不同，已重新识别"
            )
        else:
            run_notes.append(
                f"识别语言改为 {language or '自动识别'}，上次缓存的识别结果"
                f"（{project.get('asr_language') or '自动识别'}）不再适用，已重新识别"
            )
        words = []
    if words:
        update_progress(task_id, stage="asr", progress=0.45, message="复用已缓存的 ASR 结果")
        trace.step(
            "asr_windows",
            {
                "cached": True,
                "note": "复用上一次识别结果，没有重新跑 ASR",
                "separate": separate,
                "windows": [],
            },
        )
    else:
        transcribed: dict[str, Any] = {}
        words = _transcribe(
            task_id,
            project,
            audio_path,
            work_dir,
            asr_provider,
            # "" = auto-detect; never fall back to a guessed default here.
            language=language or "",
            separate=separate,
            notes=run_notes,
            trace=trace,
            meta=transcribed,
        )
        # Whatever the stream decided while it ran (a fallback mid-run).
        for note in stream.notes:
            if note not in run_notes:
                run_notes.append(note)
        with db.connect() as conn:
            db.update_project(
                conn,
                project_id,
                asr_json=json.dumps(
                    {"words": [w.model_dump() for w in words]}, ensure_ascii=False
                ),
                # What the cache was recognised under, so a later language or
                # separation change can tell whether it is still usable.
                asr_language=language or "",
                asr_separated=int(bool(transcribed.get("separated", separate))),
            )
    # Also applied to cached results: they may predate the zero-width fix.
    words = repair_word_times(words)
    trace.step(
        "asr_words",
        {
            "count": len(words),
            "cached": cached,
            "used_for": "定位每一行；逐字时间来自识别器自带的逐字 token",
            "words": [
                {"text": word.text, "start": word.start, "end": word.end, "source": word.source}
                for word in words
            ],
        },
    )

    if not lines:
        # No lyrics anywhere: either fetch_lyrics is off (the user chose not to
        # look them up) or no source has this song (an original or unreleased
        # track) and nothing was pasted. The transcript is then the only honest
        # text — same fallback the one-shot API uses.
        update_progress(task_id, stage="align", progress=0.8, message="按识别停顿断句")
        update_progress(task_id, stage="align", progress=0.8, message="按语义断句")
        max_chars = _max_line_chars(project)
        groups, split_used = (
            line_split.split_lines(
                words,
                language=language,
                gap=LINE_GAP_SEC,
                max_chars=max_chars,
            )
            if _splits_lines(project)
            else (None, False)
        )
        if not split_used:
            groups = group_words(words)
        no_lyrics_note = (
            "未获取歌词：本次不自动查找，文本来自 ASR 识别结果，"
            + (
                f"由 LLM 按语义断句（{line_split.describe(language, max_chars)}）"
                if split_used
                else "按停顿断句"
            )
            if not bool(project.get("fetch_lyrics", True))
            else "没有歌词：网易云 / LRCLIB / LrcApi 都没有找到匹配项，"
            "文本来自 ASR 识别结果，"
            + (
                f"由 LLM 按语义断句（{line_split.describe(language, max_chars)}）"
                if split_used
                else "按停顿断句"
            )
            + "（这条路径没有原始歌词可对齐）"
        )
        timeline = Timeline(
            status="none",
            lines=lines_from_groups(groups),
            updated_at=db.now_iso(),
            notes=[no_lyrics_note],
        )
        trace.step(
            "line_split",
            {
                "used": split_used,
                "language": language or "（自动）",
                "limit": line_split.max_chars_for(language, override=max_chars),
                "lines": len(groups),
                "note": (
                    "LLM 只决定在哪里断行，不改动任何字符；断行结果与识别文本不一致时退回按停顿断句"
                    if split_used
                    else "没有可用的 LLM 断行结果，按识别停顿断句"
                ),
            },
        )
        # The transcript still needs per-character times: the recogniser's own
        # tokens are word-sized, so ``この`` would be one highlight block for
        # two characters. Nothing here has to be trusted from the ASR's
        # placement either — the lines are just where it heard a pause — so the
        # same forced aligner that times a lyric line re-times these.
        if timeline.lines and char_align.per_character_available(hfa_chars):
            update_progress(task_id, stage="align", progress=0.85, message="整理逐字时间")
            try:
                char_notes = char_align.retime(
                    timeline,
                    audio_path=prepared_audio(work_dir, audio_path),
                    work_dir=work_dir,
                    language=language,
                    trace=trace,
                    on_progress=lambda fraction, message: update_progress(
                        task_id, stage="align", progress=min(0.98, 0.9 + 0.08 * fraction), message=message
                    ),
                    use_hfa=hfa_chars,
                )
            except char_align.CharAlignUnavailable as exc:
                logger.warning("character alignment unavailable: %s", exc)
                char_notes = [f"{exc}；逐字时间只能由识别词流按行摊开估计"]
            timeline.notes.extend(char_notes)
        return timeline

    update_progress(task_id, stage="align", progress=0.55, message="LLM 正在对照原始歌词定位每一行")

    def on_progress(progress: float, message: str) -> None:
        # Mapping runs 0.55 -> 0.85, verification 0.85 -> 0.97.
        span = 0.30 if "校验" not in message else 0.12
        base = 0.55 if "校验" not in message else 0.85
        update_progress(task_id, progress=min(0.97, base + span * progress), message=message)

    reference = _reference_for(project, lines)
    if reference:
        trace.step(
            "reference",
            [
                {"index": index, "text": ref.text, "start": ref.start, "end": ref.end}
                for index, ref in sorted(reference.items())
            ],
        )

    # Word-level (逐字) lyrics from the source, if it had any. They describe some
    # recording of the song, so they are only used when the ASR heard the same
    # lines at roughly the same places; otherwise the local path runs as usual.
    sync_lines = word_sync.from_stored(
        project.get("lyrics_word_sync"), project.get("lyrics_word_sync_format")
    )
    sync_source = project.get("lyrics_word_sync_source") or project.get("lyrics_source") or ""
    edition = project.get("lyrics_word_sync_edition")
    if not sync_lines and _searches_other_editions(project) and settings.auto_search:
        # The entry the lyrics came from may have no 逐字 data at all while
        # another edition of the same song does (the studio cut usually has
        # none; the live one often does).
        update_progress(task_id, stage="align", progress=0.5, message="查找同一首歌其他版本的逐字歌词")
        found = _find_word_sync_edition(project, lines)
        if found is not None:
            sync_lines = found.word_sync_lines()
            sync_source = found.source
            edition = found.edition_label
            # Remember it: the next run reuses the payload instead of searching
            # again, and the debug page can say which edition it came from.
            with db.connect() as conn:
                db.update_project(
                    conn,
                    project_id,
                    lyrics_word_sync=found.candidate.word_sync_raw,
                    lyrics_word_sync_format=found.candidate.word_sync_format,
                    lyrics_word_sync_source=found.source,
                    lyrics_word_sync_edition=edition,
                )
            project = {
                **project,
                "lyrics_word_sync": found.candidate.word_sync_raw,
                "lyrics_word_sync_format": found.candidate.word_sync_format,
                "lyrics_word_sync_source": found.source,
                "lyrics_word_sync_edition": edition,
            }
            trace.step(
                "word_sync_search",
                {
                    "found": True,
                    "source": found.source,
                    "source_label": found.source_label,
                    "edition": edition,
                    "external_id": found.external_id,
                    "agreement": round(found.agreement, 3),
                    "line_count": len(found.lyrics.splitlines()),
                    "note": "本项目的歌词条目没有逐字歌词，改为查找同一首歌的其他版本（如现场版）",
                },
            )
        else:
            trace.step(
                "word_sync_search",
                {"found": False, "note": "同一首歌的其他版本也没有逐字歌词"},
            )
    elif not sync_lines and settings.auto_search:
        # Switched off for this project: say so, so the debug page does not
        # look like the search ran and came back empty.
        trace.step(
            "word_sync_search",
            {
                "found": False,
                "skipped": True,
                "note": "已关闭「查找其他版本的逐字歌词」，只使用本项目歌词条目自带的逐字歌词",
            },
        )
    sync = word_sync.apply(lines, sync_lines, words, source=sync_source, edition=edition)
    trace.step(
        "word_sync",
        {**sync.stats, "used": sync.used, "reason": sync.reason, "edition": edition},
    )
    if sync.used and sync.timeline is not None:
        logger.info("using %s word-level timings for %s", sync_source, project_id)
        update_progress(
            task_id, stage="align", progress=0.95, message="歌词源自带逐字歌词，已按这段音频对齐"
        )
        timeline = sync.timeline
        if sync.extras:
            # The source lyrics are the studio cut's; a live recording also has
            # a spoken intro, an MC section, ad-libs. Those are real content
            # the lyrics cannot contain, so they go through the same keep/drop
            # and re-lining judgement as on the local path.
            timeline.lines = _attach_extras(
                aligner, sync, lines, words, timeline, trace,
                detect=bool(project.get("detect_extras")),
            )
            timeline.status = compute_timeline_status(timeline)
    else:
        if not aligner.available():
            # Checked here, after the word-sync decision: a run the source
            # fully timed does not need the LLM aligner at all.
            raise ProviderUnavailable(aligner.unavailable_detail())
        asr_wav = prepared_audio(work_dir, audio_path)

        def store_words(updated: list[Word]) -> None:
            # Keep the better timings: the next run reuses the cache, and
            # re-recognising the same windows again would only waste ASR jobs.
            with db.connect() as conn:
                db.update_project(
                    conn,
                    project_id,
                    asr_json=json.dumps(
                        {"words": [w.model_dump() for w in updated]}, ensure_ascii=False
                    ),
                )

        async def repair(line_texts: list[Line], words: list[Word]) -> list[Word]:
            """Re-recognise the stretches where the ASR lost part of a phrase."""

            async def transcribe_window(lo: float, hi: float) -> list[Word]:
                update_progress(
                    task_id,
                    stage="asr",
                    progress=0.55,
                    message=f"识别时间戳间隔异常，重新识别整句 {lo:.1f}-{hi:.1f}s",
                )
                return await retranscribe_span(asr_wav, work_dir, asr_provider, lo, hi, language)

            repaired, log = await repair_gaps(words, line_texts, transcribe_window, trace=trace)
            note = repair_note(log)
            if note:
                run_notes.append(note)
            if repaired != words:
                store_words(repaired)
            return repaired

        async def resolve_extra(
            suspects: list[tuple[Line, int, str]], words: list[Word]
        ) -> list[Word] | None:
            """Re-hear segments that look like a lyric line the ASR mangled."""

            async def transcribe_window(lo: float, hi: float) -> list[Word]:
                update_progress(
                    task_id,
                    stage="asr",
                    progress=0.6,
                    message=f"疑似识别错误，重新识别 {lo:.1f}-{hi:.1f}s",
                )
                return await retranscribe_span(asr_wav, work_dir, asr_provider, lo, hi, language)

            resolved = await resolve_extra_segments(
                suspects, words, transcribe_window, trace=trace
            )
            if resolved is None:
                run_notes.append(
                    "LLM 认为部分「新句子」其实是识别错误，重新识别后仍未能对上，"
                    + ("保留为加唱行" if detect_extras else "这些内容不加入时间轴")
                )
                return None
            store_words(resolved)
            run_notes.append("部分疑似新加的内容其实是 ASR 识别错误，已重新识别并按原始歌词对齐")
            return resolved

        timeline = asyncio.run(
            aligner.align(
                audio_path,
                lines,
                asr_words=words,
                language=project.get("lyrics_language") or None,
                on_progress=on_progress,
                reference=reference,
                trace=trace,
                repair=repair,
                resolve_extra=resolve_extra,
                # 「识别现场加唱」: off by default — only the lyric lines come
                # back, and a segment the LLM names as a mangled lyric line is
                # still re-heard so that line gets placed.
                detect_extras=bool(project.get("detect_extras")),
            )
        )
    # The line positions are settled; now let the recogniser's per-character
    # tokens give every character its own time (``char_align`` re-slices them
    # per line, fills gaps and spreads stacked characters). The ASR decides
    # *where* each line is sung; its own tokens decide *when inside it* each
    # character is.
    if timeline.lines and char_align.per_character_available(hfa_chars):
        try:
            char_notes = char_align.retime(
                timeline,
                audio_path=prepared_audio(work_dir, audio_path),
                work_dir=work_dir,
                language=language,
                trace=trace,
                on_progress=lambda fraction, message: update_progress(
                    task_id, stage="align", progress=min(0.98, 0.9 + 0.08 * fraction), message=message
                ),
                use_hfa=hfa_chars,
            )
        except char_align.CharAlignUnavailable as exc:
            logger.warning("character alignment unavailable: %s", exc)
            char_notes = [f"{exc}；逐字时间只能由识别词流按行摊开估计"]
        if char_notes:
            run_notes.extend(char_notes)
    elif timeline.lines:
        # Neither source can place the characters: the per-character times in
        # the timeline are the recogniser's own token split, which is an
        # estimate and must say so rather than pass for a measurement.
        run_notes.append(
            "逐字时间来源不可用（识别服务未配置）：行位置来自识别，"
            "行内每个字的时间只能按识别词流摊开估计"
        )
    return timeline


def _attach_extras(
    aligner,
    sync,
    lines: list[Line],
    words: list[Word],
    timeline: Timeline,
    trace,
    detect: bool = True,
) -> list[Line]:
    """Add the sung-but-unwritten content to a word-synced timeline.

    The 逐字 path times the original lyrics and stops there; on a live recording
    that would silently drop the spoken intro, the MC section and the ad-libs.
    Those are judged and re-lined exactly as on the local path — kept unless
    they are short noise, never because the lyrics do not contain them. With
    the project's 「识别现场加唱」 switch off nothing is added, and the debug
    page records what was ignored.
    """
    from .providers.llm_aligner import is_substantial, merge_extras

    extras = list(sync.extras)
    lyrics_only = not detect
    if not hasattr(aligner, "filter_extras"):
        # No LLM aligner configured: the same rule as an unavailable model —
        # keep what is too long to be noise, drop the rest.
        kept = [] if lyrics_only else [extra for extra in extras if is_substantial(extra)]
        suspects: list[Any] = []
    else:
        try:
            kept, suspects = aligner.filter_extras(
                extras, lines, sync.report, trace=trace, words=words, lyrics_only=lyrics_only
            )
            kept = (
                aligner.split_extras(kept, words, sync.report, lines, trace=trace)
                if kept and not lyrics_only
                else []
            )
        except Exception as exc:  # noqa: BLE001 - extras must not fail a timed run
            logger.warning("could not judge the extras of a word-synced run: %s", exc)
            kept = [] if lyrics_only else extras
    if trace is not None:
        trace.step(
            "extras",
            {
                "before": [extra.text for extra in extras],
                "after": [line.text for line in kept],
                "suspects": len(suspects),
                "note": (
                    "未开启「识别现场加唱」：这些原始歌词里没有的内容不加入时间轴"
                    if lyrics_only
                    else "这些是原始歌词里没有、但确实唱了的内容（逐字歌词路径同样保留）"
                ),
            },
        )
    if not kept:
        return timeline.lines
    return merge_extras(list(timeline.lines), kept)


def _reference_for(project: dict[str, Any], lines: list[Line]) -> dict[int, Line]:
    """LRCLIB's own timeline, keyed by lyric line index — a hint, never output.

    The studio LRC says roughly when each line is sung; a live recording may
    differ, so the aligner only uses it to recover lines the monotonic DP could
    not place. Timings in the result still come from the ASR alone.
    """
    reference = reference_lines(project.get("lyrics_synced"))
    if not reference:
        return {}
    return match_reference([line.text for line in lines], reference)


def _searches_other_editions(project: dict[str, Any]) -> bool:
    """Whether this project allows looking for 逐字 lyrics in another edition."""
    value = project.get("word_sync_search")
    return False if value is None else bool(value)


def _splits_lines(project: dict[str, Any]) -> bool:
    """Whether a run with no lyrics asks the LLM where the lines break."""
    value = project.get("line_split")
    return True if value is None else bool(value)


def _max_line_chars(project: dict[str, Any]) -> int | None:
    """This project's own characters-per-line ceiling, or ``None`` for per language."""
    value = project.get("line_split_max_chars")
    return None if value is None else int(value)


def _find_word_sync_edition(project: dict[str, Any], lines: list[Line]):
    """Another edition of this song that has 逐字 lyrics, if one exists.

    Blocking wrapper around :func:`lyrics_fetch.find_word_sync`, like the other
    lookups in this module; a dead source must not fail the run.
    """
    try:
        return asyncio.run(
            lyrics_fetch.find_word_sync(
                project, lines, exclude=project.get("lyrics_external_id")
            )
        )
    except Exception as exc:  # noqa: BLE001 - a search failure is simply no data
        logger.warning("word-level lyrics search failed for %r: %s", project.get("title"), exc)
        return None


def _vocal_wav(
    audio_path: Path, work_dir: Path, *, separate: bool, notes: list[str]
) -> Path:
    """The 16 kHz mono vocal track the aligner reads.

    The ASR run already prepared it (``asr-input.wav``); when the recogniser was
    unavailable or failed, it is prepared here — separation included, because
    the aligner needs the vocals, not the mix (upstream measured a p90 line
    error of 0.14 s on the stem against 10.16 s on the mix).
    """
    prepared = work_dir / "asr-input.wav"
    if prepared.exists():
        return prepared
    source = audio_path
    if separate:
        separator = registry.separator()
        if separator.available():
            try:
                stems = asyncio.run(separator.separate(audio_path, work_dir))
                source = stems.get("vocals", audio_path)
            except Exception as exc:  # noqa: BLE001 - fall back to the raw mix
                logger.warning("vocal separation failed, aligning the mix: %s", exc)
                notes.append(f"人声分离失败，已改用原始音频对齐，逐字时间可能不准：{exc}")
    return to_asr_wav(source, prepared)

def _transcribe(
    task_id: str,
    project: dict[str, Any],
    audio_path: Path,
    work_dir: Path,
    asr_provider,
    *,
    language: str | None = None,
    separate: bool = True,
    notes: list[str] | None = None,
    trace: "Trace | None" = None,
    meta: dict[str, Any] | None = None,
) -> list[Word]:
    def on_progress(progress: float, message: str) -> None:
        update_progress(task_id, stage="asr", progress=progress, message=message)

    return transcribe_audio(
        audio_path,
        work_dir,
        language=language,
        asr_provider=asr_provider,
        on_progress=on_progress,
        separate=separate,
        notes=notes,
        trace=trace,
        meta=meta,
    )


def transcribe_audio(
    audio_path: Path,
    work_dir: Path,
    *,
    language: str | None,
    asr_provider,
    on_progress: Callable[[float, str], None] | None = None,
    separate: bool = True,
    notes: list[str] | None = None,
    trace: "Trace | None" = None,
    meta: dict[str, Any] | None = None,
) -> list[Word]:
    """Separate the vocals (when asked) and run the windowed ASR; returns timed tokens.

    Shared by the project flow and the one-shot ``/api/align`` endpoint, which
    has no task row to report progress into. ``notes`` collects run-level
    caveats (e.g. the separation fell back to the raw mix) for the caller, and
    ``meta`` receives ``separated``: whether the tokens describe the separated
    vocals or the raw mix, which is what makes the cache reusable or not.
    """
    report = on_progress or (lambda progress, message: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_audio = audio_path
    separator = registry.separator()
    if separate and separator.available():
        report(0.1, "分离人声（提升识别准确率）")
        try:
            stems = asyncio.run(separator.separate(audio_path, work_dir))
            source_audio = stems.get("vocals", audio_path)
        except Exception as exc:  # noqa: BLE001 - fall back to the raw mix
            logger.warning("vocal separation failed, using the original mix: %s", exc)
            report(0.2, f"人声分离失败，改用原始音频：{exc}")
            if notes is not None:
                notes.append(f"人声分离失败，已改用原始音频识别，准确率可能下降：{exc}")
            source_audio = audio_path

    report(0.3, "语音识别（定位每一行）")
    asr_wav = to_asr_wav(source_audio, work_dir / "asr-input.wav")

    # Short windows transcribe far more accurately than one long file: the
    # recogniser drifts over minutes of singing, and every drifted phrase drags
    # the whole lyric mapping with it. A 20 s window measured 47/54 lines
    # within 2 s against 20/54 for the full song.
    duration = probe_audio(asr_wav).duration or 0.0
    windows = _window_ranges(duration)
    words: list[Word] = []
    failed = 0
    window_log: list[dict[str, Any]] = []
    for number, (start, length) in enumerate(windows, 1):
        slice_path = work_dir / f"asr-{int(start):05d}.wav"
        try:
            cut_audio(asr_wav, slice_path, start, length)
            result: ASRResult = asyncio.run(asr_provider.transcribe(slice_path, language))
        except Exception as exc:  # noqa: BLE001 - one bad window must not sink the run
            failed += 1
            logger.warning("ASR failed for window %.1fs+%.1fs: %s", start, length, exc)
            window_log.append({"window": number, "start": start, "length": length, "error": str(exc)})
            report(0.3 + 0.15 * number / len(windows), f"第 {number}/{len(windows)} 段识别失败，已跳过")
            continue
        kept = 0
        for word in result.words:
            if not word.text.strip():
                continue
            if word.start is None or word.end is None:
                continue
            words.append(
                Word(
                    text=word.text,
                    start=round(word.start + start, 3),
                    end=round(word.end + start, 3),
                    source="asr",
                )
            )
            kept += 1
        window_log.append(
            {
                "window": number,
                "start": start,
                "length": length,
                "tokens": kept,
                "language": result.language,
                "model": result.model,
                "text": "".join(word.text for word in result.words),
            }
        )
        report(
            0.3 + 0.15 * number / len(windows),
            f"识别第 {number}/{len(windows)} 段（累计 {len(words)} 个时间戳）",
        )

    if failed == len(windows):
        raise RuntimeError("所有音频分段的 ASR 都失败了，请检查 ASR 服务。")

    if trace is not None:
        trace.step(
            "asr_windows",
            {
                "language": language or "(自动识别)",
                "recogniser": getattr(asr_provider, "recogniser", None)
                or getattr(asr_provider, "name", type(asr_provider).__name__),
                "separated": source_audio != audio_path,
                "duration": duration,
                "window_count": len(windows),
                "failed": failed,
                "windows": window_log,
            },
        )

    if meta is not None:
        meta["separated"] = source_audio != audio_path

    words.sort(key=lambda word: (word.start, word.end))
    words = _drop_duplicates(words)
    words = repair_word_times(words)
    if not words:
        raise RuntimeError("识别没有返回可用的时间戳，无法定位每一行。")
    logger.info("ASR returned %d timed tokens over %d windows", len(words), len(windows))
    report(0.45, f"识别到 {len(words)} 个时间戳")
    return words


ASR_WINDOW_SEC = 20.0


def _window_ranges(duration: float) -> list[tuple[float, float]]:
    """Fixed-length, non-overlapping windows covering the audio."""
    if duration <= 0:
        return [(0.0, ASR_WINDOW_SEC)]
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        ranges.append((start, min(ASR_WINDOW_SEC, duration - start)))
        start += ASR_WINDOW_SEC
    return ranges


def _drop_duplicates(words: list[Word]) -> list[Word]:
    """Drop repeated tokens at window boundaries (same text, overlapping time)."""
    out: list[Word] = []
    for word in words:
        if out:
            previous = out[-1]
            same = previous.text == word.text
            overlapping = (word.start or 0) <= (previous.end or 0) + 0.05
            if same and overlapping:
                previous.end = max(previous.end or 0, word.end or 0)
                continue
        out.append(word)
    return out


def _cache_matches_audio(
    project: dict[str, Any], language: str | None, separate: bool
) -> bool:
    """Whether the cached tokens describe the audio this run will recognise.

    Two things make the cache unusable: the language (the same audio heard as
    Chinese comes back as different, wrong characters) and which audio was
    recognised at all — the separated vocals or the raw mix. A cache from
    before either column existed carries no record and is assumed usable.
    """
    if not _cache_matches_language(project, language):
        return False
    cached = project.get("asr_separated")
    if cached is None:
        return True
    return bool(cached) == bool(separate)


def _cache_matches_language(project: dict[str, Any], language: str | None) -> bool:
    """Whether the cached ASR tokens were recognised in the language we want now.

    A transcription is language-specific: the same audio recognised as Chinese
    comes back as different (wrong) characters than as Japanese, so the cache
    cannot be reused across a language change. A cache from before this column
    existed has no language recorded and is assumed usable.
    """
    cached = project.get("asr_language")
    if cached is None:
        return True
    return (cached or None) == (language or None)


def _cached_words(project: dict[str, Any]) -> list[Word]:
    raw = project.get("asr_json")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        return [Word(**item) for item in payload.get("words", [])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _lines_for_alignment(project: dict[str, Any]) -> list[Line]:
    """The original lyrics are the source of truth; LRC timestamps are stripped."""
    text = project.get("original_lyrics") or project.get("revised_lyrics") or ""
    if not text.strip():
        return []
    if "[" in text:
        parsed = parse_lrc(text)
        if parsed.lines:
            return [Line(text=line.text) for line in parsed.lines if line.text.strip()]
    return [Line(text=line.text) for line in parse_plain_lyrics(text)]


def build_timeline_from_text(text: str) -> Timeline:
    """Turn free-form lyrics text into an untimed timeline (never invents times)."""
    lines = parse_plain_lyrics(text)
    timeline = Timeline(status="none", lines=lines, updated_at=db.now_iso())
    timeline.status = compute_timeline_status(timeline)
    return timeline
