"""One-shot alignment jobs: audio in, word timestamps out.

The project flow keeps state in SQLite because a human edits the result. The
``/api/align`` endpoint has no project and no editor — it takes an audio file
(and optionally lyrics) and returns the timings. That work still takes minutes,
so it runs on a background thread and the caller polls the job.

Jobs live on disk under ``<data>/oneshot/<job_id>/`` so a poll from any worker
process (or a server restart) still finds the result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .. import db
from ..audio import probe_audio
from ..config import settings
from ..schemas import Line, Timeline, Word
from . import char_align, lyrics_fetch, word_stream, word_sync
from . import llm_merge as semantic_merge
from .asr_lines import lines_from_words
from .asr_repair import (
    prepared_audio,
    repair_gaps,
    repair_note,
    resolve_extra_segments,
    retranscribe_span,
)
from .lyrics import parse_lrc, parse_plain_lyrics
from .pipeline import transcribe_audio
from .providers.base import registry
from .reference import match_reference, reference_lines

logger = logging.getLogger("lyrictrace.oneshot")

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"

TERMINAL = {JOB_SUCCEEDED, JOB_FAILED}


class OneShotJobs:
    """A tiny disk-backed job table with a single background worker.

    One worker is deliberate: the ASR and the LLM are the slow, GPU-bound parts
    of this machine, and queueing beats oversubscribing them.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (settings.data_dir / "oneshot")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oneshot")
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    # -- paths ---------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def _lock(self, job_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(job_id, threading.Lock())

    # -- storage -------------------------------------------------------

    def create(
        self,
        *,
        audio_path: Path,
        lyrics: str | None,
        language: str | None,
        separate: bool,
        filename: str | None = None,
        word_sync_search: bool | None = None,
        detect_extras: bool = False,
        llm_merge: bool = False,
    ) -> dict[str, Any]:
        job_id = db.new_id("job_")
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        stored = directory / f"input{audio_path.suffix.lower() or '.wav'}"
        shutil.move(str(audio_path), stored)

        job: dict[str, Any] = {
            "job_id": job_id,
            "status": JOB_QUEUED,
            "stage": "queued",
            "progress": 0.0,
            "message": "排队中",
            "error": None,
            "created_at": db.now_iso(),
            "finished_at": None,
            "language": language,
            "with_lyrics": bool(lyrics and lyrics.strip()),
            "result": None,
            # What the caller sent. The cache list in the web UI names a job by
            # it, and the audio endpoint serves ``input.<ext>`` under it.
            "filename": filename or audio_path.name,
            "input": stored.name,
        }
        self._write(job)
        self._executor.submit(
            self._run,
            job_id,
            stored,
            lyrics or "",
            language,
            separate,
            filename or audio_path.name,
            settings.word_sync_search if word_sync_search is None else bool(word_sync_search),
            bool(detect_extras),
            bool(llm_merge),
        )
        return job

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._record_path(job_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def recent(self) -> list[dict[str, Any]]:
        """Every job on disk, newest first.

        This is the cache the web UI shows: the sweep keeps the newest
        ``oneshot_cache_jobs`` finished jobs (source + result), so whatever is
        here is what can still be replayed without re-running anything.
        """
        if not self.root.exists():
            return []
        found: list[dict[str, Any]] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            job = self.get(directory.name)
            if job is None:
                continue
            job.pop("_mtime", None)
            found.append(job)
        found.sort(
            key=lambda job: (str(job.get("created_at") or ""), job.get("job_id") or ""), reverse=True
        )
        return found

    def input_path(self, job_id: str) -> Path | None:
        """The audio this job was given, for the web UI's player."""
        job = self.get(job_id)
        if job is None:
            return None
        name = job.get("input")
        candidates = [self.job_dir(job_id) / name] if name else []
        candidates.extend(sorted(self.job_dir(job_id).glob("input.*")))
        for path in candidates:
            if path.is_file():
                return path
        return None

    def source_name(self, job: dict[str, Any]) -> str | None:
        """What the caller uploaded it as; jobs from before the field existed
        only have the stored ``input.<ext>`` name."""
        name = job.get("filename") or job.get("input")
        if name:
            return name
        path = self.input_path(job["job_id"])
        return path.name if path is not None else None

    def _write(self, job: dict[str, Any]) -> None:
        path = self._record_path(job["job_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock(job_id):
            job = self.get(job_id) or {"job_id": job_id}
            job.update(fields)
            self._write(job)
            return job

    # -- work ----------------------------------------------------------

    def _run(
        self,
        job_id: str,
        audio_path: Path,
        lyrics: str,
        language: str | None,
        separate: bool,
        filename: str,
        word_sync_search: bool,
        detect_extras: bool = False,
        llm_merge: bool = False,
    ) -> None:
        self._update(job_id, status=JOB_RUNNING, stage="start", progress=0.02, message="开始处理")

        def on_progress(progress: float, message: str) -> None:
            # Transcription owns 0.05-0.55, alignment the rest.
            self._update(
                job_id,
                stage="asr" if progress < 0.5 else "align",
                progress=round(0.05 + 0.5 * progress, 3),
                message=message,
            )

        try:
            notes: list[str] = []
            lyrics, synced, source_sync = self._resolve_lyrics(
                job_id,
                lyrics,
                filename,
                notes,
                word_sync_search,
                # The running time decides whether a same-titled entry by
                # another singer is this cover or a different recording.
                duration=probe_audio(audio_path).duration,
            )
            language = self._resolve_language(job_id, language, filename, lyrics, notes)
            # Persist what the run settled on: the request may have said
            # "auto", and the job record is what the cache list reads.
            self._update(job_id, language=language or None)
            work_dir = self.job_dir(job_id) / "work"
            # The recogniser says which line is sung where, and its own per-character
            # tokens give each character its time (see word_stream and char_align).
            stream = word_stream.for_project(language, asr_align=None)
            notes.extend(stream.notes)
            words = transcribe_audio(
                audio_path,
                work_dir,
                language=language,
                asr_provider=stream,
                on_progress=on_progress,
                separate=separate,
                notes=notes,
            )
            notes.extend(note for note in stream.notes if note not in notes)
            result = self._build_result(
                words,
                lyrics,
                language,
                on_progress,
                notes,
                synced=synced,
                source_sync=source_sync,
                repair=self._repairer(work_dir, audio_path, language, on_progress, notes, stream),
                resolve_extra=self._extra_resolver(
                    work_dir, audio_path, language, on_progress, notes, stream
                ),
                retime=self._retimer(work_dir, audio_path, language, on_progress, notes),
                detect_extras=detect_extras,
                llm_merge=llm_merge,
            )
            self._update(
                job_id,
                status=JOB_SUCCEEDED,
                stage="done",
                progress=1.0,
                message=f"完成：{len(result['lines'])} 行，{len(words)} 个时间戳",
                finished_at=db.now_iso(),
                result=result,
            )
            self._drop_work(job_id)
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            logger.exception("one-shot job %s failed", job_id)
            self._update(
                job_id,
                status=JOB_FAILED,
                stage="failed",
                message="失败",
                error=str(exc) or exc.__class__.__name__,
                finished_at=db.now_iso(),
            )
            self._drop_work(job_id)

    def _drop_work(self, job_id: str) -> None:
        """Delete the job's ``work/`` dir once it is over.

        The result is already in ``job.json`` and nothing reads those slices
        again; keeping them cost about one song's worth of WAV per call. The
        upload itself stays, so the job record still says what was aligned.
        """
        if settings.oneshot_keep_work:
            return
        shutil.rmtree(self.job_dir(job_id) / "work", ignore_errors=True)

    def _resolve_lyrics(
        self,
        job_id: str,
        lyrics: str,
        filename: str,
        notes: list[str],
        word_sync_search: bool = True,
        duration: float | None = None,
    ) -> tuple[str, str, list[Line]]:
        """Fetch the original lyrics from the source chain when the caller did not send any.

        Returns ``(lyrics, synced, word_sync_lines)``: the synced LRC is the
        API's own timeline, kept only as a positional hint for the aligner. The
        word-level lines are used only if they agree with this audio.
        """
        if lyrics.strip() or not settings.auto_search:
            return lyrics, "", []
        self._update(job_id, stage="lyrics", progress=0.04, message="从 网易云 / LRCLIB / LrcApi 获取原始歌词")
        match = lyrics_fetch.auto_fetch(_project_stub(filename, duration))
        if match is None:
            notes.append("没有提供歌词，网易云 / LRCLIB / LrcApi 也没找到匹配项：文本来自 ASR 识别结果")
            return lyrics, "", []
        notes.append(
            f"歌词来自 {match.source_label}："
            f"{match.candidate.track_name} — {match.candidate.artist_name}"
            f"（匹配度 {match.score:.2f}，时间由本地识别与强制对齐生成）"
        )
        self._update(
            job_id,
            stage="lyrics",
            message=f"歌词来自 {match.source_label}：{match.candidate.track_name}",
        )
        sync_lines = match.word_sync_lines()
        if not sync_lines and word_sync_search:
            # This entry has no 逐字 data, but another edition of the same song
            # may (the studio cut often has none, the live one often does).
            sync_lines = self._other_edition_word_sync(match, notes)
        return match.lyrics, match.synced, sync_lines

    def _other_edition_word_sync(self, match: lyrics_fetch.LyricsMatch, notes: list[str]) -> list[Line]:
        """Word-level timings from a different edition of the same song."""
        lines = [Line(text=line.text) for line in _lines_from_lyrics(match.lyrics)]
        if not lines:
            return []
        project = {
            "title": match.candidate.track_name or "",
            "artist": match.candidate.artist_name or "",
            "album": match.candidate.album_name or "",
            "duration": match.candidate.duration,
        }
        try:
            found = asyncio.run(
                lyrics_fetch.find_word_sync(project, lines, exclude=match.external_id)
            )
        except Exception as exc:  # noqa: BLE001 - a failed search is simply no data
            logger.info("other-edition word-sync search failed: %s", exc)
            return []
        if found is None:
            return []
        notes.append(
            f"本项目用的歌词条目没有逐字歌词，改用同一首歌的其他版本：{found.edition_label}"
            f"（{found.source_label}，歌词重合 {found.agreement:.0%}；仍要核对与这段音频是否吻合）"
        )
        return found.word_sync_lines()

    def _resolve_language(
        self,
        job_id: str,
        language: str | None,
        filename: str,
        lyrics: str,
        notes: list[str],
    ) -> str:
        """The caller's language wins; otherwise the LLM infers it from the song.

        Returns ``""`` when nothing can be determined, which makes the ASR
        auto-detect instead of being told a wrong language.
        """
        from .language import is_auto, normalize, resolve

        if not is_auto(language):
            resolved = normalize(language)
            return resolved or (language or "").strip()

        project = _project_stub(filename)
        detected, origin = resolve(project, lyrics=lyrics)
        if detected:
            if origin == "llm":
                notes.append(f"识别语言：{detected}（由 LLM 根据歌曲信息判断）")
            self._update(job_id, stage="language", message=f"识别语言：{detected}")
        else:
            notes.append("语言无法判断，ASR 使用自动识别")
            self._update(job_id, stage="language", message="语言无法判断，使用自动识别")
        return detected or ""

    def _repairer(
        self,
        work_dir: Path,
        audio_path: Path,
        language: str | None,
        on_progress: Callable[[float, str], None],
        notes: list[str],
        asr_provider=None,
    ):
        """Re-recognise implausible gaps in the timings, for ``aligner.align``."""
        asr_provider = asr_provider or registry.asr()

        async def repair(lines: list[Line], words: list[Word]) -> list[Word]:
            asr_wav = prepared_audio(work_dir, audio_path)

            async def transcribe_window(lo: float, hi: float) -> list[Word]:
                on_progress(0.9, f"识别时间戳间隔异常，重新识别整句 {lo:.1f}-{hi:.1f}s")
                return await retranscribe_span(asr_wav, work_dir, asr_provider, lo, hi, language)

            repaired, log = await repair_gaps(words, lines, transcribe_window)
            note = repair_note(log)
            if note:
                notes.append(note)
            return repaired

        return repair

    def _extra_resolver(
        self,
        work_dir: Path,
        audio_path: Path,
        language: str | None,
        on_progress: Callable[[float, str], None],
        notes: list[str],
        asr_provider=None,
    ):
        """Re-hear segments the LLM thinks are a mangled lyric line, not new content."""
        asr_provider = asr_provider or registry.asr()

        async def resolve_extra(suspects, words: list[Word]) -> list[Word] | None:
            asr_wav = prepared_audio(work_dir, audio_path)

            async def transcribe_window(lo: float, hi: float) -> list[Word]:
                on_progress(0.9, f"疑似识别错误，重新识别 {lo:.1f}-{hi:.1f}s")
                return await retranscribe_span(asr_wav, work_dir, asr_provider, lo, hi, language)

            resolved = await resolve_extra_segments(suspects, words, transcribe_window)
            if resolved is None:
                notes.append(
                    "LLM 认为部分「新句子」其实是识别错误，重新识别后仍未能对上，保留为加唱行"
                )
                return None
            notes.append("部分疑似新加的内容其实是 ASR 识别错误，已重新识别并按原始歌词对齐")
            return resolved

        return resolve_extra

    def _retimer(
        self,
        work_dir: Path,
        audio_path: Path,
        language: str | None,
        on_progress: Callable[[float, str], None],
        notes: list[str],
    ):
        """Re-time each line's characters with the forced aligner."""

        def retime(lines: list[Line]) -> list[str]:
            if not char_align.per_character_available():
                return []
            asr_wav = prepared_audio(work_dir, audio_path)
            timeline = Timeline(status="none", lines=lines)
            return char_align.retime(
                timeline,
                audio_path=asr_wav,
                work_dir=work_dir,
                language=language,
                on_progress=lambda fraction, message: on_progress(0.97, message),
            )

        return retime

    def _build_result(
        self,
        words: list[Word],
        lyrics: str,
        language: str | None,
        on_progress: Callable[[float, str], None],
        run_notes: list[str] | None = None,
        *,
        synced: str = "",
        source_sync: list[Line] | None = None,
        repair: Callable[[list[Line], list[Word]], Any] | None = None,
        resolve_extra: Callable[[list[tuple[Line, int, str]], list[Word]], Any] | None = None,
        retime: Callable[[list[Line]], list[str]] | None = None,
        detect_extras: bool = False,
        llm_merge: bool = False,
    ) -> dict[str, Any]:
        notes: list[str] = list(run_notes or [])
        lines: list[Line]
        if lyrics.strip():
            text_lines = _lines_from_lyrics(lyrics)
            # A source's word-level timings are used only when they agree with
            # this audio; otherwise everything below runs as usual.
            sync = word_sync.apply(text_lines, source_sync or [], words)
            if sync.used and sync.timeline is not None:
                on_progress(0.95, "歌词源自带逐字歌词，已按这段音频对齐")
                lines = sync.timeline.lines
                notes.extend(sync.timeline.notes)
            else:
                on_progress(0.62, "LLM 正在对照歌词定位每一行")
                aligner = registry.aligner()
                if not aligner.available():
                    raise RuntimeError(aligner.unavailable_detail())
                reference = match_reference([line.text for line in text_lines], reference_lines(synced))
                timeline = asyncio.run(
                    aligner.align(
                        Path(),  # the ASR already produced every timing
                        text_lines,
                        asr_words=words,
                        language=language,
                        on_progress=lambda fraction, message: on_progress(0.62 + 0.38 * fraction, message),
                        reference=reference,
                        repair=repair,
                        resolve_extra=resolve_extra,
                        detect_extras=detect_extras,
                    )
                )
                lines = timeline.lines
                notes.extend(timeline.notes)
        else:
            on_progress(0.8, "按识别停顿断句")
            lines = lines_from_words(words)
            notes.append("未提供歌词：文本来自 ASR 识别结果，按停顿断句")

        if retime is not None and lines:
            # Line positions are settled; the per-character pass gives every
            # character its own time (see char_align).
            on_progress(0.97, "整理逐字时间")
            try:
                notes.extend(retime(lines))
            except char_align.CharAlignUnavailable as exc:
                notes.append(f"{exc}；逐字时间只能由识别词流按行摊开估计")

        if lines and semantic_merge.available(llm_merge):
            # The last step of a run: the LLM says where the words end, and
            # each unit takes the span of the characters it covers. Times are
            # never asked of it (see llm_merge). Off unless the request asks
            # for it: the per-character times are what was measured.
            on_progress(0.99, "LLM 正在按语意合并逐字结果")
            try:
                notes.extend(
                    semantic_merge.apply(
                        Timeline(status="none", lines=lines), language=language, enabled=llm_merge
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a merge must not sink the run
                notes.append(f"按语意合并逐字结果失败（{exc}），逐字结果仍按每个字给出时间")

        return {
            "language": language,
            "time_unit": "seconds",
            "text_source": "lyrics" if lyrics.strip() else "asr",
            "notes": notes,
            "words": [
                {"text": word.text, "start": word.start, "end": word.end, "source": word.source}
                for word in words
            ],
            "lines": [
                {
                    "text": line.text,
                    "start": line.start,
                    "end": line.end,
                    "source": line.source,
                    "note": line.note,
                    "words": [
                        {"text": word.text, "start": word.start, "end": word.end, "source": word.source}
                        for word in line.words
                    ],
                }
                for line in lines
            ],
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def recover_stale(self) -> int:
        """Fail jobs left running by a previous process.

        The worker lives in this process, so a restart orphans whatever it was
        doing. Without this a client would poll a job that can never finish.
        """
        if not self.root.exists():
            return 0
        recovered = 0
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            job = self.get(directory.name)
            if not job or job.get("status") in TERMINAL:
                continue
            self._update(
                directory.name,
                status=JOB_FAILED,
                stage="failed",
                message="失败",
                error="服务重启，任务已中断，请重新提交。",
                finished_at=db.now_iso(),
            )
            recovered += 1
        if recovered:
            logger.warning("marked %d orphaned one-shot job(s) as failed", recovered)
        return recovered


def _project_stub(filename: str, duration: float | None = None) -> dict[str, Any]:
    """Just enough project fields for the lyrics lookup: title, artist, duration.

    The title and artist are extracted by the LLM (``LYRICTRACE_METADATA_EXTRACTOR``)
    before any source is searched, so the search uses real song metadata rather
    than a mangled file name.

    ``duration`` is the uploaded audio's running time. Without it the cover rule
    (``lyrics_fetch._same_recording_by_title``) can never fire — it needs the
    running time on both sides — so a cover uploaded here would never find the
    original's lyrics while the same file in a project would.
    """
    try:
        metadata = registry.metadata_extractor().extract(filename)
        return {
            "title": metadata.title,
            "artist": metadata.artist,
            "album": metadata.album,
            "duration": duration,
        }
    except Exception as exc:  # noqa: BLE001 - the lookup is best-effort
        logger.warning("metadata extraction failed for %r: %s", filename, exc)
        return {"title": "", "artist": "", "album": "", "duration": duration}


def _lines_from_lyrics(text: str) -> list[Line]:
    """Same rule as the project flow: LRC stamps are stripped, text is kept."""
    if "[" in text:
        parsed = parse_lrc(text)
        if parsed.lines:
            return [Line(text=line.text) for line in parsed.lines if line.text.strip()]
    return [Line(text=line.text) for line in parse_plain_lyrics(text)]


jobs = OneShotJobs()
