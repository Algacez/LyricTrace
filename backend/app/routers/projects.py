"""Project, upload, lyrics, alignment, timeline and export endpoints.

The flow is deliberately short: upload audio, paste the original lyrics, run
separation + ASR + LLM alignment, then fix up the word timings by hand and
export.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response

from .. import db
from ..audio import guess_mime, is_supported_audio, probe_audio, remux_audio
from ..config import settings
from ..schemas import (
    AlignRequest,
    LyricsCandidateOut,
    LyricsSearchRequest,
    LyricsSearchResponse,
    LyricsSelectRequest,
    LyricsUpdate,
    ProjectDelete,
    ProjectList,
    ProjectOut,
    ProjectUpdate,
    TaskOut,
    Timeline,
    TimelineUpdate,
)
from ..services import lyrics_fetch
from ..services.furigana import strip_furigana_text
from ..services.lyrics import export_ass, export_json, export_lrc, parse_lrc, parse_plain_lyrics
from ..services.pipeline import build_timeline_from_text
from ..services.projects import compute_timeline_status, serialize_project, serialize_project_summary
from ..services.providers.base import ProviderUnavailable, registry
from ..tasks import TASK_ALIGN, TASK_PIPELINE, runner

router = APIRouter(prefix="/api", tags=["projects"])


def _get_project_or_404(conn, project_id: str) -> dict[str, Any]:
    project = db.fetch_project(conn, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _create_task(conn, project_id: str, kind: str) -> dict[str, Any]:
    task = db.create_task(conn, project_id, kind)
    conn.commit()
    runner.submit(task["id"])
    return task


def _separates(project: dict[str, Any]) -> bool:
    """The project's vocal-separation setting; absent means the default (on)."""
    value = project.get("separate_vocals")
    return True if value is None else bool(value)


def _asr_aligns(project: dict[str, Any]) -> bool:
    """The project's ASR-alignment setting; absent means the default (off)."""
    value = project.get("asr_align")
    return False if value is None else bool(value)


def _word_sync_searches(project: dict[str, Any]) -> bool:
    """The project's other-edition 逐字 search setting; absent means off."""
    value = project.get("word_sync_search")
    return False if value is None else bool(value)


def _searching_word_sync(updates: dict[str, Any], project: dict[str, Any]) -> bool:
    """Whether turning the search off must drop a payload from another edition.

    That payload only exists because the search ran, so it must not outlive the
    switch being turned off — keeping it would silently keep using the other
    edition, the opposite of what turning the search off asks for. A payload the
    lyrics' own entry supplied has no edition label and stays.
    """
    if updates.get("word_sync_search") is None or bool(updates.get("word_sync_search")):
        return False
    return bool((project.get("lyrics_word_sync_edition") or "").strip())


def _word_sync_fields(project: dict[str, Any], requested: bool | None) -> dict[str, Any]:
    """Column updates for a per-run ``word_sync_search`` override."""
    if requested is None:
        return {}
    updates = {"word_sync_search": int(bool(requested))}
    if _searching_word_sync(updates, project):
        updates.update(
            lyrics_word_sync=None,
            lyrics_word_sync_format=None,
            lyrics_word_sync_source=None,
            lyrics_word_sync_edition=None,
        )
    return updates


# --------------------------------------------------------------------------
# Upload & projects
# --------------------------------------------------------------------------


@router.post("/projects", response_model=ProjectOut, status_code=201)
async def create_project(file: UploadFile = File(...)) -> ProjectOut:
    """Upload an audio file; metadata is read in the background."""
    filename = Path(file.filename or "audio").name
    if not is_supported_audio(filename):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{Path(filename).suffix or 'unknown'}'. "
            "Supported: mp3, flac, wav, m4a, aac, ogg, opus, wma, aiff, webm.",
        )
    return await _store_upload(file, filename)


async def _store_upload(file: UploadFile, filename: str) -> ProjectOut:
    """Write the upload to disk, make it browser-playable, and queue its pipeline.

    The rewrite comes after the tags are read: ``remux_audio`` keeps the tags
    but drops the embedded cover art, and the pipeline reads the file's metadata
    again on disk — either order works for the tags, and only this one leaves a
    file that plays in the browser.
    """
    project_id = db.new_id("prj_")
    stored_name = f"{project_id}{Path(filename).suffix.lower()}"
    destination = settings.audio_dir / stored_name
    settings.ensure_dirs()

    max_bytes = settings.max_upload_bytes
    size = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Tags first, then the rewrite that keeps only the audio (see the docstring).
    info = probe_audio(destination)
    remux_audio(destination)
    size = destination.stat().st_size
    timestamp = db.now_iso()
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, title, artist, album, duration, audio_filename, audio_path, "
            "audio_size, audio_mime, original_filename, metadata_source, metadata_json, status, "
            "progress, separate_vocals, asr_align, hfa_timing, hfa_chars, detect_extras, llm_merge, "
            "word_sync_search, fetch_lyrics, line_split, created_at, updated_at) "
            "VALUES (?, '', '', '', "
            "?, ?, ?, ?, ?, ?, 'pending', '{}', 'queued', 0, 1, 0, 0, 0, 0, 0, ?, ?, ?, ?, ?)",
            (
                project_id,
                info.duration,
                filename,
                stored_name,
                size,
                info.mime or guess_mime(filename),
                filename,
                int(settings.word_sync_search),
                int(settings.fetch_lyrics),
                int(settings.line_split),
                timestamp,
                timestamp,
            ),
        )
        _create_task(conn, project_id, TASK_PIPELINE)
        project = db.fetch_project(conn, project_id)
    assert project is not None
    return serialize_project(project)


@router.get("/projects", response_model=ProjectList)
def list_projects(
    q: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ProjectList:
    clauses: list[str] = []
    params: list[Any] = []
    if q:
        clauses.append("(title LIKE ? OR artist LIKE ? OR original_filename LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) AS n FROM projects {where}", params).fetchone()["n"]
        rows = conn.execute(
            f"SELECT * FROM projects {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
    return ProjectList(items=[serialize_project_summary(db.row_to_dict(row)) for row in rows], total=total)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str) -> ProjectOut:
    with db.connect() as conn:
        return serialize_project(_get_project_or_404(conn, project_id))


@router.patch("/projects/{project_id}", response_model=ProjectOut)
def update_project(project_id: str, payload: ProjectUpdate) -> ProjectOut:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
        if any(key in updates for key in ("title", "artist", "album")):
            updates["metadata_source"] = updates.get("metadata_source") or "manual"
        if "separate_vocals" in updates and updates["separate_vocals"] is not None:
            # The cached tokens came from the other audio path (vocals vs mix);
            # they cannot be reused for a run that recognises different audio.
            if bool(updates["separate_vocals"]) != _separates(project):
                updates["asr_json"] = None
            updates["separate_vocals"] = int(bool(updates["separate_vocals"]))
        if "detect_extras" in updates and updates["detect_extras"] is not None:
            # Turning it on only adds a step; the cached transcript (if any) is
            # still about this audio, so it stays either way.
            updates["detect_extras"] = int(bool(updates["detect_extras"]))
        if "llm_merge" in updates and updates["llm_merge"] is not None:
            # Only decides whether the finished timeline is regrouped into
            # words; nothing is cached differently, so the ASR cache stays.
            updates["llm_merge"] = int(bool(updates["llm_merge"]))
        if "hfa_chars" in updates and updates["hfa_chars"] is not None:
            # Only decides where the per-character times come from; nothing
            # cached depends on it (the word stream describes this audio either
            # way), so the ASR cache stays.
            updates["hfa_chars"] = int(bool(updates["hfa_chars"]))
        if "hfa_timing" in updates and updates["hfa_timing"] is not None:
            # Only decides which flow places the lines; nothing cached depends
            # on it (the word stream describes this audio either way).
            updates["hfa_timing"] = int(bool(updates["hfa_timing"]))
        if "asr_align" in updates and updates["asr_align"] is not None:
            # Only decides which recogniser produces the line positions (and
            # whether its character times survive as a fallback); the cached
            # word stream still describes this audio, so it stays.
            updates["asr_align"] = int(bool(updates["asr_align"]))
        if "word_sync_search" in updates:
            # Switching it off drops a payload found in another edition; with
            # the search off nothing would look for a replacement.
            updates.update(_word_sync_fields(project, updates["word_sync_search"]))
        if "fetch_lyrics" in updates and updates["fetch_lyrics"] is not None:
            # Only decides whether the *next* run asks the sources when no lyrics
            # were pasted; nothing cached depends on it, so the ASR cache stays.
            updates["fetch_lyrics"] = int(bool(updates["fetch_lyrics"]))
        if "line_split" in updates and updates["line_split"] is not None:
            # Only decides how a run *without* lyrics groups the transcript; the
            # cached word stream is about this audio either way.
            updates["line_split"] = int(bool(updates["line_split"]))
        if updates.get("line_split_max_chars") is not None:
            updates["line_split_max_chars"] = int(updates["line_split_max_chars"])
        db.update_project(conn, project_id, **updates)
        return serialize_project(_get_project_or_404(conn, project_id))


def _drop_project_files(project: dict[str, Any]) -> None:
    """Everything a project owns on disk: its audio and its ``work/`` dir.

    The DB rows (tasks / candidates / debug steps) cascade away with the project
    row, since every one of them carries ``ON DELETE CASCADE`` and
    ``db.connect`` turns foreign keys on.
    """
    (settings.audio_dir / project["audio_path"]).unlink(missing_ok=True)
    work_dir = settings.data_dir / "work" / project["id"]
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


@router.post("/projects/delete", status_code=204)
def delete_projects(payload: ProjectDelete) -> Response:
    """Delete several projects at once, with everything they own.

    A batch is refused outright (409) if any of them is queued or processing:
    a worker is writing into that project's ``work/`` right now, and deleting
    under it would leave the run to fail on files that no longer exist.
    """
    ids = list(dict.fromkeys(payload.ids))
    if not ids:
        return Response(status_code=204)
    with db.connect() as conn:
        rows = []
        for project_id in ids:
            project = _get_project_or_404(conn, project_id)
            if project["status"] in {"queued", "processing"}:
                raise HTTPException(status_code=409, detail=f"项目 {project_id} 正在处理中，不能删除。")
            rows.append(project)
        for project in rows:
            conn.execute("DELETE FROM projects WHERE id = ?", (project["id"],))
    for project in rows:
        _drop_project_files(project)
    return Response(status_code=204)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> Response:
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    _drop_project_files(project)
    return Response(status_code=204)


@router.get("/projects/{project_id}/audio")
def get_audio(project_id: str) -> FileResponse:
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    path = settings.audio_dir / project["audio_path"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    return FileResponse(
        path,
        media_type=project["audio_mime"] or guess_mime(path.name),
        filename=project["original_filename"] or path.name,
        content_disposition_type="inline",
    )


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------


@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: str) -> TaskOut:
    with db.connect() as conn:
        task = db.fetch_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskOut(**task)


@router.get("/projects/{project_id}/tasks", response_model=list[TaskOut])
def list_project_tasks(
    project_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[TaskOut]:
    """This project's tasks, newest first — a page of them, newest first.

    A run is followed by polling the newest entry, so a caller that only needs
    that one asks for a small page rather than the project's whole history.
    """
    with db.connect() as conn:
        _get_project_or_404(conn, project_id)
        rows = conn.execute(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (project_id, limit, offset),
        ).fetchall()
    return [TaskOut(**dict(row)) for row in rows]


# --------------------------------------------------------------------------
# Lyrics
# --------------------------------------------------------------------------


@router.put("/projects/{project_id}/lyrics", response_model=ProjectOut)
def update_lyrics(project_id: str, payload: LyricsUpdate) -> ProjectOut:
    """Store the original lyrics; an untimed timeline is created for editing."""
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update.")
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
        fields: dict[str, Any] = {}
        if updates.get("original_lyrics") is not None:
            # Readings in brackets are annotations, not sung text: drop them
            # once, on the way in, so every consumer (aligner, LLM, export,
            # editor) sees the same line.
            fields["original_lyrics"] = strip_furigana_text(updates["original_lyrics"])
        if updates.get("revised_lyrics") is not None:
            fields["revised_lyrics"] = updates["revised_lyrics"]
        if updates.get("language") is not None:
            # "auto" / "自动识别" clears the choice so the LLM decides again.
            from ..services.language import is_auto, normalize

            raw = updates["language"]
            chosen = None if is_auto(raw) else (normalize(raw) or raw)
            fields["lyrics_language"] = chosen
            fields["language_origin"] = None if is_auto(raw) else "manual"
            if chosen != (project.get("lyrics_language") or None):
                # The cached tokens were recognised in the old language: a
                # Japanese song heard as Chinese keeps its wrong text forever
                # otherwise, however many times the run is repeated.
                fields["asr_json"] = None
        if updates.get("lyrics_source") is not None:
            fields["lyrics_source"] = updates["lyrics_source"]
        if "lyrics_lrclib_id" in updates:
            fields["lyrics_lrclib_id"] = updates["lyrics_lrclib_id"]

        if fields.get("original_lyrics") is not None:
            # Typed/pasted lyrics replace whatever the API supplied.
            if "lyrics_source" not in fields and fields["original_lyrics"] != (
                project.get("original_lyrics") or ""
            ):
                fields["lyrics_source"] = "manual"
                fields["lyrics_lrclib_id"] = None
                fields["lyrics_external_id"] = None
                fields["lyrics_synced"] = None
                fields["lyrics_word_sync"] = None
                fields["lyrics_word_sync_format"] = None
                fields["lyrics_word_sync_source"] = None
                fields["lyrics_word_sync_edition"] = None
            # A pasted LRC carries a timeline of its own: keep it as the
            # positional hint for the aligner (the text stays the source).
            pasted = fields["original_lyrics"]
            if "[" in pasted:
                parsed = parse_lrc(pasted)
                if any(line.start is not None for line in parsed.lines):
                    fields["lyrics_synced"] = pasted
            # New lyrics invalidate the previous alignment.
            fields["asr_json"] = None
            if not project.get("timeline"):
                lines = parse_plain_lyrics(fields["original_lyrics"])
                timeline = Timeline(status="none", lines=lines, updated_at=db.now_iso())
                timeline.status = compute_timeline_status(timeline)
                fields["timeline_json"] = timeline.model_dump_json()

        db.update_project(conn, project_id, **fields)
        return serialize_project(_get_project_or_404(conn, project_id))


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


@router.post("/projects/{project_id}/lyrics/search", response_model=LyricsSearchResponse)
async def search_lyrics(project_id: str, payload: LyricsSearchRequest | None = None) -> LyricsSearchResponse:
    """Ask every lyrics source (NetEase → LRCLIB → LrcApi) for the original lyrics."""
    payload = payload or LyricsSearchRequest()
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    try:
        matches = await lyrics_fetch.search_all(project, query=payload.query)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as 502
        raise HTTPException(status_code=502, detail=f"歌词源请求失败：{exc}") from exc
    return LyricsSearchResponse(
        source=matches[0].source if matches else "netease",
        sources=sorted({match.source for match in matches}, key=lyrics_fetch.SOURCE_ORDER.index),
        query={
            "title": project.get("title") or None,
            "artist": project.get("artist") or None,
            "query": payload.query,
        },
        candidates=[
            LyricsCandidateOut(
                lrclib_id=match.lrclib_id,
                external_id=match.external_id,
                source=match.source,
                track_name=match.candidate.track_name,
                artist_name=match.candidate.artist_name,
                album_name=match.candidate.album_name,
                duration=match.candidate.duration,
                score=match.score,
                line_count=len(match.lyrics.splitlines()),
                preview=match.preview(),
                synced=bool(match.candidate.synced_lyrics),
                word_sync=match.has_word_sync,
            )
            for match in matches
        ],
    )


@router.post("/projects/{project_id}/lyrics/select", response_model=ProjectOut)
async def select_lyrics(project_id: str, payload: LyricsSelectRequest) -> ProjectOut:
    """Use one search result as the original lyrics (text only, no timings)."""
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    source = payload.source or "lrclib"
    external_id = payload.external_id or (str(payload.lrclib_id) if payload.lrclib_id else None)
    if external_id is None:
        raise HTTPException(status_code=422, detail="需要 lrclib_id 或 external_id。")
    try:
        match = await lyrics_fetch.fetch_by_external(source, external_id, project)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as 502
        raise HTTPException(status_code=502, detail=f"歌词源请求失败：{exc}") from exc
    if match is None:
        raise HTTPException(status_code=404, detail="该歌词条目没有歌词文本。")

    timeline = build_timeline_from_text(match.lyrics)
    with db.connect() as conn:
        db.update_project(
            conn,
            project_id,
            original_lyrics=match.lyrics,
            lyrics_source=match.source,
            lyrics_lrclib_id=match.lrclib_id,
            lyrics_external_id=match.external_id,
            lyrics_synced=match.synced or None,
            lyrics_word_sync=match.candidate.word_sync_raw or None,
            lyrics_word_sync_format=match.candidate.word_sync_format,
            lyrics_word_sync_source=match.source if match.candidate.word_sync_raw else None,
            lyrics_word_sync_edition=None,
            asr_json=None,
            timeline_json=timeline.model_dump_json(),
            status="needs_review",
        )
        return serialize_project(_get_project_or_404(conn, project_id))


@router.post("/projects/{project_id}/align", response_model=TaskOut, status_code=202)
def request_alignment(project_id: str, payload: AlignRequest | None = None) -> TaskOut:
    """Queue: ASR -> LLM line mapping against the lyrics -> per-character times.

    Lyrics are not required: with none, the ASR transcript becomes the text
    (the recogniser's own words, split at pauses) — never invented lyrics.
    """
    payload = payload or AlignRequest()
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
        if payload.reset_cache:
            db.update_project(conn, project_id, asr_json=None)
        if payload.separate is not None:
            # Switching between vocals and the raw mix changes what the ASR
            # hears, so the cached tokens no longer describe this run's audio.
            if payload.separate != _separates(project):
                db.update_project(conn, project_id, asr_json=None)
            db.update_project(conn, project_id, separate_vocals=int(payload.separate))
        if payload.asr_align is not None:
            db.update_project(conn, project_id, asr_align=int(payload.asr_align))
        if payload.hfa_chars is not None:
            db.update_project(conn, project_id, hfa_chars=int(payload.hfa_chars))
        if payload.hfa_timing is not None:
            db.update_project(conn, project_id, hfa_timing=int(payload.hfa_timing))
        if payload.word_sync_search is not None:
            db.update_project(conn, project_id, **_word_sync_fields(project, payload.word_sync_search))
        if payload.fetch_lyrics is not None:
            db.update_project(conn, project_id, fetch_lyrics=int(payload.fetch_lyrics))
        if payload.line_split is not None:
            db.update_project(conn, project_id, line_split=int(payload.line_split))
        if payload.line_split_max_chars is not None:
            db.update_project(conn, project_id, line_split_max_chars=int(payload.line_split_max_chars))
        db.update_project(conn, project_id, status="queued", error=None, progress=0)
        task = _create_task(conn, project_id, TASK_ALIGN)
    return TaskOut(**task)


@router.get("/projects/{project_id}/asr")
def get_asr(project_id: str) -> dict[str, Any]:
    """Raw ASR tokens, for inspection while fixing up timings."""
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    raw = project.get("asr_json")
    if not raw:
        return {"words": []}
    return json.loads(raw)


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


@router.put("/projects/{project_id}/timeline", response_model=ProjectOut)
def update_timeline(project_id: str, payload: TimelineUpdate) -> ProjectOut:
    timeline = payload.timeline
    _validate_timeline(timeline)
    timeline.status = compute_timeline_status(timeline)
    timeline.updated_at = db.now_iso()
    with db.connect() as conn:
        _get_project_or_404(conn, project_id)
        db.update_project(
            conn,
            project_id,
            timeline_json=timeline.model_dump_json(),
            revised_lyrics="\n".join(line.text for line in timeline.lines),
            status="ready" if timeline.status == "complete" else "needs_review",
        )
        return serialize_project(_get_project_or_404(conn, project_id))


def _validate_timeline(timeline: Timeline) -> None:
    for index, line in enumerate(timeline.lines):
        if line.start is not None and line.start < 0:
            raise HTTPException(status_code=422, detail=f"Line {index + 1}: start must be >= 0.")
        if line.start is not None and line.end is not None and line.end < line.start:
            raise HTTPException(status_code=422, detail=f"Line {index + 1}: end is before start.")
        previous: float | None = None
        for word in line.words:
            if word.start is not None and word.end is not None and word.end < word.start:
                raise HTTPException(status_code=422, detail=f"Line {index + 1}: word end before start.")
            if word.start is not None and previous is not None and word.start < previous - 1e-6:
                raise HTTPException(
                    status_code=422, detail=f"Line {index + 1}: word timestamps are not ascending."
                )
            if word.start is not None:
                previous = word.start


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def _timeline_or_404(project: dict[str, Any]) -> Timeline:
    timeline = Timeline(**project["timeline"]) if project.get("timeline") else None
    if timeline is None:
        raise HTTPException(status_code=409, detail="Project has no timeline yet.")
    return timeline


@router.get("/projects/{project_id}/lyrics/export")
def export_original_lyrics(
    project_id: str,
    format: str = Query(default="lrc", pattern="^(lrc|txt)$"),
) -> Response:
    """Download the original lyrics, with the source's own timeline when it has one.

    ``lrc`` returns the source LRC verbatim (studio timings included) so it can
    be compared against the aligned result; when no source supplied a timeline
    the plain text is returned instead — nothing is ever invented here.
    """
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    reference = (project.get("lyrics_synced") or "").strip()
    lyrics = project.get("original_lyrics") or ""
    if not lyrics.strip() and not reference:
        raise HTTPException(status_code=409, detail="还没有原始歌词。")

    if format == "lrc":
        body = reference or lyrics
        suffix = "lrc"
    else:
        body = "\n".join(line.text for line in parse_lrc(lyrics).lines) if "[" in lyrics else lyrics
        suffix = "txt"
    stem = _safe_stem(f"{project['title'] or project['original_filename'] or project_id} 原始歌词")
    ascii_stem = _ascii_stem(stem)
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{ascii_stem}.{suffix}"; '
            f"filename*=UTF-8''{quote(f'{stem}.{suffix}')}"
        )
    }
    return PlainTextResponse(
        body if body.endswith("\n") else body + "\n",
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


@router.get("/projects/{project_id}/export")
def export_project(
    project_id: str,
    format: str = Query(default="json", pattern="^(json|lrc|ass)$"),
    karaoke: bool = Query(default=True, description="Include per-word \\k tags in ASS"),
) -> Response:
    with db.connect() as conn:
        project = _get_project_or_404(conn, project_id)
    timeline = _timeline_or_404(project)
    stem = _safe_stem(project["title"] or project["original_filename"] or project_id)
    # HTTP headers are latin-1; CJK titles go in the RFC 5987 filename* instead.
    ascii_stem = _ascii_stem(stem)
    headers = {
        "Content-Disposition": (
            f'attachment; filename="{ascii_stem}.{format}"; '
            f"filename*=UTF-8''{quote(f'{stem}.{format}')}"
        )
    }

    if format == "json":
        payload = export_json(
            timeline,
            project_id=project_id,
            title=project["title"],
            artist=project["artist"],
            album=project["album"],
        )
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
            headers=headers,
        )
    if format == "lrc":
        text = export_lrc(timeline, title=project["title"], artist=project["artist"], album=project["album"])
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8", headers=headers)
    text = export_ass(timeline, title=project["title"], artist=project["artist"], karaoke=karaoke)
    return PlainTextResponse(text, media_type="text/plain; charset=utf-8", headers=headers)


def _safe_stem(name: str) -> str:
    keep = "".join(ch for ch in name if ch.isalnum() or ch in " -_()[]").strip()
    return keep[:80] or "lyrics"


def _ascii_stem(name: str) -> str:
    """A latin-1-safe fallback name for clients that ignore ``filename*``."""
    ascii_only = "".join(ch for ch in name if ch.isascii() and ch.isalnum()).strip()
    return ascii_only[:60] or "lyrics"
