"""One-shot alignment API: post an audio file, get word timestamps back.

    POST /api/align                 (multipart: file, lyrics?, language?, separate?)
    GET  /api/align                 (the cached jobs, newest first)
    GET  /api/align/{job_id}        (poll; returns the full result when done)
    GET  /api/align/{job_id}/audio  (the source file, for replaying a cached result)

The job runs in the background because a four-minute song takes minutes of GPU
time. Poll the job until ``status`` is ``succeeded`` or ``failed``.

Job records are history: they are never swept, so every request can be polled
again and the web UI lists them all. ``services/housekeeping`` drops only what
nobody reads twice — the ``work/`` slices, and the uploaded source once the job
is older than ``LYRICTRACE_ONESHOT_MAX_AGE_HOURS`` (the result stays).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from .. import db
from ..audio import guess_mime, is_supported_audio, remux_audio
from ..config import settings
from ..schemas import JobDelete, OneShotResult
from ..services.oneshot import TERMINAL, jobs

router = APIRouter(prefix="/api", tags=["align"])


@router.post("/align", status_code=202)
async def create_alignment_job(
    file: UploadFile = File(...),
    lyrics: str | None = Form(default=None, description="Optional original lyrics, one line each"),
    language: str | None = Form(default=None),
    separate: bool = Form(default=True, description="Run vocal separation before ASR"),
    word_sync_search: bool | None = Form(
        default=None,
        description="Look for 逐字 lyrics in other editions of the song (default: off)",
    ),
    detect_extras: bool = Form(
        default=False,
        description="Keep sung content the lyrics do not contain as extra lines (现场加唱)",
    ),
    llm_merge: bool = Form(
        default=False,
        description="Group the per-character timeline into words with the LLM (按语意合并成词)",
    ),
    fetch_lyrics: bool = Form(
        default=True,
        description="Look the song up in NetEase / LRCLIB / LrcApi when no lyrics were pasted (默认开；关掉则直接 ASR 或用手动粘贴的歌词，不自动去抓)",
    ),
    line_split: bool = Form(
        default=True,
        description="When no lyrics are given, ask the LLM where the sung lines break (默认开；只决定断行，不改动任何字符，对不上就退回按停顿断句)",
    ),
    line_split_max_chars: int | None = Form(
        default=None,
        description="Ceiling on the characters per line when there are no lyrics; empty means per language (中文 12、日语 18、其余 24)",
    ),
) -> dict[str, Any]:
    filename = Path(file.filename or "audio").name
    if not is_supported_audio(filename):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type '{Path(filename).suffix or 'unknown'}'.",
        )

    settings.ensure_dirs()
    upload_dir = settings.data_dir / "oneshot" / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    staging = upload_dir / f"{db.new_id('up_')}{Path(filename).suffix.lower()}"
    size = 0
    try:
        with staging.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
                    )
                handle.write(chunk)
    except HTTPException:
        staging.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        staging.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Same rewrite as the project upload: the job's audio is replayed from here
    # (``GET /api/align/{job_id}/audio``), and a browser refuses a container with
    # a malformed PICTURE block no matter how well ffmpeg reads it. The source is
    # left as uploaded if the rewrite fails — the ASR can still read that.
    remux_audio(staging)

    try:
        job = jobs.create(
            audio_path=staging,
            lyrics=lyrics,
            # No ``settings.asr_language`` fallback: that hint is the ASR
            # provider's default for the project flow, not a decision about
            # *this* song. Inheriting it made every call without ``language``
            # inherit the deployment's hint (Japanese here), so Chinese songs
            # were recognised as Japanese. Omitted means auto: the LLM names
            # the language from the file name's metadata and the lyrics, and
            # the ASR auto-detects when nothing can be determined.
            language=language or None,
            separate=separate,
            filename=filename,
            word_sync_search=word_sync_search,
            detect_extras=detect_extras,
            llm_merge=llm_merge,
            fetch_lyrics=fetch_lyrics,
            line_split=line_split,
            line_split_max_chars=line_split_max_chars,
        )
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return _public(job)


@router.get("/align")
def list_alignment_jobs(
    limit: int = Query(default=50, ge=1, le=500, description="How many jobs to return"),
    offset: int = Query(default=0, ge=0, description="Skip this many of the newest jobs"),
) -> dict[str, Any]:
    """A page of the history, newest first — what the web UI's cache page reads.

    Job records are never swept (only their ``work/`` slices and, past
    ``oneshot_max_age_hours``, their uploaded source), so an entry here always
    has its result; ``has_audio`` says whether the source is still playable.
    ``total`` counts everything, so the UI can page through all of it.
    """
    total = len(jobs.recent())
    page = jobs.recent(limit=limit, offset=offset)
    items = [_public(job, with_audio=jobs.input_path(job["job_id"]) is not None) for job in page]
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        # No longer a deletion rule: how long a finished job keeps its uploaded
        # source, and the count/size limits the UI quotes as reference.
        "max_age_hours": settings.oneshot_max_age_hours,
        "max_jobs": settings.oneshot_max_jobs,
    }


@router.get("/align/{job_id}")
def get_alignment_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public(job)


@router.get("/align/{job_id}/audio")
def get_alignment_audio(job_id: str) -> FileResponse:
    """The uploaded source, so a cached result can be played back in the UI."""
    if jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = jobs.input_path(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    job = jobs.get(job_id) or {}
    return FileResponse(
        path,
        media_type=guess_mime(path.name),
        filename=job.get("filename") or path.name,
        content_disposition_type="inline",
    )


@router.delete("/align/{job_id}", status_code=204)
def delete_alignment_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in TERMINAL:
        raise HTTPException(status_code=409, detail="Job is still running")
    shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)


@router.post("/align/delete", status_code=204)
def delete_alignment_jobs(payload: JobDelete) -> None:
    """Delete several cached jobs at once, with their audio and results.

    The whole job directory goes — that is the record (``job.json``) as well as
    the uploaded source, which is what "删除所有相关文件" means here. A job that
    is still queued or running is refused (409): its worker is writing into that
    directory right now.
    """
    ids = list(dict.fromkeys(payload.ids))
    if not ids:
        return
    running: list[str] = []
    for job_id in ids:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        if job["status"] not in TERMINAL:
            running.append(job_id)
    if running:
        raise HTTPException(status_code=409, detail="Job is still running")
    for job_id in ids:
        shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)


@router.put("/align/{job_id}/result")
def update_alignment_result(job_id: str, payload: OneShotResult) -> dict[str, Any]:
    """Write a hand-edited result back into a cached job.

    The web UI's cache page edits a cached result with the same editor the
    project page uses (per-character times and all), so the correction has to
    survive a reload. Only ``result`` is replaced: what the request asked for
    stays as it was.
    """
    updated = jobs.update_result(job_id, payload.model_dump())
    if updated is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _public(updated)


def _public(job: dict[str, Any], *, with_audio: bool = True) -> dict[str, Any]:
    """Job record minus the absolute paths that mean nothing to a caller.

    ``with_audio`` says whether the uploaded source is still on disk: the sweep
    drops it once a job is older than ``oneshot_max_age_hours``, and a job
    without it still lists and still polls — only playback is gone.
    """
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job.get("stage", ""),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "with_lyrics": job.get("with_lyrics", False),
        # Jobs from before this field existed only have the stored name.
        "filename": jobs.source_name(job),
        "language": job.get("language"),
        "result": job.get("result"),
        "with_audio": with_audio,
        # What this request asked of the line splitter; jobs from before these
        # fields existed simply look like the defaults.
        "line_split": bool(job.get("line_split", True)),
        "line_split_max_chars": job.get("line_split_max_chars"),
        # Set when a correction was written back through PUT /result.
        "edited_at": job.get("edited_at"),
    }
