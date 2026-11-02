"""One-shot alignment API: post an audio file, get word timestamps back.

    POST /api/align                 (multipart: file, lyrics?, language?, separate?)
    GET  /api/align                 (the cached jobs, newest first)
    GET  /api/align/{job_id}        (poll; returns the full result when done)
    GET  /api/align/{job_id}/audio  (the source file, for replaying a cached result)

The job runs in the background because a four-minute song takes minutes of GPU
time. Poll the job until ``status`` is ``succeeded`` or ``failed``.

The newest ``LYRICTRACE_ONESHOT_CACHE_JOBS`` (5) finished jobs are kept with
their source *and* their result, so a result can be listened to again in the web
UI without re-running anything; ``services/housekeeping`` sweeps the rest.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import db
from ..audio import guess_mime, is_supported_audio
from ..config import settings
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
        description="Look for 逐字 lyrics in other editions of the song (default: server setting)",
    ),
    detect_extras: bool = Form(
        default=False,
        description="Keep sung content the lyrics do not contain as extra lines (现场加唱)",
    ),
    llm_merge: bool = Form(
        default=False,
        description="Group the per-character timeline into words with the LLM (按语意合并成词)",
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
        )
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return _public(job)


@router.get("/align")
def list_alignment_jobs() -> dict[str, Any]:
    """The cached jobs — what the web UI lists, newest first.

    Only the newest ``oneshot_cache_jobs`` finished jobs survive the sweep, so
    every entry here has both its source file and its result on disk.
    """
    items = [_public(job) for job in jobs.recent()]
    return {"items": items, "total": len(items), "max_jobs": settings.oneshot_cache_jobs}


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


def _public(job: dict[str, Any]) -> dict[str, Any]:
    """Job record minus the absolute paths that mean nothing to a caller."""
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
    }
