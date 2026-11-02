"""Bounded caches: what may grow, how big, and when it is swept.

Every long-running part of this app leaves files behind: a one-shot job keeps
its upload and its 20-second ASR slices (≈100 MB for a four-minute song), a
project keeps its work dir (the prepared 16 kHz vocals, the per-line slices,
the second-backend cache), and an aborted upload leaves a staging file. None of
that is needed once the answer has been read — but nothing deleted it either,
so the disk grew by roughly one song per run.

The sweep is deliberately conservative about *what* is a cache:

- the uploaded audio in ``data/audio`` belongs to a project and is only removed
  when the project row is gone (the same rule ``DELETE /api/projects`` uses);
- a one-shot job is only removed when it is ``succeeded``/``failed`` and either
  older than ``oneshot_max_age_hours``, beyond ``oneshot_cache_jobs``, or the
  directory is over ``oneshot_max_mb`` — oldest first;
- a project work dir is skipped while that project has a queued or running
  task, because the run is writing into it;
- anything created in the last ``ORPHAN_GRACE`` is left alone: an upload is
  written to disk before its row is inserted, and a project's work dir is
  created after the sweep read the project table, so "no row points at it" is
  only evidence once the file has had time to be claimed;
- ``job.json`` / the SQLite rows are never touched, so a job that is still
  inside its limits can always be polled again.

``run()`` is idempotent and safe to call at any time; ``start()`` runs it at
startup and then every ``housekeeping_interval_min``.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .. import db
from ..config import settings

logger = logging.getLogger("lyrictrace.housekeeping")

#: A job dir with no readable record is only an orphan after this long: the
#: record is written immediately after the directory, so a minute of grace is
#: enough and a crash between the two does not leave it forever.
ORPHAN_GRACE = timedelta(minutes=10)
#: A project whose task is running writes into its work dir; never sweep it.
ACTIVE_TASK_STATUSES = ("queued", "running")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _age(path: Path) -> timedelta:
    try:
        return datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        return timedelta(0)


def _finished_at(job: dict[str, Any], path: Path) -> datetime:
    raw = job.get("finished_at") or job.get("created_at")
    if raw:
        try:
            value = datetime.fromisoformat(str(raw))
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


@dataclass
class Sweep:
    """What one pass removed, for the log line and the API."""

    jobs: list[str] = field(default_factory=list)
    work_dirs: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    uploads: int = 0
    freed_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "jobs": self.jobs,
            "work_dirs": self.work_dirs,
            "audio": self.audio,
            "uploads": self.uploads,
            "freed_mb": round(self.freed_bytes / (1024 * 1024), 1),
        }

    def __bool__(self) -> bool:
        return bool(self.jobs or self.work_dirs or self.audio or self.uploads)


# --------------------------------------------------------------------------
# One-shot job directories
# --------------------------------------------------------------------------


def _read_job(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads((path / "job.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sweep_oneshot(root: Path, sweep: Sweep) -> None:
    """Drop finished jobs past the age, count and size limits; oldest first."""
    if not root.exists():
        return
    finished: list[tuple[datetime, Path, int]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        job = _read_job(path)
        if job is None:
            if _age(path) > ORPHAN_GRACE:
                _remove(path, sweep)
            continue
        if job.get("status") not in ("succeeded", "failed"):
            # Running or queued: its worker is writing into this directory.
            continue
        finished.append((_finished_at(job, path), _size(path), path))

    finished.sort(key=lambda item: item[0], reverse=True)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.oneshot_max_age_hours)
    keep: list[tuple[datetime, int, Path]] = []
    for index, item in enumerate(finished):
        when = item[0]
        # The count rule is the cache size the UI shows: the newest
        # ``oneshot_cache_jobs`` requests keep their source *and* their result,
        # so the web UI always has those five playable. The age rule still
        # applies on top of it.
        if when < cutoff or index >= settings.oneshot_cache_jobs:
            _remove(item[2], sweep)
        else:
            keep.append(item)

    # A job that stays inside its limits keeps ``input`` + ``job.json`` — that
    # is the result the caller polls. Its ``work/`` (the ASR slices, ~100 MB a
    # song) is what the job itself drops when it ends; jobs from before that
    # rule still carry one, so the sweep drops it too.
    if not settings.oneshot_keep_work:
        for _when, _size_, path in keep:
            work = path / "work"
            if not work.is_dir():
                continue
            work_size = _size(work)
            shutil.rmtree(work, ignore_errors=True)
            if not work.exists():
                sweep.freed_bytes += work_size
    keep = [(when, _size(path), path) for when, _size_, path in keep]

    # The count rule keeps the newest; the size rule drops the oldest, so the
    # result is "as many recent jobs as fit" rather than "whatever came last".
    total = sum(size for _when, size, _path in keep)
    for _when, size, path in sorted(keep, key=lambda item: item[0]):
        if total <= settings.oneshot_max_mb * 1024 * 1024:
            break
        _remove(path, sweep)
        total -= size


def _remove(path: Path, sweep: Sweep) -> None:
    size = _size(path)
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        sweep.freed_bytes += size
        sweep.jobs.append(path.name)


# --------------------------------------------------------------------------
# Project work directories, uploaded audio, upload staging
# --------------------------------------------------------------------------


def _live_projects(conn) -> tuple[set[str], set[str], set[str]]:
    """Project ids, the ids with a task queued or running, and their audio files."""
    ids = {row["id"] for row in conn.execute("SELECT id FROM projects")}
    placeholders = ", ".join("?" for _ in ACTIVE_TASK_STATUSES)
    busy = {
        row["project_id"]
        for row in conn.execute(
            f"SELECT DISTINCT project_id FROM tasks WHERE status IN ({placeholders})",
            ACTIVE_TASK_STATUSES,
        )
    }
    referenced = {
        row["audio_path"] for row in conn.execute("SELECT audio_path FROM projects")
    }
    return ids, busy, referenced


def sweep_work(root: Path, projects: set[str], busy: set[str], sweep: Sweep) -> None:
    """Work dirs of deleted projects, then the oldest ones over the size budget."""
    if not root.exists():
        return
    candidates: list[tuple[float, Path, int]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        size = _size(path)
        if path.name not in projects:
            # A project created after this sweep read its rows can already have
            # a work dir; its mtime is being refreshed by that run, so the grace
            # period tells the two apart.
            if _age(path) > ORPHAN_GRACE:
                _remove_work(path, size, sweep)
            continue
        if path.name in busy:
            # A run is writing into it; its size still counts toward the budget
            # so the sweep does not free space by deleting someone else's cache
            # and then leave this one over the limit.
            continue
        try:
            candidates.append((path.stat().st_mtime, path, size))
        except OSError:
            continue

    total = sum(item[2] for item in candidates)
    candidates.sort(key=lambda item: item[0])
    budget = settings.work_max_mb * 1024 * 1024
    for _mtime, path, size in candidates:
        if total <= budget:
            break
        _remove_work(path, size, sweep)
        total -= size


def _remove_work(path: Path, size: int, sweep: Sweep) -> None:
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        sweep.freed_bytes += size
        sweep.work_dirs.append(path.name)


def sweep_audio(root: Path, referenced: set[str], sweep: Sweep) -> None:
    """Audio files no project points at (a crash between upload and insert).

    A file is only a candidate after ``ORPHAN_GRACE``: the upload is written to
    disk *before* its row is inserted, so for a moment every new project's
    audio looks unreferenced — and the sweep runs on its own thread, so that
    moment can fall inside a sweep.
    """
    if not root.exists():
        return
    for path in root.iterdir():
        if not path.is_file() or path.name in referenced or _age(path) <= ORPHAN_GRACE:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        if not path.exists():
            sweep.freed_bytes += size
            sweep.audio.append(path.name)


def sweep_uploads(root: Path, sweep: Sweep) -> None:
    """Staging files of uploads that never became a job."""
    if not root.exists():
        return
    cutoff = timedelta(minutes=settings.upload_stale_minutes)
    for path in root.iterdir():
        if not path.is_file() or _age(path) <= cutoff:
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        if not path.exists():
            sweep.freed_bytes += size
            sweep.uploads += 1


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run() -> dict[str, Any]:
    """One sweep. Never raises: housekeeping must not fail a request or a run."""
    sweep = Sweep()
    # The job dirs do not need the database, so they are swept even when the
    # read below fails — a broken DB is exactly when the disk matters most.
    try:
        sweep_oneshot(settings.data_dir / "oneshot", sweep)
        sweep_uploads(settings.data_dir / "oneshot" / "_uploads", sweep)
    except Exception as exc:  # noqa: BLE001 - a failed sweep is not a failed run
        logger.warning("housekeeping (one-shot) failed: %s", exc)
    try:
        with db.connect() as conn:
            projects, busy, referenced = _live_projects(conn)
        sweep_work(settings.data_dir / "work", projects, busy, sweep)
        sweep_audio(settings.audio_dir, referenced, sweep)
    except Exception as exc:  # noqa: BLE001 - without the rows, do not guess
        logger.warning("housekeeping (projects) failed: %s", exc)
    if sweep:
        logger.info(
            "housekeeping freed %.1f MB (jobs %d, work dirs %d, audio %d, uploads %d)",
            sweep.freed_bytes / (1024 * 1024),
            len(sweep.jobs),
            len(sweep.work_dirs),
            len(sweep.audio),
            sweep.uploads,
        )
    return sweep.as_dict()


def usage() -> dict[str, Any]:
    """Current sizes and limits, for ``GET /api/storage``."""
    oneshot_root = settings.data_dir / "oneshot"
    jobs = 0
    if oneshot_root.exists():
        jobs = sum(
            1
            for path in oneshot_root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )
    work_root = settings.data_dir / "work"
    work_dirs = sum(1 for path in work_root.iterdir() if path.is_dir()) if work_root.exists() else 0
    return {
        "data_dir": str(settings.data_dir),
        "oneshot": {
            "jobs": jobs,
            "mb": round(_size(oneshot_root) / (1024 * 1024), 1),
            "max_jobs": settings.oneshot_cache_jobs,
            "max_mb": settings.oneshot_max_mb,
            "max_age_hours": settings.oneshot_max_age_hours,
        },
        "work": {
            "dirs": work_dirs,
            "mb": round(_size(work_root) / (1024 * 1024), 1),
            "max_mb": settings.work_max_mb,
        },
        "audio": {"mb": round(_size(settings.audio_dir) / (1024 * 1024), 1)},
        "database": {
            "mb": round(_size(settings.db_path) / (1024 * 1024), 1)
            if settings.db_path.exists()
            else 0.0
        },
        "interval_min": settings.housekeeping_interval_min,
    }


class Housekeeping:
    """The scheduled sweep: once at startup, then every interval."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if settings.housekeeping_interval_min <= 0:
            # 0 disables the timer; ``POST /api/storage/cleanup`` still works.
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="lyrictrace-housekeeping", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        run()
        interval = settings.housekeeping_interval_min * 60
        while not self._stop.wait(interval):
            run()


housekeeping = Housekeeping()
