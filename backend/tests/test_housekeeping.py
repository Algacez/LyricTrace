"""Bounded caches: the sweep deletes what is a cache and nothing else.

Every rule here protects something: a running job's directory, a project's
work dir while its task runs, the uploaded audio of a live project, and the
result a caller may still poll. The limits are the other half — a cache that is
never swept is how the disk filled up in the first place.

One-shot jobs are the exception: their *record* is history and is never swept.
Only the derived files inside the directory go — ``work/`` always, and the
uploaded source once the job is older than ``oneshot_max_age_hours``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app import db
from app.config import settings
from app.services import housekeeping


def _job(root: Path, name: str, status: str = "succeeded", *, age_hours: float = 0.0, size: int = 1024) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    finished = time.time() - age_hours * 3600
    (directory / "job.json").write_text(
        json.dumps(
            {
                "job_id": name,
                "status": status,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(finished)),
            }
        ),
        encoding="utf-8",
    )
    (directory / "input.mp3").write_bytes(b"x" * size)
    os.utime(directory / "job.json", (finished, finished))
    return directory


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    oneshot = tmp_path / "oneshot"
    work = tmp_path / "work"
    audio = tmp_path / "audio"
    for path in (oneshot, work, audio):
        path.mkdir()
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_dir", audio)
    # Running this file alone never starts the app, so nothing has created the
    # directory the session database lives in yet.
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return oneshot, work, audio


def _old(path: Path, seconds: float = 3600) -> Path:
    """Backdate a path past the grace period."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))
    return path


def test_an_old_job_loses_its_source_but_keeps_its_result(roots):
    """The record is history: past the age limit only the upload goes."""
    oneshot, _work, _audio = roots
    old = _job(oneshot, "job_old", age_hours=settings.oneshot_max_age_hours + 1, size=4096)
    fresh = _job(oneshot, "job_fresh", age_hours=0, size=4096)

    sweep = housekeeping.Sweep()
    housekeeping.sweep_oneshot(oneshot, sweep)

    # The result the caller polls survives; the ~4 MB source does not.
    assert old.exists() and (old / "job.json").exists()
    assert not (old / "input.mp3").exists()
    assert fresh.exists() and (fresh / "input.mp3").exists()
    assert sweep.jobs == []
    assert sweep.audio == ["input.mp3"]
    assert sweep.freed_bytes == 4096


def test_jobs_are_never_dropped_for_being_too_many(roots):
    """History is not a cache: a hundred requests all stay listed and pollable."""
    oneshot, _work, _audio = roots
    for index in range(8):
        _job(oneshot, f"job_{index}", age_hours=0)

    original = settings.oneshot_max_jobs
    try:
        settings.oneshot_max_jobs = 2
        sweep = housekeeping.Sweep()
        housekeeping.sweep_oneshot(oneshot, sweep)
    finally:
        settings.oneshot_max_jobs = original

    left = sorted(path.name for path in oneshot.iterdir())
    assert left == [f"job_{index}" for index in range(8)]
    assert sweep.jobs == []


def test_the_size_limit_no_longer_deletes_jobs(roots):
    oneshot, _work, _audio = roots
    for index in range(3):
        _job(oneshot, f"job_{index}", age_hours=index, size=400 * 1024)

    original_jobs, original_mb = settings.oneshot_max_jobs, settings.oneshot_max_mb
    try:
        settings.oneshot_max_jobs = 100
        settings.oneshot_max_mb = 1  # room for two of the three
        sweep = housekeeping.Sweep()
        housekeeping.sweep_oneshot(oneshot, sweep)
    finally:
        settings.oneshot_max_jobs, settings.oneshot_max_mb = original_jobs, original_mb

    # Every record and every source stays; the size limit is reported, not
    # enforced against history (LYRICTRACE_ONESHOT_KEEP_INPUT=false is the
    # default, and all three are inside the age limit here).
    assert sorted(path.name for path in oneshot.iterdir()) == ["job_0", "job_1", "job_2"]
    assert sweep.jobs == []


def test_a_running_job_keeps_everything_in_it(roots):
    oneshot, _work, _audio = roots
    running = _job(oneshot, "job_running", status="running", age_hours=1000)
    (running / "work").mkdir()
    (running / "work" / "vocals.wav").write_bytes(b"x" * 2048)

    original = settings.oneshot_max_age_hours
    try:
        settings.oneshot_max_age_hours = 1
        housekeeping.sweep_oneshot(oneshot, housekeeping.Sweep())
    finally:
        settings.oneshot_max_age_hours = original

    # Its worker is writing into that directory right now: source and slices.
    assert running.exists()
    assert (running / "input.mp3").exists()
    assert (running / "work" / "vocals.wav").exists()


def test_a_kept_job_loses_its_work_dir_but_keeps_its_result(roots):
    """The caller polls job.json; the ASR slices under work/ are not read again."""
    oneshot, _work, _audio = roots
    job = _job(oneshot, "job_a")
    (job / "work").mkdir()
    (job / "work" / "vocals.wav").write_bytes(b"x" * 2048)

    original = settings.oneshot_keep_work
    try:
        settings.oneshot_keep_work = False
        sweep = housekeeping.Sweep()
        housekeeping.sweep_oneshot(oneshot, sweep)
    finally:
        settings.oneshot_keep_work = original

    assert job.exists() and (job / "job.json").exists()
    assert not (job / "work").exists()
    assert sweep.freed_bytes == 2048


def test_keep_work_leaves_the_slices_alone(roots):
    oneshot, _work, _audio = roots
    job = _job(oneshot, "job_a")
    (job / "work").mkdir()
    (job / "work" / "vocals.wav").write_bytes(b"x" * 2048)

    original = settings.oneshot_keep_work
    try:
        settings.oneshot_keep_work = True
        housekeeping.sweep_oneshot(oneshot, housekeeping.Sweep())
    finally:
        settings.oneshot_keep_work = original

    assert (job / "work" / "vocals.wav").exists()


def test_work_dirs_of_deleted_projects_are_removed(roots):
    _oneshot, work, _audio = roots
    (work / "prj_gone").mkdir()
    (work / "prj_gone" / "asr-input.wav").write_bytes(b"x" * 10)
    _old(work / "prj_gone")
    (work / "prj_live").mkdir()

    sweep = housekeeping.Sweep()
    housekeeping.sweep_work(work, {"prj_live"}, set(), sweep)

    assert not (work / "prj_gone").exists()
    assert (work / "prj_live").exists()
    assert sweep.work_dirs == ["prj_gone"]


def test_a_work_dir_that_just_appeared_is_left_alone(roots):
    """A project created after the sweep read its rows must not lose its cache."""
    _oneshot, work, _audio = roots
    (work / "prj_new").mkdir()
    (work / "prj_new" / "vocals.wav").write_bytes(b"x" * 10)

    sweep = housekeeping.Sweep()
    housekeeping.sweep_work(work, set(), set(), sweep)

    assert (work / "prj_new").exists()
    assert sweep.work_dirs == []


def test_a_running_project_keeps_its_work_dir_even_over_the_budget(roots):
    _oneshot, work, _audio = roots
    for name in ("prj_busy", "prj_idle"):
        directory = work / name
        directory.mkdir()
        (directory / "vocals.wav").write_bytes(b"x" * (2 * 1024 * 1024))

    original = settings.work_max_mb
    try:
        settings.work_max_mb = 1
        sweep = housekeeping.Sweep()
        housekeeping.sweep_work(work, {"prj_busy", "prj_idle"}, {"prj_busy"}, sweep)
    finally:
        settings.work_max_mb = original

    assert (work / "prj_busy").exists()  # its task is writing into it
    assert not (work / "prj_idle").exists()  # over budget, nothing using it
    assert sweep.work_dirs == ["prj_idle"]


def test_audio_nothing_points_at_is_removed_and_live_audio_is_not(roots):
    _oneshot, _work, audio = roots
    (audio / "live.mp3").write_bytes(b"x")
    (audio / "orphan.mp3").write_bytes(b"x")
    _old(audio / "orphan.mp3")

    sweep = housekeeping.Sweep()
    housekeeping.sweep_audio(audio, {"live.mp3"}, sweep)

    assert (audio / "live.mp3").exists()
    assert not (audio / "orphan.mp3").exists()
    assert sweep.audio == ["orphan.mp3"]


def test_an_aborted_upload_is_cleared_after_the_grace_period(roots):
    oneshot, _work, _audio = roots
    uploads = oneshot / "_uploads"
    uploads.mkdir()
    stale = uploads / "up_old.wav"
    stale.write_bytes(b"x")
    long_ago = time.time() - (settings.upload_stale_minutes + 1) * 60
    os.utime(stale, (long_ago, long_ago))
    fresh = uploads / "up_new.wav"
    fresh.write_bytes(b"x")

    sweep = housekeeping.Sweep()
    housekeeping.sweep_uploads(uploads, sweep)

    assert not stale.exists()
    assert fresh.exists()
    assert sweep.uploads == 1


def test_a_full_sweep_reads_the_projects_and_never_touches_a_row(roots, fresh_db):
    """One ``run()``: a deleted project's cache goes, a live project's does not."""
    oneshot, work, audio = roots
    (work / "prj_gone").mkdir()
    (work / "prj_gone" / "vocals.wav").write_bytes(b"x" * 10)
    _old(work / "prj_gone")
    (work / "prj_live").mkdir()
    with fresh_db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, audio_filename, audio_path, created_at, updated_at) "
            "VALUES ('prj_live', 'b.mp3', 'live.mp3', ?, ?)",
            (db.now_iso(), db.now_iso()),
        )
    (audio / "live.mp3").write_bytes(b"x")
    (audio / "leftover.mp3").write_bytes(b"x")
    _old(audio / "leftover.mp3")

    sweep = housekeeping.run()

    assert not (work / "prj_gone").exists()  # its project row is gone
    assert (work / "prj_live").exists()
    assert not (audio / "leftover.mp3").exists()
    assert (audio / "live.mp3").exists()
    assert sorted(sweep["work_dirs"]) == ["prj_gone"]
    assert sweep["audio"] == ["leftover.mp3"]
    with fresh_db.connect() as conn:
        # The session database holds other tests' projects too; what matters is
        # that the row this sweep read is still there.
        ids = {row["id"] for row in conn.execute("SELECT id FROM projects")}
    assert "prj_live" in ids


def test_usage_reports_the_limits_and_the_sources_kept(roots):
    oneshot, work, _audio = roots
    _job(oneshot, "job_a", size=1024)

    report = housekeeping.usage()

    assert report["oneshot"]["jobs"] == 1
    # Both the job and its source are still there, so both counts are 1.
    assert report["oneshot"]["with_source"] == 1
    assert report["oneshot"]["max_jobs"] == settings.oneshot_max_jobs
    assert report["oneshot"]["max_age_hours"] == settings.oneshot_max_age_hours
    assert report["oneshot"]["keep_input"] == settings.oneshot_keep_input
    assert report["work"]["max_mb"] == settings.work_max_mb
    assert report["interval_min"] == settings.housekeeping_interval_min


def test_storage_endpoints_report_and_sweep(roots, client, fresh_db):
    oneshot, _work, _audio = roots
    old = _job(oneshot, "job_old", age_hours=settings.oneshot_max_age_hours + 1, size=2048)
    (old / "work").mkdir()
    (old / "work" / "vocals.wav").write_bytes(b"x" * 1024)

    report = client.get("/api/storage").json()
    assert report["oneshot"]["jobs"] == 1
    assert report["oneshot"]["max_age_hours"] == settings.oneshot_max_age_hours

    removed = client.post("/api/storage/cleanup").json()
    # The record is history: the sweep only freed the derived files.
    assert removed["jobs"] == []
    assert removed["audio"] == ["input.mp3"]
    assert (oneshot / "job_old" / "job.json").exists()
    assert not (oneshot / "job_old" / "input.mp3").exists()
    assert client.get("/api/storage").json()["oneshot"]["jobs"] == 1
    assert client.get("/api/storage").json()["oneshot"]["with_source"] == 0
