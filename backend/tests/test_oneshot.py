"""One-shot endpoint: audio (+ optional lyrics) in, word timestamps out."""

import time

import pytest

from app.schemas import Word
from app.services.asr_lines import group_words, lines_from_words
from app.services.providers.llm_aligner import AlignReport, LineSpan, coverage_note


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


def test_group_words_breaks_at_pauses():
    words = _words(("あ", 0.0, 0.3), ("い", 0.3, 0.6), ("う", 3.0, 3.3))
    groups = group_words(words)
    assert [len(group) for group in groups] == [2, 1]


def test_group_words_caps_very_long_runs():
    words = _words(*[(text, index * 0.2, index * 0.2 + 0.2) for index, text in enumerate("あ" * 40)])
    groups = group_words(words, max_chars=10)
    assert all(sum(len(word.text) for word in group) <= 10 for group in groups)
    assert sum(len(group) for group in groups) == 40


def test_lines_from_words_keeps_asr_text_and_timing():
    lines = lines_from_words(_words(("残機", 1.0, 1.4), ("だ", 1.4, 1.6)))
    assert len(lines) == 1
    line = lines[0]
    assert line.text == "残機だ"
    assert line.start == 1.0 and line.end == 1.6
    assert [word.text for word in line.words] == ["残機", "だ"]
    # The provenance is explicit: this text is the recogniser's, not the lyrics'.
    assert line.note


def test_coverage_note_flags_words_without_lyrics():
    words = _words(*[(text, index, index + 1) for index, text in enumerate("あいうえお")])
    report = AlignReport(mapped={0: LineSpan(0, 0, 0)})
    note = coverage_note(report, words)
    assert note and "没有对应歌词" in note


def test_coverage_note_silent_when_everything_is_covered():
    words = _words(("あ", 0.0, 1.0), ("い", 1.0, 2.0))
    report = AlignReport(mapped={0: LineSpan(index=0, first=0, last=1, confidence=1.0)})
    assert coverage_note(report, words) is None


@pytest.fixture()
def fake_asr(monkeypatch):
    """Replace the network ASR with a fixed token stream."""
    from app.services import oneshot

    words = _words(("君の", 0.0, 0.6), ("心", 0.6, 1.2), ("が", 1.2, 1.4), ("見えない", 4.0, 4.8))

    def fake_transcribe(
        audio_path, work_dir, *, language, asr_provider, on_progress=None, separate=True, notes=None,
        trace=None, meta=None,
    ):
        if on_progress:
            on_progress(0.5, "识别完成")
        return [word.model_copy() for word in words]

    monkeypatch.setattr(oneshot, "transcribe_audio", fake_transcribe)
    return words


def _wait_for_job(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/align/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.05)
    raise AssertionError("job did not settle in time")


def test_oneshot_does_not_inherit_the_servers_asr_language(client, fake_asr, monkeypatch):
    """Omitting ``language`` means auto, not "whatever the deployment defaults to".

    ``LYRICTRACE_ASR_LANGUAGE`` is the ASR provider's default for the project
    flow. The one-shot API used to pass it through, so every caller that left
    the field empty had its song recognised as the server's language (Japanese
    on this deployment) — Chinese songs came back as Japanese.
    """
    from app.config import settings
    from tests.conftest import AUDIO_FIXTURE

    monkeypatch.setattr(settings, "asr_language", "Japanese")
    with AUDIO_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/align",
            files={"file": ("song.mp3", handle, "audio/mpeg")},
            data={"separate": "false"},
        )
    assert response.status_code == 202, response.text
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    # Nothing in "song.mp3" names a language, so the run falls back to auto.
    assert job["result"]["language"] == ""
    assert job["language"] is None  # the record says auto, not "Japanese"
    assert any("自动识别" in note for note in job["result"]["notes"])


def test_oneshot_honours_an_explicit_language(client, fake_asr):
    """An explicit value is still passed through (``ja`` → ``Japanese``)."""
    from tests.conftest import AUDIO_FIXTURE

    with AUDIO_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/align",
            files={"file": ("song.mp3", handle, "audio/mpeg")},
            data={"separate": "false", "language": "ja"},
        )
    assert response.status_code == 202, response.text
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["language"] == "Japanese"
    # What the run settled on is written back, so the cache list can show it.
    assert job["language"] == "Japanese"


def test_the_original_filename_survives_the_upload(client, fake_asr):
    """``input.<ext>`` is the on-disk name; the name the caller sent is what the
    metadata lookup runs on, so it must be recorded, not replaced."""
    from tests.conftest import AUDIO_FIXTURE

    with AUDIO_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/align",
            files={"file": ("【秋葵AKIAOI】Ham.m4a", handle, "audio/mp4")},
            data={"separate": "false"},
        )
    assert response.status_code == 202, response.text
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["filename"] == "【秋葵AKIAOI】Ham.m4a"
    # The audio endpoint serves it back under the name the caller used.
    audio = client.get(f"/api/align/{job['job_id']}/audio")
    assert audio.status_code == 200
    assert "Ham.m4a" in audio.headers["content-disposition"]


def test_align_without_lyrics_returns_asr_lines(client, fake_asr):
    from tests.conftest import AUDIO_FIXTURE

    with AUDIO_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/align",
            files={"file": ("song.mp3", handle, "audio/mpeg")},
            data={"separate": "false"},
        )
    assert response.status_code == 202, response.text
    job = _wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["text_source"] == "asr"
    assert len(result["lines"]) == 2
    assert result["lines"][0]["words"][0]["start"] == 0.0
    assert len(result["words"]) == len(fake_asr)


def test_align_rejects_non_audio(client):
    response = client.post("/api/align", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert response.status_code == 415


def test_align_job_404(client):
    assert client.get("/api/align/job_missing").status_code == 404


def test_recover_stale_fails_orphaned_jobs(tmp_path):
    """A restart orphans the worker thread; polling must not hang forever."""
    from app.services.oneshot import OneShotJobs

    jobs = OneShotJobs(root=tmp_path)
    try:
        (tmp_path / "job_a").mkdir()
        (tmp_path / "job_a" / "job.json").write_text(
            '{"job_id": "job_a", "status": "running", "stage": "asr", "progress": 0.1,'
            ' "message": "分离人声", "error": null, "created_at": null, "finished_at": null,'
            ' "language": null, "with_lyrics": false, "result": null}',
            encoding="utf-8",
        )
        (tmp_path / "job_b").mkdir()
        (tmp_path / "job_b" / "job.json").write_text(
            '{"job_id": "job_b", "status": "succeeded", "result": {"lines": []}}', encoding="utf-8"
        )
        assert jobs.recover_stale() == 1
        assert jobs.get("job_a")["status"] == "failed"
        assert "重启" in jobs.get("job_a")["error"]
        # Finished jobs are left untouched.
        assert jobs.get("job_b")["status"] == "succeeded"
    finally:
        jobs.shutdown()


def test_transcribe_creates_its_work_dir(tmp_path, monkeypatch):
    """``separate=false`` skips the separator, which used to be what made the
    directory; ffmpeg then failed on a missing output path."""
    from app.services import pipeline
    from app.services.providers.base import registry

    class FakeASR:
        def transcribe(self, path, language):
            raise AssertionError("should not be reached")

    class FakeSeparator:
        def available(self):
            return False

    monkeypatch.setattr(registry, "separator", lambda name=None: FakeSeparator())
    work = tmp_path / "nested" / "work"
    with pytest.raises(Exception):
        pipeline.transcribe_audio(
            tmp_path / "missing.mp3", work, language=None, asr_provider=FakeASR(), separate=False
        )
    assert work.is_dir()


def test_the_lyrics_lookup_is_told_how_long_the_audio_is(client, fake_asr, monkeypatch):
    """The cover rule needs the running time on both sides.

    Without it ``_plausible`` can only compare titles, so a cover (credited to
    the singer, lyrics filed under the original) matched nothing at all — the
    same upload in a project, which has a duration, found the lyrics.
    """
    from app.config import settings
    from app.services import lyrics_fetch
    from tests.conftest import AUDIO_FIXTURE

    seen: dict = {}

    def capture(project):
        seen.update(project)
        return None

    monkeypatch.setattr(lyrics_fetch, "auto_fetch", capture)
    monkeypatch.setattr(settings, "auto_search", True)
    try:
        with AUDIO_FIXTURE.open("rb") as handle:
            response = client.post(
                "/api/align",
                files={"file": ("鸟憨 - 勾指起誓.m4a", handle, "audio/mp4")},
                data={"separate": "false"},
            )
        assert response.status_code == 202, response.text
        _wait_for_job(client, response.json()["job_id"])
    finally:
        settings.auto_search = False

    assert seen["title"]
    assert seen["duration"] and seen["duration"] > 0


def test_oneshot_fetches_lyrics_from_the_api_when_none_are_sent(client, fake_asr, monkeypatch):
    """The one-shot API should use the API lyrics rather than fall back to ASR text."""
    from app.config import settings
    from app.services import lyrics_fetch, oneshot
    from tests.conftest import AUDIO_FIXTURE

    match = lyrics_fetch.LyricsMatch(
        candidate=lyrics_fetch.LyricsCandidate(
            track_name="残機", artist_name="ずっと真夜中でいいのに。", lrclib_id=34756429
        ),
        score=0.93,
        lyrics="君の心が見えない",
    )
    monkeypatch.setattr(lyrics_fetch, "auto_fetch", lambda project: match)

    async def no_other_edition(project, lines, *, exclude=None):
        return None

    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_other_edition)

    class FakeAligner:
        def available(self):
            return True

        async def align(
            self, audio_path, lines, *, asr_words=None, language=None, on_progress=None,
            reference=None, keep_extras=True, trace=None, repair=None, resolve_extra=None,
            detect_extras=False,
        ):
            from app.schemas import Line, Timeline

            timeline = Timeline(lines=[Line(text=line.text, start=0.0, end=1.0) for line in lines])
            return timeline

    monkeypatch.setattr(oneshot.registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(settings, "auto_search", True)
    try:
        with AUDIO_FIXTURE.open("rb") as handle:
            response = client.post(
                "/api/align",
                files={"file": ("ずっと真夜中でいいのに。 - 残機.mp3", handle, "audio/mpeg")},
                data={"separate": "false"},
            )
        job = _wait_for_job(client, response.json()["job_id"])
    finally:
        settings.auto_search = False

    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["text_source"] == "lyrics"
    assert [line["text"] for line in result["lines"]] == ["君の心が見えない"]
    assert any("LRCLIB" in note for note in result["notes"])


def test_project_align_falls_back_to_the_asr_transcript_without_lyrics(client, monkeypatch):
    """An original song has no LRCLIB entry; the run must still finish."""
    from app.config import settings
    from app.schemas import Word
    from app.services import pipeline
    from app.services.providers.base import registry
    from tests.conftest import AUDIO_FIXTURE, wait_for_task

    words = [
        Word(text="君の", start=0.0, end=0.6, source="asr"),
        Word(text="心", start=0.6, end=1.2, source="asr"),
        Word(text="見えない", start=4.0, end=4.8, source="asr"),
    ]

    class FakeASR:
        def available(self):
            return True

        async def transcribe(self, path, language):
            raise AssertionError("transcribe_audio is replaced below")

    def fake_transcribe(audio_path, work_dir, *, language, asr_provider, on_progress=None,
                        separate=True, notes=None, trace=None, meta=None):
        return [word.model_copy() for word in words]

    monkeypatch.setattr(registry, "asr", lambda name=None: FakeASR())
    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(settings, "auto_search", False)

    with AUDIO_FIXTURE.open("rb") as handle:
        created = client.post(
            "/api/projects", files={"file": ("原创曲 - 泠鸢.mp3", handle, "audio/mpeg")}
        ).json()
    task = client.post(f"/api/projects/{created['id']}/align", json={}).json()
    finished = wait_for_task(client, task["id"], timeout=15)
    assert finished["status"] == "succeeded", finished

    data = client.get(f"/api/projects/{created['id']}").json()
    timeline = data["timeline"]
    assert [line["text"] for line in timeline["lines"]] == ["君の心", "見えない"]
    assert all(line["source"] == "asr" for line in timeline["lines"])
    assert any("ASR" in note for note in timeline["notes"])
    # The lyric field stays empty: nothing was invented to fill it.
    assert data["original_lyrics"] == ""


def test_a_later_align_retries_the_lyrics_lookup_the_fallback_filled_in(client, monkeypatch, fake_aligner):
    """Once the ASR transcript filled the timeline, a fixed lookup must still win.

    The transcript lives in ``revised_lyrics`` too, so "the project has text"
    is not the same as "the project has lyrics": only ``original_lyrics`` says
    the sources (or the user) supplied the words.
    """
    from app.config import settings
    from app.schemas import Word
    from app.services import lyrics_fetch, pipeline
    from app.services.providers.base import LyricsCandidate, registry
    from tests.conftest import AUDIO_FIXTURE, wait_for_task

    words = [
        Word(text="君の", start=0.0, end=0.6, source="asr"),
        Word(text="心", start=0.6, end=1.2, source="asr"),
    ]

    class FakeASR:
        def available(self):
            return True

        async def transcribe(self, path, language):
            raise AssertionError("transcribe_audio is replaced below")

    def fake_transcribe(audio_path, work_dir, *, language, asr_provider, on_progress=None,
                        separate=True, notes=None, trace=None, meta=None):
        return [word.model_copy() for word in words]

    async def found(project, *, query=None, sources=None):
        return [
            lyrics_fetch.LyricsMatch(
                candidate=LyricsCandidate(
                    track_name="残機", artist_name="ずっと真夜中でいいのに。", plain_lyrics="君の心\n見えない"
                ),
                score=1.0,
                lyrics="君の心\n見えない",
            )
        ]

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            from app.schemas import Line, Timeline

            return Timeline(
                lines=[
                    Line(
                        text=line.text,
                        start=asr_words[0].start if asr_words else 0.0,
                        end=asr_words[-1].end if asr_words else 1.0,
                    )
                    for line in lines
                ]
            )

    monkeypatch.setattr(registry, "asr", lambda name=None: FakeASR())
    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(lyrics_fetch, "search", found)

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    with AUDIO_FIXTURE.open("rb") as handle:
        created = client.post(
            "/api/projects", files={"file": ("原创曲 - 泠鸢.mp3", handle, "audio/mpeg")}
        ).json()
    try:
        first = client.post(f"/api/projects/{created['id']}/align", json={}).json()
        assert wait_for_task(client, first["id"], timeout=15)["status"] == "succeeded"
        data = client.get(f"/api/projects/{created['id']}").json()
        assert data["original_lyrics"] == ""
        assert data["revised_lyrics"]  # the ASR transcript is there

        # The sources answer now; the earlier miss must not be permanent.
        monkeypatch.setattr(settings, "auto_search", True)
        monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
        second = client.post(f"/api/projects/{created['id']}/align", json={}).json()
        assert wait_for_task(client, second["id"], timeout=15)["status"] == "succeeded"
    finally:
        settings.auto_search = False

    data = client.get(f"/api/projects/{created['id']}").json()
    assert data["original_lyrics"] == "君の心\n見えない"
    assert data["lyrics_source"] == "lrclib"
    assert [line["text"] for line in data["timeline"]["lines"]] == ["君の心", "見えない"]


def test_changing_the_language_drops_the_cached_asr(client, project):
    """A transcript is language-specific: recognising it again is the point."""
    from app import db
    from app.services import pipeline

    assert pipeline._cache_matches_language({"asr_language": "Chinese"}, "Japanese") is False
    assert pipeline._cache_matches_language({"asr_language": "Chinese"}, "Chinese") is True
    # A cache predating the column carries no language; keep it, as before.
    assert pipeline._cache_matches_language({"asr_language": None}, "Japanese") is True

    with db.connect() as conn:
        db.update_project(conn, project["id"], asr_json='{"words": []}', asr_language="Chinese")

    response = client.put(f"/api/projects/{project['id']}/lyrics", json={"language": "Japanese"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["lyrics_language"] == "Japanese"
    assert body["language_origin"] == "manual"
    # The stale tokens are gone; the next run recognises the audio again.
    assert client.get(f"/api/projects/{project['id']}/asr").json()["words"] == []


def test_project_align_skips_the_other_edition_search_when_switched_off(client, monkeypatch, fake_aligner):
    """Off means: use the lyrics entry's own 逐字 data, never another edition."""
    from app.config import settings
    from app.schemas import Line, Timeline, Word
    from app.services import lyrics_fetch, pipeline
    from app.services.providers.base import registry
    from tests.conftest import AUDIO_FIXTURE, wait_for_task

    words = [
        Word(text="君の", start=0.0, end=0.6, source="asr"),
        Word(text="心", start=0.6, end=1.2, source="asr"),
    ]

    searched: list[bool] = []

    class FakeASR:
        def available(self):
            return True

    def fake_transcribe(audio_path, work_dir, *, language, asr_provider, on_progress=None,
                        separate=True, notes=None, trace=None, meta=None):
        return [word.model_copy() for word in words]

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            return Timeline(lines=[Line(text=line.text, start=0.0, end=1.2) for line in lines])

    async def no_search(*_args, **_kwargs):
        searched.append(True)
        return None

    monkeypatch.setattr(registry, "asr", lambda name=None: FakeASR())
    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_search)

    with AUDIO_FIXTURE.open("rb") as handle:
        created = client.post(
            "/api/projects", files={"file": ("残機 - ずっと真夜中でいいのに。.mp3", handle, "audio/mpeg")}
        ).json()
    pid = created["id"]
    # Let the upload task settle before the sources are switched on, so it does
    # not start a network lookup of its own.
    for item in client.get(f"/api/projects/{pid}/tasks").json():
        wait_for_task(client, item["id"], timeout=15)
    client.put(f"/api/projects/{pid}/lyrics", json={"original_lyrics": "君の心"})
    assert client.patch(f"/api/projects/{pid}", json={"word_sync_search": False}).status_code == 200

    monkeypatch.setattr(settings, "auto_search", True)
    try:
        task = client.post(f"/api/projects/{pid}/align", json={}).json()
        assert wait_for_task(client, task["id"], timeout=15)["status"] == "succeeded"
    finally:
        settings.auto_search = False

    assert searched == []
    # The debug page says the search was skipped rather than that it found nothing.
    steps = client.get(f"/api/projects/{pid}/debug").json()["runs"][0]["steps"]
    search = next(step for step in steps if step["step"] == "word_sync_search")
    assert search["payload"]["skipped"] is True
    assert search["payload"]["found"] is False


def test_oneshot_skips_the_other_edition_search_when_switched_off(tmp_path, monkeypatch):
    """The one-shot API honours the same switch: no search, no edition note."""
    from app.config import settings
    from app.services import lyrics_fetch
    from app.services.oneshot import OneShotJobs

    match = lyrics_fetch.LyricsMatch(
        candidate=lyrics_fetch.LyricsCandidate(
            track_name="残機", artist_name="ずっと真夜中でいいのに。", plain_lyrics="君の心"
        ),
        score=1.0,
        lyrics="君の心",
    )

    async def should_not_search(*_args, **_kwargs):
        raise AssertionError("the other-edition search ran while switched off")

    monkeypatch.setattr(lyrics_fetch, "auto_fetch", lambda project: match)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", should_not_search)
    monkeypatch.setattr(settings, "auto_search", True)
    jobs = OneShotJobs(root=tmp_path)
    try:
        notes: list[str] = []
        lyrics, _synced, sync_lines = jobs._resolve_lyrics(
            "job_off", "", "残機.mp3", notes, word_sync_search=False
        )
    finally:
        settings.auto_search = False
        jobs.shutdown()
    assert lyrics == "君の心"
    assert sync_lines == []
    assert not any("其他版本" in note for note in notes)


def test_oneshot_uses_source_word_timings_when_they_fit_the_audio(tmp_path):
    """A source's 逐字 timings replace the local pipeline only when they agree."""
    from app.schemas import Word
    from app.services.oneshot import OneShotJobs
    from app.services.word_sync import parse_lyricsfile

    source = parse_lyricsfile(
        "lines:\n"
        "- text: 君の心\n  start_ms: 1000\n  words:\n"
        "  - text: 君の\n    start_ms: 1000\n    end_ms: 1600\n"
        "  - text: 心\n    start_ms: 1600\n    end_ms: 2200\n"
    )
    words = [
        Word(text="君の", start=1.5, end=2.1, source="asr"),
        Word(text="心", start=2.1, end=2.7, source="asr"),
    ]
    jobs = OneShotJobs(root=tmp_path)
    try:
        result = jobs._build_result(
            words, "君の心", "Japanese", lambda _p, _m: None, [], source_sync=source
        )
    finally:
        jobs.shutdown()
    line = result["lines"][0]
    assert line["source"] == "word-sync"
    # The source said 1.0 s; this recording sings it at 1.5 s.
    assert line["start"] == 1.5
    assert [word["start"] for word in line["words"]] == [1.5, 2.1]
    assert any("逐字歌词" in note for note in result["notes"])


def test_oneshot_ignores_source_word_timings_that_do_not_fit(tmp_path, monkeypatch):
    from app.schemas import Line, Timeline, Word
    from app.services.oneshot import OneShotJobs
    from app.services.word_sync import parse_lyricsfile

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            return Timeline(lines=[Line(text=line.text, start=asr_words[0].start, end=asr_words[-1].end) for line in lines])

    source = parse_lyricsfile(
        "lines:\n"
        "- text: 君の心\n  start_ms: 1000\n  words:\n"
        "  - text: 君の\n    start_ms: 1000\n    end_ms: 1600\n"
        "  - text: 心\n    start_ms: 1600\n    end_ms: 2200\n"
        "- text: 見えない\n  start_ms: 3000\n  words:\n"
        "  - text: 見え\n    start_ms: 3000\n    end_ms: 3600\n"
        "  - text: ない\n    start_ms: 3600\n    end_ms: 4200\n"
    )
    words = [
        Word(text="君の", start=30.0, end=30.6, source="asr"),
        Word(text="心", start=30.6, end=31.2, source="asr"),
        Word(text="見え", start=90.0, end=90.6, source="asr"),
        Word(text="ない", start=90.6, end=91.2, source="asr"),
    ]
    monkeypatch.setattr("app.services.oneshot.registry.aligner", lambda name=None: FakeAligner())
    jobs = OneShotJobs(root=tmp_path)
    try:
        result = jobs._build_result(
            words, "君の心\n見えない", "Japanese", lambda _p, _m: None, [], source_sync=source
        )
    finally:
        jobs.shutdown()
    # The two lines are 29 s and 57 s off the source: that is not one shifted
    # recording but a different edit, so the normal aligner ran instead.
    assert all(line["source"] != "word-sync" for line in result["lines"])
    assert result["lines"][0]["start"] == 30.0
    assert not any("逐字歌词" in note for note in result["notes"])


def test_the_cache_lists_the_newest_jobs_with_their_source(client, fake_asr, tmp_path, monkeypatch):
    """`GET /api/align` is what the web UI's cache page reads: the source file
    stays with the result, so a cached answer can be listened to again."""
    from app.services import oneshot
    from tests.conftest import AUDIO_FIXTURE

    # A job dir of its own, so this test does not count the other tests' jobs.
    jobs = oneshot.OneShotJobs(root=tmp_path / "cache")
    monkeypatch.setattr("app.routers.align.jobs", jobs)
    try:
        with AUDIO_FIXTURE.open("rb") as handle:
            created = client.post(
                "/api/align",
                files={"file": ("残機.mp3", handle, "audio/mpeg")},
                data={"separate": "false"},
            )
        assert created.status_code == 202, created.text
        job_id = created.json()["job_id"]
        assert _wait_for_job(client, job_id)["status"] == "succeeded"

        listing = client.get("/api/align").json()
        assert listing["max_jobs"] == 5
        assert [item["job_id"] for item in listing["items"]] == [job_id]
        item = listing["items"][0]
        assert item["filename"] == "残機.mp3"
        assert item["result"]["lines"]  # the finished per-character answer

        audio = client.get(f"/api/align/{job_id}/audio")
        assert audio.status_code == 200
        assert audio.content  # the uploaded source is still there
        assert client.get("/api/align/job_missing/audio").status_code == 404
    finally:
        jobs.shutdown()


def test_the_cache_keeps_only_the_newest_five(monkeypatch, tmp_path):
    """Six finished jobs, five kept — the newest, with their source files."""
    import json
    from datetime import datetime, timedelta, timezone

    from app.config import settings
    from app.services import housekeeping, oneshot

    jobs = oneshot.OneShotJobs(root=tmp_path / "oneshot")
    try:
        for index in range(6):
            directory = jobs.job_dir(f"job_{index}")
            directory.mkdir(parents=True)
            (directory / "input.mp3").write_bytes(b"x")
            # Inside the 24-hour age limit, so only the count rule can fire.
            stamp = datetime.now(timezone.utc) - timedelta(minutes=index)
            (directory / "job.json").write_text(
                json.dumps(
                    {"job_id": f"job_{index}", "status": "succeeded", "created_at": stamp.isoformat()}
                ),
                encoding="utf-8",
            )
        sweep = housekeeping.Sweep()
        housekeeping.sweep_oneshot(tmp_path / "oneshot", sweep)
    finally:
        jobs.shutdown()

    # job_0 is the newest; job_5 is the one over the count limit.
    left = sorted(path.name for path in (tmp_path / "oneshot").iterdir())
    assert left == [f"job_{index}" for index in range(5)]
    assert sweep.jobs == ["job_5"]
    assert settings.oneshot_cache_jobs == 5
