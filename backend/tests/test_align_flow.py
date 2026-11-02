"""Which flow places the lines, and what each one says when its model is missing.

The default flow takes every line position from the ASR word stream and lets the
recogniser's own tokens time each character; the optional forced aligner (when a
project turns it on) re-times the characters inside each line. Both are exercised
here through the real task, so a change that silently swaps them fails a test.
"""

from __future__ import annotations

import pytest

from app.schemas import Line, Timeline, Word
from tests.conftest import AUDIO_FIXTURE, wait_for_task


@pytest.fixture()
def asr_words(monkeypatch):
    """A fixed word stream, so nothing here needs a recogniser."""
    from app.services import pipeline

    words = [
        Word(text="君の", start=0.0, end=0.6, source="asr"),
        Word(text="心", start=0.6, end=1.2, source="asr"),
    ]

    class FakeStream:
        notes: list[str] = []

        def available(self):
            return True

        def unavailable_detail(self):
            return ""

    def fake_transcribe(audio_path, work_dir, *, language, asr_provider, on_progress=None,
                        separate=True, notes=None, trace=None, meta=None):
        return [word.model_copy() for word in words]

    monkeypatch.setattr(pipeline, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(pipeline.word_stream, "for_project", lambda *a, **k: FakeStream())
    return words


def _project_with_lyrics(client, name: str, lyrics: str) -> str:
    with AUDIO_FIXTURE.open("rb") as handle:
        created = client.post("/api/projects", files={"file": (name, handle, "audio/mpeg")}).json()
    for item in client.get(f"/api/projects/{created['id']}/tasks").json():
        wait_for_task(client, item["id"], timeout=15)
    assert client.put(f"/api/projects/{created['id']}/lyrics", json={"original_lyrics": lyrics}).status_code == 200
    return created["id"]


def _project_without_lyrics(client, name: str) -> str:
    """A project whose lyrics lookup finds nothing: the transcript is the text."""
    with AUDIO_FIXTURE.open("rb") as handle:
        created = client.post("/api/projects", files={"file": (name, handle, "audio/mpeg")}).json()
    for item in client.get(f"/api/projects/{created['id']}/tasks").json():
        wait_for_task(client, item["id"], timeout=15)
    return created["id"]


def _align(client, project_id: str) -> dict:
    task = client.post(f"/api/projects/{project_id}/align", json={}).json()
    return wait_for_task(client, task["id"], timeout=15)


def _debug(client, project_id: str) -> list[dict]:
    return client.get(f"/api/projects/{project_id}/debug").json()["runs"][0]["steps"]


def test_default_flow_takes_line_positions_from_the_asr(client, monkeypatch, asr_words):
    """No aligner: the ASR places the lines and times each character."""
    from app.config import settings
    from app.services import char_align, lyrics_fetch
    from app.services.providers.base import registry

    seen: dict[str, object] = {}

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            seen["asr_words"] = list(asr_words or [])
            return Timeline(
                lines=[
                    Line(text=line.text, start=0.0, end=1.2, source="asr") for line in lines
                ]
            )

    def fake_retime(timeline, **_kwargs):
        seen["retime"] = [line.text for line in timeline.lines]
        return []

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(char_align, "retime", fake_retime)
    monkeypatch.setattr(char_align, "available", lambda: True)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_with_lyrics(client, "残機 - ずっと真夜中でいいのに。.mp3", "君の心\n見えない")
    # The per-character pass runs only when the project asks for the aligner; it is
    # off by default now (the recogniser's own tokens are enough).
    assert client.patch(f"/api/projects/{pid}", json={"hfa_chars": True}).status_code == 200
    assert _align(client, pid)["status"] == "succeeded"

    # The DP ran on the recogniser's tokens: that is what positions the lines.
    assert [word.text for word in seen["asr_words"]] == ["君の", "心"]
    # ... and the per-character pass ran over the lines it produced.
    assert seen["retime"] == ["君の心", "見えない"]
    flow = next(step for step in _debug(client, pid) if step["step"] == "align_flow")
    assert flow["payload"]["flow"] == "asr"


def test_missing_aligner_degrades_to_a_note_in_the_default_flow(client, monkeypatch, asr_words):
    """The default flow must still finish when only the per-character pass is gone.

    Its line positions come from the ASR; losing the forced aligner costs the
    per-character accuracy, not the run. The note has to say so — the timeline
    then holds estimates and may not pass them off as an alignment.
    """
    from app.config import settings
    from app.services import char_align, lyrics_fetch
    from app.services.providers.base import registry

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            return Timeline(
                lines=[
                    Line(text=line.text, start=0.0, end=1.2, source="asr") for line in lines
                ]
            )

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(char_align, "available", lambda: False)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_with_lyrics(client, "我的故事未写完 - 泠鸢.mp3", "君の心\n見えない")
    finished = _align(client, pid)
    assert finished["status"] == "succeeded", finished

    # Neither source can place the characters here, so the note has to say the
    # times are the word stream's estimate rather than a measurement.
    notes = client.get(f"/api/projects/{pid}").json()["timeline"]["notes"]
    assert any("只能按识别词流摊开估计" in note for note in notes)


def test_a_transcript_without_lyrics_is_re_timed_character_by_character(client, monkeypatch, asr_words):
    """No lyrics at all: the transcript still gets per-character times.

    Nothing in that path has a lyric to align against, so the line positions are
    just where the recogniser heard a pause — but the *text* is the recogniser's
    own tokens, which are word-sized (``君の`` is one token for two characters).
    Without the forced aligner those two characters share one highlight block,
    so the per-character pass has to run here too, not only on the lyric path.
    """
    from app.config import settings
    from app.services import char_align, lyrics_fetch

    seen: dict[str, object] = {}

    def fake_retime(timeline, **_kwargs):
        seen["retime"] = [line.text for line in timeline.lines]
        return ["逐字时间来自强制对齐（2/2 句）"]

    async def no_lyrics(*_args, **_kwargs):
        return None

    monkeypatch.setattr(char_align, "retime", fake_retime)
    monkeypatch.setattr(char_align, "available", lambda: True)
    monkeypatch.setattr(lyrics_fetch, "auto_fetch", no_lyrics)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_without_lyrics(client, "无歌词的歌 - 某人.mp3")
    assert client.patch(f"/api/projects/{pid}", json={"hfa_chars": True}).status_code == 200
    assert _align(client, pid)["status"] == "succeeded"

    # The transcript became the lines, and the per-character pass saw them.
    assert seen["retime"] == ["君の心"]
    notes = client.get(f"/api/projects/{pid}").json()["timeline"]["notes"]
    assert any("强制对齐" in note for note in notes)


def test_the_semantic_merge_is_a_project_switch_and_runs_when_asked(
    client, monkeypatch, asr_words
):
    """「按语意合并成词」 is off by default; on, the LLM says where words end."""
    from app.config import settings
    from app.services import char_align, llm_merge, lyrics_fetch
    from app.services.providers.base import registry

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            return Timeline(
                lines=[Line(text=line.text, start=0.0, end=1.2, source="asr") for line in lines]
            )

    def fake_retime(timeline, **_kwargs):
        for line in timeline.lines:
            chars = list(line.text)
            line.words = [
                Word(
                    text=char,
                    start=round(0.1 * index, 3),
                    end=round(0.1 * (index + 1), 3),
                    source="asr",
                )
                for index, char in enumerate(chars)
            ]
            line.start, line.end = line.words[0].start, line.words[-1].end
        return []

    seen: dict[str, object] = {}

    def fake_apply(timeline, **kwargs):
        seen["lines"] = [line.text for line in timeline.lines]
        seen["enabled"] = kwargs.get("enabled")
        return ["逐字结果已按语意合并成词"]

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(char_align, "retime", fake_retime)
    monkeypatch.setattr(char_align, "per_character_available", lambda *a, **k: True)
    monkeypatch.setattr(llm_merge, "apply", fake_apply)
    monkeypatch.setattr(llm_merge, "available", lambda enabled=True: bool(enabled))
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_with_lyrics(client, "秒針を噛む - ずっと真夜中でいいのに。.mp3", "君の心\n見えない")
    assert _align(client, pid)["status"] == "succeeded"
    # Default off: the per-character times are what was measured, no request.
    assert seen == {}

    assert client.patch(f"/api/projects/{pid}", json={"llm_merge": True}).status_code == 200
    assert _align(client, pid)["status"] == "succeeded"

    # It runs after the per-character pass, on the finished timeline.
    assert seen["lines"] == ["君の心", "見えない"]
    assert seen["enabled"] is True
    notes = client.get(f"/api/projects/{pid}").json()["timeline"]["notes"]
    assert any("按语意合并" in note for note in notes)


def test_aligner_is_not_asked_for_characters_unless_the_project_asks(client, monkeypatch, asr_words):
    """``hfa_chars`` off (the default) reaches ``retime`` as ``use_hfa=False``."""
    from app.config import settings
    from app.services import char_align, llm_merge, lyrics_fetch
    from app.services.providers.base import registry

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **_kwargs):
            return Timeline(
                lines=[Line(text=line.text, start=0.0, end=1.2, source="asr") for line in lines]
            )

    seen: dict[str, object] = {}

    def fake_retime(timeline, **kwargs):
        seen["use_hfa"] = kwargs.get("use_hfa")
        return []

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(char_align, "retime", fake_retime)
    monkeypatch.setattr(char_align, "per_character_available", lambda *a, **k: True)
    monkeypatch.setattr(llm_merge, "available", lambda *args, **kwargs: False)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_with_lyrics(client, "残機 - ずっと真夜中でいいのに。.mp3", "君の心\n見えない")
    assert _align(client, pid)["status"] == "succeeded"
    assert seen["use_hfa"] is False

    assert client.patch(f"/api/projects/{pid}", json={"hfa_chars": True}).status_code == 200
    assert _align(client, pid)["status"] == "succeeded"
    assert seen["use_hfa"] is True


def test_the_live_extras_switch_reaches_the_aligner(client, monkeypatch, asr_words):
    """「识别现场加唱」 is off by default and travels as ``detect_extras``."""
    from app.config import settings
    from app.services import char_align, llm_merge, lyrics_fetch
    from app.services.providers.base import registry

    seen: list[bool] = []

    class FakeAligner:
        def available(self):
            return True

        async def align(self, audio_path, lines, *, asr_words=None, **kwargs):
            seen.append(kwargs.get("detect_extras"))
            return Timeline(
                lines=[Line(text=line.text, start=0.0, end=1.2, source="asr") for line in lines]
            )

    async def no_edition(*_args, **_kwargs):
        return None

    monkeypatch.setattr(registry, "aligner", lambda name=None: FakeAligner())
    monkeypatch.setattr(char_align, "retime", lambda timeline, **_kwargs: [])
    monkeypatch.setattr(char_align, "per_character_available", lambda *a, **k: False)
    monkeypatch.setattr(llm_merge, "available", lambda *args, **kwargs: False)
    monkeypatch.setattr(lyrics_fetch, "find_word_sync", no_edition)
    monkeypatch.setattr(settings, "auto_search", False)

    pid = _project_with_lyrics(client, "残機 - ずっと真夜中でいいのに。.mp3", "君の心\n見えない")
    assert _align(client, pid)["status"] == "succeeded"
    # Default off: only the lyric lines come back.
    assert seen == [False]

    assert client.patch(f"/api/projects/{pid}", json={"detect_extras": True}).status_code == 200
    assert _align(client, pid)["status"] == "succeeded"
    assert seen == [False, True]
