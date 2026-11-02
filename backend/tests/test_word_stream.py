"""Which recogniser supplies the word stream, and what happens when it cannot."""

from pathlib import Path

import pytest

from app.schemas import Word
from app.services import word_stream
from app.services.providers.base import ASRProvider, ASRResult


class _Fake(ASRProvider):
    name = "fake"
    kind = "asr"

    def __init__(self, label: str, *, fail: bool = False, align: bool = False) -> None:
        self.label = label
        self.fail = fail
        self.align = align

    def available(self) -> bool:
        return not self.fail

    def unavailable_detail(self) -> str:
        return f"{self.label} unavailable"

    def with_align(self, align: bool):
        return _Fake(self.label, fail=self.fail, align=align)

    async def transcribe(self, audio_path: Path, language: str | None = None) -> ASRResult:
        return ASRResult(words=[Word(text=self.label, start=0.0, end=1.0, source="asr")])


@pytest.fixture()
def asr_service(monkeypatch):
    """The ASR service, available or not, for each test."""
    holder: dict[str, _Fake] = {"service": _Fake("qwen")}

    def install(*, fail: bool = False):
        holder["service"] = _Fake("qwen", fail=fail)
        monkeypatch.setattr(word_stream.registry, "asr", lambda name=None: holder["service"])
        return holder

    return install


def _transcribe(stream) -> str:
    import asyncio

    result = asyncio.run(stream.transcribe(Path("audio.wav"), "Japanese"))
    return "".join(word.text for word in result.words)


def test_asr_alignment_off_uses_the_asr_service_plain_transcription(asr_service):
    asr_service()
    stream = word_stream.for_project("Japanese", asr_align=False)
    # The aligner is off by default: the word stream is the recogniser's plain
    # transcription, not a forced-aligner service.
    assert stream._active.label == "qwen"
    assert stream._active.align is False
    assert _transcribe(stream) == "qwen"
    assert stream.notes == []


def test_turning_asr_alignment_on_enables_the_aligner(asr_service):
    asr_service()
    stream = word_stream.for_project("Japanese", asr_align=True)
    assert stream._active.label == "qwen"
    assert stream._active.align is True
    assert _transcribe(stream) == "qwen"
    assert stream.notes == []


def test_an_unavailable_asr_service_is_reported(asr_service):
    asr_service(fail=True)
    stream = word_stream.for_project("Japanese", asr_align=False)
    assert stream.available() is False
    assert stream.notes and "unavailable" in stream.notes[0]
