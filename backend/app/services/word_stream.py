"""Which recogniser produces the word stream the line mapping runs on.

The pipeline needs one thing from a recogniser: tokens with times, so the kana
DP can decide which lyric line is sung where. The recogniser's own per-character
tokens also give each character its time (``char_align`` re-slices them per
line, fills the gaps and spreads stacked characters), so no external forced
aligner is needed for either job.

The recogniser's own forced aligner is off by default (``LYRICTRACE_ASR_ALIGN``,
or the project's ``asr_align`` switch): its aligner is weak at singing
(upstream measured 80% of Japanese characters coming back zero-width), so the
word stream uses the plain transcription instead. With the aligner off, a
recording that needs per-word tokenisation still gets it from the recognition
itself.
"""

from __future__ import annotations

from pathlib import Path

from ..config import settings
from ..schemas import Word
from .providers.base import ASRProvider, ASRResult, registry


class WordStream(ASRProvider):
    """The configured ASR service, optionally with its own forced aligner on."""

    name = "word-stream"
    kind = "asr"

    def __init__(self, language: str | None, *, use_asr_align: bool) -> None:
        self.language = language
        self.use_asr_align = use_asr_align
        self.notes: list[str] = []
        self._active: ASRProvider = _with_align(registry.asr(), use_asr_align)
        if not self._active.available():
            self.notes.append(self._active.unavailable_detail())

    def available(self) -> bool:
        return self._active.available()

    @property
    def recogniser(self) -> str:
        """Which service actually produced (or will produce) the tokens."""
        return getattr(self._active, "name", type(self._active).__name__)

    def unavailable_detail(self) -> str:
        return self._active.unavailable_detail()

    async def transcribe(self, audio_path: Path, language: str | None = None) -> ASRResult:
        return await self._active.transcribe(audio_path, language)


def _with_align(provider: ASRProvider, align: bool) -> ASRProvider:
    switch = getattr(provider, "with_align", None)
    return switch(align) if switch is not None else provider


def for_project(language: str | None, *, asr_align: bool | None) -> WordStream:
    """The word stream for a run; ``None`` means the configured default (off)."""
    if asr_align is None:
        asr_align = settings.asr_align
    return WordStream(language, use_asr_align=bool(asr_align))


def words_of(result: ASRResult) -> list[Word]:
    return [word for word in result.words if word.start is not None and word.end is not None]
