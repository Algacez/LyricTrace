"""Pluggable service interfaces for the lyric-sync pipeline.

Every heavy model (ASR, forced alignment, source separation, LLM correction,
LLM metadata extraction) and every external importer (Bilibili, YouTube) is
expressed as a small abstract provider. The default registry wires in only the
implementations that actually work today:

* ``llm`` metadata extractor (reads the messy filename; falls back to the
  rule-based parser when the LLM is unreachable) and ``filename``
* ``lrclib`` lyrics provider

Everything else resolves to ``UnavailableProvider``, which raises a 503 with a
clear message instead of silently returning fake data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...config import settings
from ...schemas import Line, SongMetadata, Timeline


class ProviderUnavailable(RuntimeError):
    """Raised when a configured provider has no working implementation yet."""


class Provider(ABC):
    name: str = "base"
    kind: str = "base"
    description: str = ""

    @abstractmethod
    def available(self) -> bool: ...

    def unavailable_detail(self) -> str:
        return "Not implemented in this build."


class UnavailableProvider(Provider):
    def __init__(self, name: str, kind: str, description: str, detail: str) -> None:
        self.name = name
        self.kind = kind
        self.description = description
        self._detail = detail

    def available(self) -> bool:
        return False

    def unavailable_detail(self) -> str:
        return self._detail


# --------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------


class MetadataExtractor(Provider):
    kind = "metadata_extractor"

    @abstractmethod
    def extract(self, filename: str, embedded: dict[str, Any] | None = None) -> SongMetadata: ...


class FilenameMetadataExtractor(MetadataExtractor):
    """Heuristic parser: strips video/platform noise and splits title/artist.

    Kept as the offline fallback for the LLM extractor, which shares the same
    ``extract`` signature.
    """

    name = "filename"
    description = "Rule-based filename and tag parsing (no network calls)"

    def available(self) -> bool:
        return True

    def extract(self, filename: str, embedded: dict[str, Any] | None = None) -> SongMetadata:
        from ..filename_parser import parse_filename

        parsed = parse_filename(filename)
        embedded = embedded or {}
        title = embedded.get("title") or parsed.title
        artist = embedded.get("artist") or parsed.artist
        album = embedded.get("album") or parsed.album
        return SongMetadata(
            title=title,
            artist=artist,
            album=album,
            duration=embedded.get("duration"),
            source="embedded" if embedded.get("title") else "filename",
        )


# --------------------------------------------------------------------------
# Lyrics retrieval
# --------------------------------------------------------------------------


@dataclass
class LyricsCandidate:
    track_name: str = ""
    artist_name: str = ""
    album_name: str = ""
    duration: float | None = None
    instrumental: bool = False
    plain_lyrics: str | None = None
    synced_lyrics: str | None = None
    word_sync_raw: str | None = None
    """The provider's word-level (逐字) payload, verbatim.

    LRCLIB hands over a ``lyricsfile`` (YAML, ``lines[].words``) when
    ``hasWordSync`` is set; NetEase hands over ``yrc`` text. Parsed by
    :mod:`app.services.word_sync`, never by the provider itself.
    """
    word_sync_format: str | None = None
    """``lyricsfile`` / ``yrc``: which parser the payload needs."""
    lrclib_id: int | None = None
    score: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    source: str = "lrclib"
    """Which provider answered: ``lrclib``, ``lrcapi`` or ``netease``."""
    external_id: str | None = None
    """That provider's own id, needed to select the entry later."""


class LyricsProvider(Provider):
    kind = "lyrics_provider"

    @abstractmethod
    async def search(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
    ) -> list[LyricsCandidate]: ...


# --------------------------------------------------------------------------
# ASR / alignment / separation / correction / import
# --------------------------------------------------------------------------


@dataclass
class ASRResult:
    """Raw ASR output: timed tokens (chars for CJK, words for latin)."""

    words: list[Word] = field(default_factory=list)
    text: str = ""
    cues: list[dict[str, Any]] = field(default_factory=list)
    language: str | None = None
    model: str | None = None


class ASRProvider(Provider):
    kind = "asr"

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: str | None = None) -> ASRResult: ...


class AlignerProvider(Provider):
    kind = "aligner"

    @abstractmethod
    async def align(
        self,
        audio_path: Path,
        lines: list[Line],
        *,
        asr_words: list[Word] | None = None,
        language: str | None = None,
        on_progress: Callable[[float, str], None] | None = None,
        reference: dict[int, Line] | None = None,
        keep_extras: bool = True,
        detect_extras: bool = False,
        trace: Any = None,
    ) -> Timeline: ...


class SeparatorProvider(Provider):
    kind = "separator"

    @abstractmethod
    async def separate(self, audio_path: Path, out_dir: Path) -> dict[str, Path]: ...


class CorrectorProvider(Provider):
    kind = "corrector"

    @abstractmethod
    async def correct(self, lines: list[Line], language: str | None = None) -> list[Line]: ...


class ImporterProvider(Provider):
    kind = "importer"

    @abstractmethod
    async def import_audio(self, url: str, out_dir: Path) -> Path: ...


def _unavailable(name: str, kind: str, description: str, detail: str) -> UnavailableProvider:
    return UnavailableProvider(name, kind, description, detail)


class ProviderRegistry:
    """Resolves configured provider names to instances."""

    def __init__(self) -> None:
        from .llm_metadata import LLMMetadataExtractor

        self._metadata_extractors: dict[str, MetadataExtractor] = {
            "filename": FilenameMetadataExtractor(),
            "llm": LLMMetadataExtractor(),
        }
        from .qwen_asr import QwenASRProvider

        self._asr = {
            "unavailable": _unavailable(
                "qwen3-asr", "asr", "Qwen3-ASR-1.7B", "ASR provider not configured."
            ),
            "qwen-asr-http": QwenASRProvider(
                settings.asr_url, language=settings.asr_language, timeout=settings.asr_timeout
            ),
        }
        from .llm_aligner import LLMLyricAligner

        self._aligner = {
            "unavailable": _unavailable(
                "llm-aligner", "aligner", "LLM lyric alignment", "Aligner not configured."
            ),
            "llm-aligner": LLMLyricAligner(),
        }
        from .bs_roformer import BsRoformerSeparator

        self._separator = {
            "unavailable": _unavailable(
                "bs-roformer", "separator", "BS-RoFormer", "Vocal separation not configured."
            ),
            "bs-roformer": BsRoformerSeparator(
                settings.separator_url, model=settings.separator_model, timeout=settings.separator_timeout
            ),
        }
        self._corrector = {
            "unavailable": _unavailable(
                "llm-corrector", "corrector", "LLM lyric correction", "Correction not configured."
            ),
        }
        self._importer = {
            "unavailable": _unavailable(
                "url-importer", "importer", "Bilibili / YouTube import", "Audio import not configured."
            ),
        }

    def _pick(self, table: dict[str, Provider], configured: str, kind: str) -> Provider:
        provider = table.get(configured)
        if provider is None:
            return _unavailable(
                configured,
                kind,
                configured,
                f"Unknown {kind} provider '{configured}'.",
            )
        return provider

    def metadata_extractor(self, name: str | None = None) -> MetadataExtractor:
        chosen = (name or settings.metadata_extractor).strip().lower()
        provider = self._metadata_extractors.get(chosen)
        if provider is None:
            return self._metadata_extractors["filename"]
        return provider

    def asr(self, name: str | None = None) -> ASRProvider:
        return self._pick(self._asr, (name or settings.asr_provider).lower(), "asr")  # type: ignore[return-value]

    def aligner(self, name: str | None = None) -> AlignerProvider:
        return self._pick(self._aligner, (name or settings.aligner_provider).lower(), "aligner")  # type: ignore[return-value]

    def separator(self, name: str | None = None) -> SeparatorProvider:
        return self._pick(self._separator, (name or settings.separator_provider).lower(), "separator")  # type: ignore[return-value]

    def corrector(self, name: str | None = None) -> CorrectorProvider:
        return self._pick(self._corrector, (name or settings.corrector_provider).lower(), "corrector")  # type: ignore[return-value]

    def importer(self, name: str | None = None) -> ImporterProvider:
        return self._pick(self._importer, (name or settings.importer_provider).lower(), "importer")  # type: ignore[return-value]

    def all(self) -> list[Provider]:
        from .lrclib import LrclibProvider
        from .lrcapi import LrcApiProvider
        from .netease import NeteaseProvider

        providers: list[Provider] = [
            *self._metadata_extractors.values(),
            LrclibProvider(),
            LrcApiProvider(),
            NeteaseProvider(),
        ]
        for table in (self._asr, self._aligner, self._separator, self._corrector, self._importer):
            # de-duplicate the generic "unavailable" entries
            providers.extend(p for p in table.values() if p.name != "unavailable")
        return providers


registry = ProviderRegistry()
