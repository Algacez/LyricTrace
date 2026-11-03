"""Misc endpoints: health, providers, format capabilities, config."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import ExportFormat, HealthOut, ProviderInfo, ProvidersOut
from ..services.providers.base import registry
from ..services import housekeeping as housekeeping_service
from ..services.providers.lrclib import LrclibProvider
from .. import db

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    database = "ok"
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:  # noqa: BLE001
        database = f"error: {exc}"
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        version=__import__("app").__version__,
        database=database,
        lrclib_url=settings.lrclib_base_url,
    )


@router.get("/health/lrclib")
async def lrclib_health() -> dict[str, object]:
    provider = LrclibProvider()
    reachable = await provider.health()
    return {"provider": "lrclib", "url": settings.lrclib_base_url, "reachable": reachable}


@router.get("/providers", response_model=ProvidersOut)
def list_providers() -> ProvidersOut:
    items: list[ProviderInfo] = []
    for provider in registry.all():
        items.append(
            ProviderInfo(
                name=provider.name,
                kind=provider.kind,
                available=provider.available(),
                detail=provider.description if provider.available() else provider.unavailable_detail(),
            )
        )
    return ProvidersOut(providers=items)


@router.get("/formats", response_model=list[ExportFormat])
def list_formats() -> list[ExportFormat]:
    return [
        ExportFormat(
            format="json",
            precision_ms=1,
            notes="Canonical format. Float seconds, millisecond precision, per-word start/end.",
        ),
        ExportFormat(
            format="lrc",
            precision_ms=10,
            notes="Centisecond timestamps [mm:ss.xx]. Word timing uses Enhanced LRC <mm:ss.xx> tags; "
            "words without timing are concatenated.",
        ),
        ExportFormat(
            format="ass",
            precision_ms=10,
            notes="H:MM:SS.cc centisecond timestamps. Karaoke \\k tags are quantized to centiseconds.",
        ),
    ]


@router.get("/storage")
def storage() -> dict[str, object]:
    """What the caches occupy and what the limits are (see services/housekeeping)."""
    return housekeeping_service.usage()


@router.post("/storage/cleanup")
def cleanup() -> dict[str, object]:
    """Sweep the caches now; returns what was removed. Safe to call any time."""
    return housekeeping_service.run()


@router.get("/config")
def get_config() -> dict[str, object]:
    """Safe, non-secret runtime configuration for the UI."""
    return {
        "max_upload_mb": settings.max_upload_mb,
        "auto_search": settings.auto_search,
        "auto_select_threshold": settings.auto_select_threshold,
        "metadata_extractor": settings.metadata_extractor,
        # Whether the ASR service's own forced aligner runs by default (off).
        "asr_align": settings.asr_align,
        # Whether a new project looks for 逐字 lyrics in other editions (off).
        "word_sync_search": settings.word_sync_search,
        # Whether a new project (and the one-shot API by default) looks the song
        # up in the lyric sources when no lyrics were pasted (on).
        "fetch_lyrics": settings.fetch_lyrics,
        # Whether a run with no lyrics asks the LLM where the lines break (on),
        # and the per-language ceiling it then uses (Chinese 12, Japanese 18).
        "line_split": settings.line_split,
        "line_split_max_chars": dict(settings.line_split_max_chars),
        # Whether the forced aligner is configured; without it there are no
        # per-character times from the aligner path.
        "char_aligner": bool(settings.char_aligner_url),
        # Cache limits and the sweep interval (see GET /api/storage).
        "housekeeping_interval_min": settings.housekeeping_interval_min,
        "lrclib_url": settings.lrclib_base_url,
        "providers": {
            "asr": settings.asr_provider,
            "aligner": settings.aligner_provider,
            "separator": settings.separator_provider,
            "corrector": settings.corrector_provider,
            "importer": settings.importer_provider,
        },
    }
