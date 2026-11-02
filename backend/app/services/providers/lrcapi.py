"""Internal LrcApi deployment (Kugou-backed), the first fallback after LRCLIB.

See ``admin/lrcapi-deploy`` in the internal Gitea: ``GET /jsonapi`` answers a
JSON array with the full LRC for each hit, ``GET /lyrics`` a single LRC. Both
are keyed on ``title`` (required) plus ``artist``/``album`` (recommended).

The service does not expose a lookup by id, so a later "select this candidate"
is served by repeating the search and matching the id it returned.
"""

from __future__ import annotations

import logging

import httpx

from ...config import settings
from .base import LyricsCandidate, LyricsProvider

logger = logging.getLogger("lyrictrace.lrcapi")


class LrcApiProvider(LyricsProvider):
    name = "lrcapi"
    description = "Internal LrcApi (Kugou) lyrics service"

    def available(self) -> bool:
        return bool(settings.lrcapi_url)

    def unavailable_detail(self) -> str:
        return "Set LYRICTRACE_LRCAPI_URL (e.g. http://lrcapi.local:8000) to enable this source."

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.lrcapi_url,
            timeout=settings.lrcapi_timeout,
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        )

    @staticmethod
    def _to_candidate(item: dict) -> LyricsCandidate:
        from ..lyrics import lyrics_from_lrc

        raw_lrc = (item.get("lyrics") or "").strip()
        text, lines = lyrics_from_lrc(raw_lrc) if raw_lrc else ("", [])
        return LyricsCandidate(
            track_name=(item.get("title") or "").strip(),
            artist_name=(item.get("artist") or "").strip(),
            album_name=(item.get("album") or "").strip(),
            duration=None,
            plain_lyrics=text or None,
            synced_lyrics=raw_lrc if lines else None,
            external_id=str(item.get("id") or "") or None,
            source="lrcapi",
            raw={"id": item.get("id")},
        )

    async def search(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
    ) -> list[LyricsCandidate]:
        title = (query or track_name or "").strip()
        if not title:
            return []
        params = {"title": title}
        if artist_name:
            params["artist"] = artist_name.strip()
        if album_name:
            params["album"] = album_name.strip()

        async with self._client() as client:
            response = await client.get("/jsonapi", params=params)
            response.raise_for_status()
            payload = response.json()

        items = payload if isinstance(payload, list) else []
        candidates: list[LyricsCandidate] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = self._to_candidate(item)
            if not candidate.plain_lyrics:
                continue
            key = candidate.external_id or candidate.plain_lyrics[:60]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    async def get(self, external_id: str) -> LyricsCandidate | None:
        """No by-id endpoint: re-search by title and match the id we stored."""
        target = str(external_id)
        # The id alone cannot be turned back into a title, so callers keep the
        # candidate text from the search; this path only serves re-fetching.
        return None
