"""LRCLIB client (https://lrclib.net/docs)."""

from __future__ import annotations

import asyncio
import logging

import httpx

from ...config import settings
from .base import LyricsCandidate, LyricsProvider

logger = logging.getLogger("lyrictrace.lrclib")

RETRIES = 3
"""LRCLIB answers 503 in short bursts; a couple of retries clears it."""
RETRY_BACKOFF = 1.5


class LrclibProvider(LyricsProvider):
    name = "lrclib"
    description = "LRCLIB community lyrics database"

    def available(self) -> bool:
        return True

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.lrclib_base_url,
            timeout=settings.lrclib_timeout,
            headers={"User-Agent": settings.user_agent, "Accept": "application/json"},
        )

    @staticmethod
    def _to_candidate(item: dict) -> LyricsCandidate:
        # ``lyricsfile`` is a YAML document holding ``lines[].words``; it is only
        # word-level when the entry says so (or when a words array is present).
        lyricsfile = item.get("lyricsfile") if isinstance(item.get("lyricsfile"), str) else None
        word_sync = lyricsfile if (item.get("hasWordSync") or "words:" in (lyricsfile or "")) else None
        return LyricsCandidate(
            track_name=(item.get("trackName") or "").strip(),
            artist_name=(item.get("artistName") or "").strip(),
            album_name=(item.get("albumName") or "").strip(),
            duration=float(item["duration"]) if item.get("duration") is not None else None,
            instrumental=bool(item.get("instrumental")),
            plain_lyrics=item.get("plainLyrics"),
            synced_lyrics=item.get("syncedLyrics"),
            word_sync_raw=word_sync,
            word_sync_format="lyricsfile" if word_sync else None,
            lrclib_id=item.get("id"),
            raw={
                key: item.get(key)
                for key in (
                    "id",
                    "trackName",
                    "artistName",
                    "albumName",
                    "duration",
                    "instrumental",
                    "hasWordSync",
                )
            },
        )

    async def search(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
    ) -> list[LyricsCandidate]:
        params: dict[str, str] = {}
        if query:
            params["q"] = query
        else:
            if track_name:
                params["track_name"] = track_name
            if artist_name:
                params["artist_name"] = artist_name
            if album_name:
                params["album_name"] = album_name
        if not params:
            raise ValueError("Provide a query or at least a track name.")

        payload = await self._get_json("/api/search", params)
        items = payload if isinstance(payload, list) else []
        candidates = [self._to_candidate(item) for item in items if isinstance(item, dict)]
        return candidates

    async def get(self, lrclib_id: int) -> LyricsCandidate:
        payload = await self._get_json("/api/get", {"id": lrclib_id})
        return self._to_candidate(payload if isinstance(payload, dict) else {})

    async def _get_json(self, path: str, params: dict) -> object:
        """GET with retries: the public instance 503s while it rebuilds its cache."""
        last: Exception | None = None
        for attempt in range(RETRIES):
            try:
                async with self._client() as client:
                    response = await client.get(path, params=params)
                    if response.status_code == 503 or response.status_code == 429:
                        raise httpx.HTTPStatusError(
                            f"LRCLIB busy ({response.status_code})",
                            request=response.request,
                            response=response,
                        )
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last = exc
                if attempt + 1 < RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
        assert last is not None
        raise last

    async def health(self) -> bool:
        try:
            async with self._client() as client:
                response = await client.get("/api/search", params={"q": "test"})
                return response.status_code == 200
        except httpx.HTTPError:
            return False
