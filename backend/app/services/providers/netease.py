"""NetEase Cloud Music lyrics, the last fallback (search id -> lyric).

See ``admin/netease-lyric-tutorial`` in the internal Gitea. Two unauthenticated
GETs, a browser User-Agent is all that is needed:

    /api/search/get?type=1&s=<title artist>&limit=5   -> songs[] with ids
    /api/song/lyric?os=pc&id=<id>&lv=-1&tv=-1         -> lrc.lyric (LRC text)
    /api/song/lyric/v1?id=<id>&…&yv=0                 -> yrc.lyric (逐字, if any)

The lyric endpoint returns *timed* LRC; that timeline is kept as a positional
hint for the aligner, never as the output timing. ``yrc`` (word-level) is only
present for part of the catalogue — see the ``逐字歌词`` section of the tutorial.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from ...config import settings
from .base import LyricsCandidate, LyricsProvider

logger = logging.getLogger("lyrictrace.netease")

SEARCH_LIMIT = 6
#: The 逐字 pass looks at every edition of the song, so it needs a wider net:
#: the studio cut is usually the best text match and often has no ``yrc``,
#: while a live edition further down the list does.
WORD_SYNC_SEARCH_LIMIT = 20
#: The normal lookup searches wide too (广泛搜索): the same reason — the edition
#: with word-level lyrics is often a few ranks down — but not as wide, because
#: every hit costs two lyric requests.
BROAD_SEARCH_LIMIT = 12
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class NeteaseProvider(LyricsProvider):
    name = "netease"
    description = "NetEase Cloud Music lyric API"

    def available(self) -> bool:
        return bool(settings.netease_url)

    def unavailable_detail(self) -> str:
        return "Set LYRICTRACE_NETEASE_URL (default https://music.163.com) to enable this source."

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.netease_url,
            timeout=settings.netease_timeout,
            headers={"User-Agent": BROWSER_UA, "Accept": "application/json"},
        )

    async def search(
        self,
        *,
        track_name: str | None = None,
        artist_name: str | None = None,
        album_name: str | None = None,
        query: str | None = None,
        limit: int = BROAD_SEARCH_LIMIT,
    ) -> list[LyricsCandidate]:
        keyword = " ".join(part for part in ((query or track_name or "").strip(), (artist_name or "").strip()) if part)
        if not keyword:
            return []

        async with self._client() as client:
            response = await client.get(
                "/api/search/get", params={"type": 1, "s": keyword, "limit": limit}
            )
            response.raise_for_status()
            payload = response.json()

        songs = ((payload or {}).get("result") or {}).get("songs") or []
        usable = [song for song in songs if isinstance(song, dict) and song.get("id") is not None]
        # Both lyric endpoints are independent, and each song needs both: ask
        # for everything at once so a word-level lookup costs one round trip.
        fetched = await asyncio.gather(
            *(asyncio.gather(self._lyric(song["id"]), self._yrc(song["id"])) for song in usable)
        )
        candidates: list[LyricsCandidate] = []
        for song, (lyric, yrc) in zip(usable, fetched):
            if not lyric:
                continue
            from ..lyrics import lyrics_from_lrc

            text, lines = lyrics_from_lrc(lyric)
            if not text.strip():
                continue
            artists = song.get("artists") or song.get("ar") or []
            names = [str(item.get("name") or "") for item in artists if isinstance(item, dict)]
            duration_ms = song.get("duration") or song.get("dt")
            candidates.append(
                LyricsCandidate(
                    track_name=str(song.get("name") or "").strip(),
                    artist_name=" / ".join(name for name in names if name).strip(),
                    album_name=str((song.get("album") or {}).get("name") or "").strip(),
                    duration=float(duration_ms) / 1000.0 if duration_ms else None,
                    plain_lyrics=text,
                    synced_lyrics=lyric if lines else None,
                    word_sync_raw=yrc or None,
                    word_sync_format="yrc" if yrc else None,
                    external_id=str(song["id"]),
                    source="netease",
                    raw={"id": song["id"]},
                )
            )
        return candidates

    async def _yrc(self, song_id: int | str) -> str:
        """Word-level lyrics (``yrc``), which only ``/api/song/lyric/v1`` returns.

        Most of the catalogue has none — that is an empty string, not an error.
        """
        try:
            async with self._client() as client:
                response = await client.get(
                    "/api/song/lyric/v1",
                    params={
                        "id": song_id,
                        "cp": "false",
                        "lv": -1,
                        "kv": 0,
                        "tv": 0,
                        "rv": 0,
                        "yv": 0,
                        "ytv": 0,
                        "yrv": 0,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.info("netease yrc fetch failed for %s: %s", song_id, exc)
            return ""
        lyric = ((payload or {}).get("yrc") or {}).get("lyric") or ""
        # Only the timed lines matter; the JSON credit rows carry no word times.
        return lyric if any(line.startswith("[") for line in lyric.split("\n")) else ""

    async def _lyric(self, song_id: int | str) -> str:
        try:
            async with self._client() as client:
                response = await client.get(
                    "/api/song/lyric", params={"os": "pc", "id": song_id, "lv": -1, "tv": -1}
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("netease lyric fetch failed for %s: %s", song_id, exc)
            return ""
        return ((payload or {}).get("lrc") or {}).get("lyric") or ""

    async def get(self, external_id: str) -> LyricsCandidate | None:
        """Fetch one song by its id (the search step already knows the title)."""
        lyric = await self._lyric(external_id)
        if not lyric.strip():
            return None
        from ..lyrics import lyrics_from_lrc

        text, lines = lyrics_from_lrc(lyric)
        if not text.strip():
            return None
        yrc = await self._yrc(external_id)
        return LyricsCandidate(
            plain_lyrics=text,
            synced_lyrics=lyric if lines else None,
            word_sync_raw=yrc or None,
            word_sync_format="yrc" if yrc else None,
            external_id=str(external_id),
            source="netease",
        )
