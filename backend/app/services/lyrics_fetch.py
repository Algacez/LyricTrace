"""Original lyrics, looked up in a chain of sources.

Order matters and mirrors what the user asked for: NetEase Cloud Music first,
because it is the only source with a large word-level (逐字, ``yrc``) catalogue —
LRCLIB's ``lyricsfile`` covers very little. Then LRCLIB, then the internal LrcApi
deployment (Kugou). A source that answers with nothing usable falls through to
the next one; the first source that produces a confident match wins.

Every source is searched several ways before giving up — ``track_name`` +
``artist_name``, then the track name alone, then a free-text ``q`` — because the
artist filter is an exact-ish match and a partially-wrong artist (``泠鸢`` vs
``泠鸢yousa``) hides an entry that exists.

Every source only supplies *text* by default; a synced timeline, when present,
is kept as a positional hint for the aligner and never copied into the output.

The exception is word-level (逐字) data — LRCLIB's ``lyricsfile`` and NetEase's
``yrc``. That is kept verbatim on the candidate so :mod:`word_sync` can check it
against the ASR and, only when the two agree, use it re-anchored to this audio.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ..config import settings
from ..schemas import Line
from .furigana import strip_furigana_text
from .lyrics import lines_to_text, parse_lrc
from .matching import score_candidate, similarity
from .providers.base import LyricsCandidate
from .providers.lrcapi import LrcApiProvider
from .providers.lrclib import LrclibProvider
from .providers.netease import NeteaseProvider

logger = logging.getLogger("lyrictrace.lyrics_fetch")

MAX_CANDIDATES = 8
PREVIEW_CHARS = 160

#: Provider order. ``lyrics_source`` in the database records which one answered.
#: NetEase first: it is the only source with a large word-level (逐字, ``yrc``)
#: catalogue, and a song's other editions (live, remix) each carry their own
#: ``yrc`` — LRCLIB has almost none. LRCLIB and LrcApi stay as fallbacks for
#: songs NetEase does not have.
SOURCE_ORDER = ("netease", "lrclib", "lrcapi")

#: A candidate whose artist is this unlike the project's is a different song —
#: unless it is the *original* of a cover (see ``_plausible``).
ARTIST_MIN_SIMILARITY = 0.5

#: A cover is credited to whoever sang it, while the lyrics of the song it
#: covers belong to the original entry (「【秋葵AKIAOI】怪物（TV size）」 must find
#: YOASOBI's 怪物). For such a project a different artist is tolerated when the
#: title says it is the same song *and* the running time lines up — that pair,
#: not the artist string, is what identifies a recording.
COVER_TITLE_MIN_SIMILARITY = 0.85

#: How far a cover may differ from the entry it points at and still be the same
#: recording: a TV-size cover of a 90 s clip is not the 206 s full version, and
#: the full version's lyrics would carry verses this audio never sings.
COVER_DURATION_TOLERANCE = 5.0
#: The same tolerance, but one-sided: a recording *longer* than the entry is
#: accepted however much longer. A live cut carries talk, an intro and applause
#: around the same song (「勾指起誓」 331 s live against the 184 s studio entry),
#: and those extra seconds cannot introduce lyrics the entry does not have — the
#: worst case is an entry that holds fewer lines than were sung, which the
#: aligner shows as unmapped sung content. The other direction is the harmful
#: one and stays at ``COVER_DURATION_TOLERANCE``: taking the full version's
#: lyrics for a 90 s TV-size cut puts verses in the timeline that were never sung.

#: Human-readable names, used in notes and progress messages.
SOURCE_LABEL = {"lrclib": "LRCLIB", "lrcapi": "LrcApi", "netease": "网易云"}

#: Share of our lyric lines that must appear verbatim in another edition for it
#: to count as the same song worth taking 逐字 timings from. A live cut repeats
#: the studio lyrics almost exactly; a same-titled different song does not.
WORD_SYNC_MIN_AGREEMENT = 0.6

#: How much score a word-timed entry may give up to be taken as the project's
#: lyrics. A 逐字 edition is worth having, but not at the price of a different
#: edit's text (a live cut carries its MC section); beyond this margin the best
#: text wins and the 逐字 timings are fetched from another edition at align time.
WORD_SYNC_SCORE_MARGIN = 0.05


@dataclass
class LyricsMatch:
    candidate: LyricsCandidate
    score: float
    lyrics: str
    agreement: float = 0.0
    """For a 逐字 search: share of *our* lyric lines found in this edition."""

    @property
    def edition_label(self) -> str:
        """Which entry this is, for notes: the studio cut and the live cut of
        the same song are different entries with different timings."""
        name = self.candidate.track_name or "（未命名）"
        artist = self.candidate.artist_name
        return f"{name} — {artist}" if artist else name

    @property
    def source(self) -> str:
        return self.candidate.source

    @property
    def source_label(self) -> str:
        return SOURCE_LABEL.get(self.source, self.source)

    @property
    def lrclib_id(self) -> int | None:
        return self.candidate.lrclib_id if self.source == "lrclib" else None

    @property
    def external_id(self) -> str | None:
        return self.candidate.external_id

    @property
    def synced(self) -> str:
        """The source's own LRC timeline, kept as a *reference* for the LLM.

        It is never copied into the output: timings come from the ASR. It only
        tells the aligner roughly where each line is expected to land.
        """
        return (self.candidate.synced_lyrics or "").strip()

    @property
    def has_word_sync(self) -> bool:
        return bool((self.candidate.word_sync_raw or "").strip())

    def word_sync_lines(self) -> list[Line]:
        """The source's word-level (逐字) timings, or ``[]`` when it has none.

        LRCLIB ships a ``lyricsfile``, NetEase a ``yrc`` blob. Both describe
        *some* recording of the song; :mod:`app.services.word_sync` decides
        whether they may be used for this audio.
        """
        from . import word_sync

        raw = self.candidate.word_sync_raw
        if not raw:
            return []
        if self.candidate.word_sync_format == "yrc":
            return word_sync.parse_yrc(raw)
        return word_sync.parse_lyricsfile(raw)

    def preview(self) -> str:
        head = self.lyrics[:PREVIEW_CHARS].replace("\n", " / ")
        return head + ("…" if len(self.lyrics) > PREVIEW_CHARS else "")


def candidate_lyrics(candidate: LyricsCandidate) -> str:
    """Plain lyric text for a candidate; LRC stamps are stripped.

    ``plainLyrics`` is preferred because it is already clean; entries that only
    carry ``syncedLyrics`` are reduced to their text so no API timing can leak
    into the alignment.
    """
    plain = (candidate.plain_lyrics or "").strip()
    if plain:
        return strip_furigana_text(plain)
    synced = (candidate.synced_lyrics or "").strip()
    if not synced:
        return ""
    parsed = parse_lrc(synced)
    if parsed.lines:
        return lines_to_text(parsed.lines)
    return strip_furigana_text(synced)


def _plausible(candidate: LyricsCandidate, fields: dict[str, Any]) -> bool:
    """Reject entries that are plainly a different recording.

    Score alone does not catch this: a same-titled song by someone else still
    scores ~0.65 on title alone. Falling through to the next source is better
    than handing the aligner a completely different song's words.

    A *cover* (翻唱) is the one case where the artist is expected to differ: the
    project is credited to whoever sang it, and the lyrics are the original's.
    There the title and the running time decide instead — see
    :func:`_same_recording_by_title`.
    """
    artist = fields.get("artist") or ""
    if not artist or not candidate.artist_name:
        return True
    if similarity(artist, candidate.artist_name) >= ARTIST_MIN_SIMILARITY:
        return True
    return _same_recording_by_title(candidate, fields)


def _same_recording_by_title(candidate: LyricsCandidate, fields: dict[str, Any]) -> bool:
    """Same song under another artist: a cover, or the original of one.

    「【秋葵AKIAOI】怪物（TV size）」 is a cover, so its artist matches no entry —
    while YOASOBI's 怪物 with the wrong running time is a *different* recording
    of a same-titled song. Title plus duration is what names a recording;
    without a running time on one side there is nothing but the title, and that
    is not enough to overrule the artist.
    """
    title = fields.get("title") or ""
    if not title or not candidate.track_name:
        return False
    if similarity(title, candidate.track_name) < COVER_TITLE_MIN_SIMILARITY:
        return False
    duration = fields.get("duration")
    if not duration or not candidate.duration:
        return False
    # Longer than the entry is allowed without limit (live cuts add talk and
    # applause around the same song); shorter has to line up, or it is a
    # different edit whose missing verses would be filled in from this entry.
    if float(duration) >= float(candidate.duration):
        return True
    return float(candidate.duration) - float(duration) <= COVER_DURATION_TOLERANCE


def rank_candidates(
    candidates: list[LyricsCandidate],
    *,
    title: str = "",
    artist: str = "",
    album: str = "",
    duration: float | None = None,
    require_word_sync: bool = False,
    limit: int = MAX_CANDIDATES,
) -> list[LyricsMatch]:
    fields = {"title": title, "artist": artist, "duration": duration}
    matches: list[LyricsMatch] = []
    for candidate in candidates:
        if require_word_sync and not (candidate.word_sync_raw or "").strip():
            continue
        if not _plausible(candidate, fields):
            continue
        text = candidate_lyrics(candidate)
        if not text:
            continue
        score = score_candidate(
            candidate, title=title, artist=artist, album=album, duration=duration
        )
        matches.append(LyricsMatch(candidate=candidate, score=score, lyrics=text))
    matches.sort(key=lambda match: (match.score, len(match.lyrics)), reverse=True)
    return matches[:limit]


def _project_fields(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": (project.get("title") or "").strip(),
        "artist": (project.get("artist") or "").strip(),
        "album": (project.get("album") or "").strip(),
        "duration": project.get("duration"),
    }


def _provider(source: str):
    if source == "lrclib":
        return LrclibProvider()
    if source == "lrcapi":
        return LrcApiProvider()
    if source == "netease":
        return NeteaseProvider()
    raise ValueError(f"Unknown lyrics source '{source}'")


# --------------------------------------------------------------------------
# Per-source search
# --------------------------------------------------------------------------


async def _search_lrclib(fields: dict[str, Any], query: str | None) -> list[LyricsCandidate]:
    provider = LrclibProvider()
    if query:
        return await provider.search(query=query)
    title, artist = fields["title"], fields["artist"]
    attempts: list[dict[str, str]] = []
    if title and artist:
        attempts.append({"track_name": title, "artist_name": artist})
    if title:
        # The artist filter is strict; a near-miss artist (泠鸢 vs 泠鸢yousa)
        # would hide an entry that does exist.
        attempts.append({"track_name": title})
        if artist:
            attempts.append({"query": f"{title} {artist}"})
    if not attempts and artist:
        attempts.append({"query": artist})
    for params in attempts:
        try:
            results = await provider.search(**params)
        except Exception as exc:  # noqa: BLE001 - try the next query shape
            logger.warning("LRCLIB search %s failed: %s", params, exc)
            continue
        if results:
            return results
    return []


async def _search_provider(source: str, fields: dict[str, Any], query: str | None) -> list[LyricsCandidate]:
    if source == "lrclib":
        return await _search_lrclib(fields, query)
    provider = _provider(source)
    if not provider.available():
        return []
    if query:
        return await provider.search(query=query)
    title, artist, album = fields["title"], fields["artist"], fields["album"]
    collected: list[LyricsCandidate] = []
    seen: set[str] = set()

    async def run(params: dict[str, str | None]) -> list[LyricsCandidate]:
        try:
            results = await provider.search(**params)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - try the next query shape
            logger.warning("%s search %s failed: %s", source, params, exc)
            return []
        fresh: list[LyricsCandidate] = []
        for candidate in results:
            key = f"{candidate.external_id or ''}:{candidate.track_name}:{candidate.artist_name}"
            if key in seen:
                continue
            seen.add(key)
            fresh.append(candidate)
        collected.extend(fresh)
        return fresh

    def confident(items: list[LyricsCandidate]) -> bool:
        return any(
            _plausible(item, fields)
            and score_candidate(
                item,
                title=title,
                artist=artist,
                album=album,
                duration=fields["duration"],
            )
            >= settings.auto_select_threshold
            for item in items
        )

    if title:
        await run({"track_name": title, "artist_name": artist or None, "album_name": album or None})
        # A cover is credited to its own singer, while the song's lyrics are
        # filed under the original — searching under the cover credit alone
        # never reaches them, and it *does* reach the singer's own unrelated
        # songs, which score too low to use. Only then search by title as well:
        # the normal case stays one round trip.
        if artist and not confident(collected):
            await run({"track_name": title, "artist_name": None, "album_name": None})
    else:
        await run({"track_name": None, "artist_name": artist or None, "album_name": None})
    return collected


async def search(
    project: dict[str, Any],
    *,
    query: str | None = None,
    sources: tuple[str, ...] = SOURCE_ORDER,
) -> list[LyricsMatch]:
    """Search the sources in order; the first one with a usable hit wins.

    Within a source, the top-ranked entry is not necessarily the useful one: the
    studio cut of a song usually scores highest and often has no word-level
    (逐字) lyrics, while the live cut — further down the same result list — has a
    ``yrc``. So the *best-ranked entry that carries word timings* wins, even if
    it is not the top hit; a source whose whole result list is text-only falls
    through to the next source, and the best text hit overall is kept as the
    fallback.
    """
    fields = _project_fields(project)
    if not (query and query.strip()) and not (fields["title"] or fields["artist"]):
        return []
    fallback: list[LyricsMatch] = []
    for source in sources:
        try:
            candidates = await _search_provider(source, fields, query)
        except Exception as exc:  # noqa: BLE001 - a dead source falls through
            logger.warning("%s lyrics search failed: %s", source, exc)
            continue
        # Keep the whole ranked list here: a word-timed edition may sit below
        # the top few, and the search is wide precisely to find it.
        matches = rank_candidates(candidates, limit=len(candidates) or 1, **fields)
        if not matches:
            continue
        confident = [match for match in matches if match.score >= settings.auto_select_threshold]
        if not confident:
            # Weak hits (a same-titled song by someone else) are worse than
            # asking the next source, but they are still offered to a human.
            logger.info(
                "%s best match for %r scored %.2f, trying the next source",
                source,
                fields["title"],
                matches[0].score,
            )
            continue
        synced = [match for match in confident if match.has_word_sync]
        # A word-timed entry is only taken as *the lyrics* when it is about as
        # good a match as the top hit. The live cut of a song carries its MC
        # section, which is not lyrics at all; there the text comes from the
        # best entry and :func:`find_word_sync` fetches the 逐字 timings from
        # the live edition at align time instead.
        synced = [match for match in synced if match.score >= confident[0].score - WORD_SYNC_SCORE_MARGIN]
        if synced:
            if synced[0] is not confident[0]:
                logger.info(
                    "%s: %r ranked %.2f without word timings, taking %r (%.2f) which has them",
                    source,
                    confident[0].edition_label,
                    confident[0].score,
                    synced[0].edition_label,
                    synced[0].score,
                )
            # The confident word-timed entries first (best-ranked among them),
            # then everything else in ranked order — the manual picker still
            # shows the whole list, and a weak hit never outranks a good one.
            rest = [match for match in matches if not any(match is item for item in synced)]
            return synced + rest
        if not fallback:
            fallback = matches
            logger.info(
                "%s has text for %r but no word-level timings, looking for a source that does",
                source,
                fields["title"],
            )
    return fallback


async def search_all(
    project: dict[str, Any], *, query: str | None = None
) -> list[LyricsMatch]:
    """Every source that answers, for the manual search panel.

    Unlike :func:`search` this does not stop at the first confident hit, so the
    user can pick a different edition (studio vs live, another translation).
    """
    fields = _project_fields(project)
    if not (query and query.strip()) and not (fields["title"] or fields["artist"]):
        return []
    collected: list[LyricsMatch] = []
    seen: set[str] = set()
    for source in SOURCE_ORDER:
        try:
            candidates = await _search_provider(source, fields, query)
        except Exception as exc:  # noqa: BLE001 - one dead source is not fatal
            logger.warning("%s lyrics search failed: %s", source, exc)
            continue
        for match in rank_candidates(candidates, **fields):
            key = f"{match.source}:{match.external_id or match.lyrics[:40]}"
            if key in seen:
                continue
            seen.add(key)
            collected.append(match)
    # Word-timed entries first, then by score: the user is picking an edition to
    # align against, and an edition with 逐字 lyrics is worth more than a
    # slightly better title match without them.
    collected.sort(key=lambda match: (match.has_word_sync, match.score), reverse=True)
    return collected[:MAX_CANDIDATES]


async def find_word_sync(
    project: dict[str, Any],
    lines: list[Line],
    *,
    exclude: str | None = None,
) -> LyricsMatch | None:
    """Look for 逐字 lyrics in *other editions* of the same song.

    The entry the project's lyrics came from is often the studio cut, and a
    large part of the catalogue has no ``yrc`` at all — while the live edition
    of the same song does (it is a different NetEase entry, with its own
    ``yrc``). The user's instruction: when walking the lyrics, search widely,
    including other versions of the same song, for word-level timings.

    A match is only a *candidate*: :func:`app.services.word_sync.apply` still
    has to find the same lines at the same places in this audio (allowing for
    one overall offset), so a live edition 35 s behind the studio one is fine
    and a different edit is rejected.

    ``agreement`` says how much of *our* lyric text the edition shares, which
    is what picks between several word-timed editions.
    """
    fields = _project_fields(project)
    title = fields["title"] or (lines[0].text if lines else "")
    if not title:
        return None
    from .providers.netease import WORD_SYNC_SEARCH_LIMIT
    from .word_sync import fold as _fold

    artist = fields["artist"]
    # Search with the artist first, then on the title alone: the project's
    # artist may be a cover credit (泠鸢 vs 泠鸢yousa) and the live edition is
    # usually credited to whoever sang it.
    attempts: list[str | None] = [artist or None, None] if artist else [None]

    candidates: list[LyricsCandidate] = []
    for source in SOURCE_ORDER:
        provider = _provider(source)
        if not provider.available():
            continue
        for attempt in attempts:
            try:
                if source == "netease":
                    found = await provider.search(  # type: ignore[call-arg]
                        track_name=title, artist_name=attempt, limit=WORD_SYNC_SEARCH_LIMIT
                    )
                else:
                    found = await provider.search(track_name=title, artist_name=attempt)
            except Exception as exc:  # noqa: BLE001 - one dead query is not fatal
                logger.info("%s 逐字 search %r (%s) failed: %s", source, title, attempt, exc)
                continue
            candidates.extend(found)
            # No early stop on a 逐字 hit: the first attempt searches under the
            # project's own artist, and for a cover that artist's catalogue can
            # carry a same-titled 逐字 entry that is a different song. The
            # agreement check below is what picks the right edition anyway.

    keys = [_fold(line.text) for line in lines if line.text.strip()]
    ranked: list[tuple[float, float, LyricsMatch]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not (candidate.word_sync_raw or "").strip():
            continue
        if exclude and candidate.external_id == exclude:
            continue
        key = f"{candidate.source}:{candidate.external_id or candidate.track_name}"
        if key in seen:
            continue
        seen.add(key)
        if not _plausible(
            candidate, {"title": title, "artist": artist, "duration": fields["duration"]}
        ):
            continue
        text = candidate_lyrics(candidate)
        if not text:
            continue
        edition = {_fold(line) for line in text.splitlines() if line.strip()}
        agreement = (sum(1 for key_text in keys if key_text in edition) / len(keys)) if keys else 0.0
        if agreement < WORD_SYNC_MIN_AGREEMENT:
            continue
        score = score_candidate(
            candidate, title=title, artist=artist, album=fields["album"], duration=fields["duration"]
        )
        ranked.append(
            (agreement, score, LyricsMatch(candidate=candidate, score=score, lyrics=text, agreement=agreement))
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = ranked[0][2]
    logger.info(
        "word-level lyrics for %r found in a different edition: %s (%s), %.0f%% of the lines match",
        title,
        best.edition_label,
        best.source,
        best.agreement * 100,
    )
    return best


async def fetch_best(project: dict[str, Any]) -> LyricsMatch | None:
    """The top match across the source chain, if any clears the threshold."""
    matches = await search(project)
    if not matches:
        return None
    best = matches[0]
    if best.score < settings.auto_select_threshold:
        logger.info(
            "best lyrics match for %r scored %.2f, below %.2f",
            project.get("title"),
            best.score,
            settings.auto_select_threshold,
        )
        return None
    return best


async def fetch_by_external(
    source: str, external_id: str, project: dict[str, Any] | None = None
) -> LyricsMatch | None:
    """Re-fetch one entry the user picked from a search result."""
    provider = _provider(source)
    candidate: LyricsCandidate | None = None
    if source == "lrclib":
        candidate = await provider.get(int(external_id))  # type: ignore[attr-defined]
    elif source == "netease":
        candidate = await provider.get(external_id)  # type: ignore[attr-defined]
    else:
        # LrcApi has no by-id endpoint: repeat the search and match the id.
        fields = _project_fields(project or {})
        candidates = await provider.search(
            track_name=fields["title"] or None, artist_name=fields["artist"] or None
        )
        candidate = next((item for item in candidates if item.external_id == external_id), None)
    if candidate is None:
        return None
    text = candidate_lyrics(candidate)
    if not text:
        return None
    return LyricsMatch(candidate=candidate, score=1.0, lyrics=text)


async def fetch_by_id(lrclib_id: int) -> LyricsMatch | None:
    """LRCLIB by id (kept for the existing select endpoint)."""
    return await fetch_by_external("lrclib", str(lrclib_id))


def auto_fetch(project: dict[str, Any]) -> LyricsMatch | None:
    """Blocking, best-effort lookup used by the background pipeline handlers."""
    if not settings.auto_search:
        return None
    try:
        return asyncio.run(fetch_best(project))
    except Exception as exc:  # noqa: BLE001 - a dead source must not fail the task
        logger.warning("lyrics lookup failed for %r: %s", project.get("title"), exc)
        return None
