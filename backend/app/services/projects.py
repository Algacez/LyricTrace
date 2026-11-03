"""Project serialization and timeline state helpers."""

from __future__ import annotations

from typing import Any

from ..db import now_iso
from ..schemas import (
    CandidateOut,
    Line,
    ProjectOut,
    ProjectSummary,
    SongMetadata,
    Timeline,
    TimelineStatus,
    Word,
)
from .reference import matched_reference


def compute_timeline_status(timeline: Timeline | None) -> TimelineStatus:
    if timeline is None or not timeline.lines:
        return "none"
    timed_lines = [line for line in timeline.lines if line.start is not None]
    if not timed_lines:
        return "untimed"
    worded = [line for line in timed_lines if line.words and any(w.start is not None for w in line.words)]
    if not worded:
        return "line_only"
    complete = all(
        line.words and all(word.start is not None and word.end is not None for word in line.words)
        for line in timed_lines
    )
    if complete and len(worded) == len(timed_lines):
        return "complete"
    return "partial"


def timeline_from_lines(lines: list[Line]) -> Timeline:
    timeline = Timeline(status="none", lines=lines, updated_at=now_iso())
    timeline.status = compute_timeline_status(timeline)
    return timeline


def build_timeline_from_lrc(synced: str | None, plain: str | None) -> Timeline | None:
    from .lyrics import parse_lrc, parse_plain_lyrics

    if synced:
        parsed = parse_lrc(synced)
        if parsed.lines:
            return timeline_from_lines(parsed.lines)
    if plain:
        return timeline_from_lines(parse_plain_lyrics(plain))
    return None


def timeline_to_dict(timeline: Timeline | None) -> dict[str, Any] | None:
    return timeline.model_dump() if timeline else None


def serialize_project(row: dict[str, Any]) -> ProjectOut:
    timeline = Timeline(**row["timeline"]) if row.get("timeline") else None
    metadata = SongMetadata(
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        duration=row["duration"],
        source=row.get("metadata_source") or "manual",
    )
    base = dict(
        id=row["id"],
        title=row["title"],
        artist=row["artist"],
        album=row["album"],
        duration=row["duration"],
        status=row["status"],
        progress=row["progress"],
        error=row["error"],
        metadata=metadata,
        lyrics_source=row.get("lyrics_source"),
        timeline_status=timeline.status if timeline else "none",
        audio_url=f"/api/projects/{row['id']}/audio",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
    return ProjectOut(
        **base,
        original_filename=row.get("original_filename", ""),
        audio_filename=row.get("audio_filename", ""),
        audio_size=row.get("audio_size", 0),
        audio_mime=row.get("audio_mime", ""),
        metadata_source=row.get("metadata_source") or "manual",
        search_query=row.get("search_query"),
        selected_candidate_id=row.get("selected_candidate_id"),
        original_lyrics=row.get("original_lyrics") or "",
        revised_lyrics=row.get("revised_lyrics") or "",
        lyrics_lrclib_id=row.get("lyrics_lrclib_id"),
        lyrics_external_id=row.get("lyrics_external_id"),
        lyrics_language=row.get("lyrics_language"),
        language_origin=row.get("language_origin"),
        separate_vocals=True if row.get("separate_vocals") is None else bool(row.get("separate_vocals")),
        asr_align=False if row.get("asr_align") is None else bool(row.get("asr_align")),
        hfa_timing=False if row.get("hfa_timing") is None else bool(row.get("hfa_timing")),
        hfa_chars=False if row.get("hfa_chars") is None else bool(row.get("hfa_chars")),
        detect_extras=False if row.get("detect_extras") is None else bool(row.get("detect_extras")),
        llm_merge=False if row.get("llm_merge") is None else bool(row.get("llm_merge")),
        word_sync_search=(
            False if row.get("word_sync_search") is None else bool(row.get("word_sync_search"))
        ),
        fetch_lyrics=(
            True if row.get("fetch_lyrics") is None else bool(row.get("fetch_lyrics"))
        ),
        line_split=(
            True if row.get("line_split") is None else bool(row.get("line_split"))
        ),
        line_split_max_chars=(
            None if row.get("line_split_max_chars") is None else int(row["line_split_max_chars"])
        ),
        lyrics_reference=bool((row.get("lyrics_synced") or "").strip()),
        lyrics_reference_lrc=(row.get("lyrics_synced") or None),
        lyrics_word_sync=bool((row.get("lyrics_word_sync") or "").strip()),
        lyrics_word_sync_edition=row.get("lyrics_word_sync_edition"),
        reference_lines=matched_reference(
            row.get("original_lyrics") or "", row.get("lyrics_synced")
        ),
        timeline=timeline,
    )


def serialize_project_summary(row: dict[str, Any]) -> ProjectSummary:
    full = serialize_project(row)
    return ProjectSummary(**full.model_dump(include=set(ProjectSummary.model_fields)))


def serialize_candidate(row: dict[str, Any]) -> CandidateOut:
    return CandidateOut(
        id=row["id"],
        project_id=row["project_id"],
        lrclib_id=row["lrclib_id"],
        track_name=row["track_name"],
        artist_name=row["artist_name"],
        album_name=row["album_name"],
        duration=row["duration"],
        instrumental=bool(row["instrumental"]),
        score=row["score"],
        selected=bool(row["selected"]),
        has_synced_lyrics=bool(row.get("synced_lyrics")),
        has_plain_lyrics=bool(row.get("plain_lyrics")),
        synced_lyrics=row.get("synced_lyrics"),
        plain_lyrics=row.get("plain_lyrics"),
    )


def ensure_words(line: Line) -> list[Word]:
    return line.words
