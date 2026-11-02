"""Scoring LRCLIB candidates against the project's metadata."""

from __future__ import annotations

import re
import unicodedata

from .providers.base import LyricsCandidate

NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
FEAT_RE = re.compile(r"\s*[\(\[]?\s*(feat\.?|ft\.?|featuring)\s+[^\)\]]*[\)\]]?", re.IGNORECASE)


def normalize(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = FEAT_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    """0..1 likeness of two names (normalised, token/CJK-bigram based)."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Token overlap (Jaccard-ish) handles partial titles and CJK strings where
    # character-level similarity is more meaningful.
    tokens_a, tokens_b = set(a.split()), set(b.split())
    if tokens_a and tokens_b:
        token_score = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    else:
        token_score = 0.0
    # Substring containment (e.g. "Song (Live)" vs "Song", or an artist name
    # written with and without its suffix: "泠鸢" vs "泠鸢yousa").
    if a in b or b in a:
        return max(token_score, 0.85)
    if " " not in a and " " not in b:
        # CJK/no-space scripts: compare character bigrams.
        grams_a = {a[i : i + 2] for i in range(max(len(a) - 1, 1))}
        grams_b = {b[i : i + 2] for i in range(max(len(b) - 1, 1))}
        if grams_a and grams_b:
            return max(token_score, len(grams_a & grams_b) / len(grams_a | grams_b))
    return token_score


def score_candidate(
    candidate: LyricsCandidate,
    *,
    title: str = "",
    artist: str = "",
    album: str = "",
    duration: float | None = None,
) -> float:
    title_score = similarity(title, candidate.track_name) if title else 0.0
    artist_score = similarity(artist, candidate.artist_name) if artist else 0.0
    album_score = similarity(album, candidate.album_name) if album else 0.0

    duration_score = 0.0
    if duration and candidate.duration:
        # Only an entry that is *longer* than the audio can put lyrics in the
        # timeline that were never sung, so only that direction is penalised on
        # its exact difference. An entry shorter than the audio is either the
        # same recording (a live cut runs minutes of talk and applause around
        # it) or a different song — the length cannot tell those apart, so it
        # gets partial credit rather than a perfect score: a live cut scored as
        # a total mismatch used to hide the right lyrics entirely.
        diff = float(candidate.duration) - float(duration)
        if diff > 0:
            if diff <= 2:
                duration_score = 1.0
            elif diff <= 5:
                duration_score = 0.8
            elif diff <= 10:
                duration_score = 0.5
            elif diff <= 20:
                duration_score = 0.2
        else:
            duration_score = 1.0 if -diff <= 2 else 0.8

    weights: list[tuple[float, float]] = [(title_score, 0.55)]
    if artist:
        weights.append((artist_score, 0.3))
    if duration:
        weights.append((duration_score, 0.15))
    if album:
        weights.append((album_score, 0.1))

    total_weight = sum(weight for _, weight in weights) or 1.0
    score = sum(value * weight for value, weight in weights) / total_weight

    # Prefer entries that actually carry timing data.
    if candidate.synced_lyrics:
        score += 0.05
    elif candidate.plain_lyrics:
        score += 0.0
    else:
        score -= 0.2
    return round(max(0.0, min(1.0, score)), 4)
