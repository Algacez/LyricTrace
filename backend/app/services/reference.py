"""LRCLIB's own timeline, used only as a *prior* for the aligner.

The API entry often carries an LRC timeline from the studio recording. A live
recording is not the studio recording, so those stamps must never become output
timings. They are still useful in one specific way: they say roughly *when* a
given line is expected to be sung, which lets the aligner recover a line the
global monotonic DP failed to place (a live version can reorder or repeat a
verse, and the DP then runs out of monotonic room).

Everything here is best-effort: a reference that does not match the lyric lines
is simply not used.
"""

from __future__ import annotations

import re

from ..schemas import Line, ReferenceLine
from .lyrics import parse_lrc

PUNCT_RE = re.compile(r"[\s　。、，！？!?,.…~〜ー「」『』（）()\[\]【】・\-—–/]+")


def normalize_text(text: str) -> str:
    return PUNCT_RE.sub("", text).lower()


def reference_lines(synced: str | None) -> list[Line]:
    """Timed lines from an LRC blob, or ``[]`` when there is no usable timeline."""
    if not synced or "[" not in synced:
        return []
    parsed = parse_lrc(synced)
    return [line for line in parsed.lines if line.text.strip() and line.start is not None]


def lyric_texts(text: str) -> list[str]:
    """The lyric's own lines, LRC timestamps stripped (same view the aligner has)."""
    if not text or not text.strip():
        return []
    if "[" in text:
        parsed = parse_lrc(text)
        if parsed.lines:
            return [line.text for line in parsed.lines if line.text.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def matched_reference(text: str, synced: str | None) -> list[ReferenceLine]:
    """The source timeline, line by line, aligned onto the displayed lyrics.

    Unmatched reference lines are dropped (a different lyric version cannot be
    shown next to ours), and lyric lines with no reference entry get no time.
    """
    reference = reference_lines(synced)
    texts = lyric_texts(text)
    if not reference or not texts:
        return []
    matched = match_reference(texts, reference)
    return [
        ReferenceLine(index=index, text=ref.text, start=ref.start, end=ref.end)
        for index, ref in sorted(matched.items())
    ]


def match_reference(texts: list[str], reference: list[Line]) -> dict[int, Line]:
    """Map lyric line index -> reference line, by text first, then by position.

    Text matching is exact after punctuation folding, which is what the LRCLIB
    plain and synced versions of the same entry satisfy. When the user edited
    the lyrics, identical line counts still let the positions line up; anything
    else is discarded rather than guessed.
    """
    if not reference or not texts:
        return {}
    keys = [normalize_text(text) for text in texts]
    used: set[int] = set()
    matched: dict[int, Line] = {}
    for index, key in enumerate(keys):
        if not key:
            continue
        for position, ref in enumerate(reference):
            if position in used:
                continue
            if normalize_text(ref.text) == key:
                matched[index] = ref
                used.add(position)
                break

    if len(matched) * 2 < len(texts) and len(reference) == len(texts):
        # Edited lyrics: same number of lines, so positions are still meaningful.
        return {index: reference[index] for index in range(len(texts))}
    return matched
