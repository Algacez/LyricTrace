"""Group ASR tokens into lyric-like lines when the caller supplied no lyrics.

The one-shot ``/api/align`` endpoint can be called with audio only. There is no
lyric text to align against then, so the honest output is what the recogniser
actually heard: its own tokens, split into lines at pauses. Nothing is invented
and nothing is rewritten — the text is the ASR transcript.
"""

from __future__ import annotations

import re

from ..schemas import Line, Word

# A gap this long between two tokens is a line break: a breath, not a syllable.
LINE_GAP_SEC = 0.9
# Hard cap so an unbroken rap verse does not become one unreadable line.
MAX_LINE_CHARS = 24

SPACE_RE = re.compile(r"\s+")

# The ASR's forced aligner occasionally stamps a token with start == end (131 of
# 386 tokens on the test song). A zero-width token makes an invisible karaoke
# chip and an LRC word tag that can never light up, so it gets a small width.
MIN_WORD_SEC = 0.06


def repair_word_times(words: list[Word]) -> list[Word]:
    """Give zero/negative-duration tokens a usable width.

    Only the *end* is adjusted, and never past the next token's start: the ASR's
    starts are the one thing we must not move, since they are the timing source
    of truth.
    """
    out = [word.model_copy() for word in words]
    for index, word in enumerate(out):
        if word.start is None:
            continue
        next_start = out[index + 1].start if index + 1 < len(out) else None
        end = word.end if word.end is not None else word.start
        if end <= word.start:
            end = next_start if next_start is not None and next_start > word.start else word.start + MIN_WORD_SEC
        word.end = end
    return out


def group_words(
    words: list[Word], *, gap: float = LINE_GAP_SEC, max_chars: int = MAX_LINE_CHARS
) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    length = 0
    for word in words:
        if word.start is None or word.end is None:
            continue
        if current:
            previous_end = current[-1].end or word.start
            if word.start - previous_end > gap or length >= max_chars:
                groups.append(current)
                current, length = [], 0
        current.append(word)
        length += len(word.text.strip())
    if current:
        groups.append(current)
    return groups


def lines_from_groups(groups: list[list[Word]]) -> list[Line]:
    """``lines_from_words`` for a grouping that was decided some other way.

    The recogniser's pauses are not always where a sung line ends, so the
    grouping can come from the LLM instead (see ``line_split``). The text and
    every time are still the recogniser's: this only turns a grouping into
    lines, exactly as ``lines_from_words`` does for the pause-based one.
    """
    lines: list[Line] = []
    for group in groups:
        text = SPACE_RE.sub(" ", "".join(word.text for word in group)).strip()
        if not text:
            continue
        timed = [
            Word(text=word.text, start=word.start, end=word.end, source="asr") for word in group
        ]
        lines.append(
            Line(
                text=text,
                start=timed[0].start,
                end=timed[-1].end,
                words=timed,
                source="asr",
                note="未提供歌词，这一行来自识别文本",
            )
        )
    return lines


def lines_from_words(words: list[Word]) -> list[Line]:
    return lines_from_groups(group_words(words))
