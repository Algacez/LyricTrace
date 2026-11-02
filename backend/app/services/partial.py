"""A cover that sings only part of the song: keep the lines it actually sings.

A 翻唱 often takes one verse and the chorus, or is a short / TV-size edit, while
the lyrics source still holds the whole song. Aligning every line then leaves a
long stretch that was never sung: those lines come back untimed and marked
``unmatched``, which reads like a recogniser failure and leaves the user with a
song's worth of rows to delete by hand.

The decision is made from **time**, not from the transcript. A run of lines
nothing heard is absent from this recording when

* the audio has no room for them — the gap between the line before the run and
  the line after it is far shorter than those lines take to sing (measured with
  the seconds-per-character of the lines that *were* heard), or
* that gap holds no recognised speech at all, even though there is room.

A single line the recogniser missed is never dropped: a cover cuts whole verses,
so one missing line inside a sung verse is a recognition miss, and it stays in
the timeline marked for review. Dropping is the lossy direction — a line that
was sung and thrown away is content the user cannot get back — so the rule is
deliberately one-sided and the run has to be long.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Word

#: Shortest run of unheard lines that can be a cut. Covers cut verses; one or
#: two lines the recogniser lost inside a sung verse are not a cut.
MIN_ABSENT_LINES = 3
#: The room between the neighbouring lines must be at least this share of the
#: time the run takes to sing before the run is kept as merely-missed.
ABSENT_ROOM_SHARE = 0.5
#: With room to spare, a window this long (and at least as long as the run
#: needs) holding no recognised speech at all still means those lines were not
#: sung — an instrumental break, not a missed verse.
ABSENT_EMPTY_ROOM_SEC = 1.0
#: Lines that must be heard before any judgement is made about the rest.
MIN_HEARD_LINES = 2
#: Fallback singing speed (seconds per character) when nothing was heard.
FALLBACK_SEC_PER_CHAR = 0.25


@dataclass
class Absent:
    """The lyric lines this recording does not contain."""

    indices: list[int] = field(default_factory=list)
    runs: list[tuple[int, int]] = field(default_factory=list)
    detail: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.indices)


def _speech_between(words: list[Word], covered: list[bool], lo: float, hi: float) -> int:
    """How many recognised tokens inside ``(lo, hi)`` no lyric line accounts for.

    Tokens a lyric line already covers belong to that line, and a line's span
    can end before its last token does (the DP's span is a word range, and a
    boundary word is shared). Counting those would make every window look sung.
    """
    count = 0
    for position, word in enumerate(words):
        if word.start is None:
            continue
        if position < len(covered) and covered[position]:
            continue
        end = word.end if word.end is not None else word.start
        middle = (word.start + end) / 2
        if lo <= middle <= hi:
            count += 1
    return count


def find_absent(
    *,
    heard: list[bool],
    spans: list[tuple[float, float] | None],
    chars: list[int],
    words: list[Word],
    covered: list[bool] | None = None,
) -> Absent:
    """Which lyric lines this recording never sings.

    ``heard[i]`` says whether line *i* was located in the audio, ``spans[i]``
    is its time span when it was, and ``chars[i]`` how many characters it has
    (its reading length). ``covered`` marks the tokens some lyric line already
    accounts for. Nothing here looks at the text: a line is absent because the
    audio around it cannot hold it, not because the transcript disagrees with it.
    """
    total = len(heard)
    if not words:
        # No transcript to measure the audio with: nothing can be decided.
        return Absent()
    usable = [
        index
        for index in range(total)
        if heard[index] and spans[index] is not None and (spans[index][1] or 0) > (spans[index][0] or 0)
    ]
    if len(usable) < MIN_HEARD_LINES:
        # Nothing to measure against: leave every line as it was.
        return Absent()

    sung = sum((spans[index][1] or 0) - (spans[index][0] or 0) for index in usable)  # type: ignore[misc]
    typed = sum(max(1, chars[index]) for index in usable)
    sec_per_char = sung / typed if typed else FALLBACK_SEC_PER_CHAR
    if sec_per_char <= 0:
        sec_per_char = FALLBACK_SEC_PER_CHAR

    starts = [word.start for word in words if word.start is not None]
    audio_start = min(starts) if starts else 0.0
    ends = [word.end for word in words if word.end is not None]
    audio_end = max(ends) if ends else audio_start

    result = Absent()
    position = 0
    while position < total:
        if heard[position]:
            position += 1
            continue
        first = position
        while position + 1 < total and not heard[position + 1]:
            position += 1
        last = position
        position += 1

        count = last - first + 1
        if count < MIN_ABSENT_LINES:
            continue
        previous = next((index for index in range(first - 1, -1, -1) if heard[index]), None)
        following = next((index for index in range(last + 1, total) if heard[index]), None)
        lo = spans[previous][1] if previous is not None and spans[previous] else audio_start  # type: ignore[index]
        hi = spans[following][0] if following is not None and spans[following] else audio_end  # type: ignore[index]
        lo = float(lo if lo is not None else audio_start)
        hi = float(hi if hi is not None else audio_end)
        room = max(0.0, hi - lo)
        needed = sum(max(1, chars[index]) for index in range(first, last + 1)) * sec_per_char
        speech = _speech_between(words, covered or [], lo, hi)
        too_short = room < ABSENT_ROOM_SHARE * needed
        empty = speech == 0 and room >= max(ABSENT_EMPTY_ROOM_SEC, needed)
        if not (too_short or empty):
            continue

        indices = list(range(first, last + 1))
        result.indices.extend(indices)
        result.runs.append((first, last))
        if too_short:
            reason = f"前后之间只有 {room:.1f} 秒，按字长需要约 {needed:.0f} 秒"
        else:
            reason = f"前后 {room:.1f} 秒里没有识别到任何演唱"
        if previous is None:
            reason = f"这段录音从第 {last + 2} 行才开始唱"
        elif following is None:
            reason = f"这段录音到第 {first} 行就唱完了"
        result.detail.append(
            {
                "lines": [first + 1, last + 1],
                "count": count,
                "window": [round(lo, 3), round(hi, 3)],
                "room_s": round(room, 3),
                "needed_s": round(needed, 3),
                "speech": speech,
                "reason": reason,
            }
        )
        result.notes.append(
            f"这段录音只唱了歌词的一部分：第 {first + 1}–{last + 1} 行（共 {count} 行）"
            f"没有对应的演唱（{reason}），已从时间轴去掉"
        )
    return result
