"""LRC parsing and export helpers.

LRC uses centisecond timestamps (``[mm:ss.xx]``) and Enhanced LRC adds
per-word tags (``<mm:ss.xx>``). We keep the canonical model in float seconds
and round on export.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schemas import Line, Timeline, Word
from .furigana import strip_furigana, strip_furigana_text, strip_furigana_words

TIME_TAG = r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]"
LINE_TAG_RE = re.compile(rf"^{TIME_TAG}(.*)$")
WORD_TAG_RE = re.compile(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>")
META_RE = re.compile(r"^\[(ti|ar|al|by|offset|length|re|ve):(.*)\]$", re.IGNORECASE)


@dataclass
class ParsedLrc:
    lines: list[Line] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    offset_ms: int = 0
    has_word_times: bool = False


def _fraction_to_seconds(frac: str | None) -> float:
    if not frac:
        return 0.0
    # "5" -> 0.5s, "50" -> 0.50s, "500" -> 0.500s
    return int(frac) / (10 ** len(frac))


def _tag_to_seconds(minutes: str, seconds: str, fraction: str | None) -> float:
    return int(minutes) * 60 + int(seconds) + _fraction_to_seconds(fraction)


def parse_lrc(text: str) -> ParsedLrc:
    """Parse plain LRC or Enhanced LRC (``<mm:ss.xx>`` word tags) into lines.

    Multiple timestamps on one source line produce multiple entries. Line
    ``end`` times are inferred from the next line's start when absent.
    """
    result = ParsedLrc()
    offset_ms = 0
    entries: list[tuple[float, str, list[Word]]] = []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        meta = META_RE.match(line)
        if meta:
            key, value = meta.group(1).lower(), meta.group(2).strip()
            result.metadata[key] = value
            if key == "offset":
                try:
                    offset_ms = int(value)
                except ValueError:
                    pass
            continue

        timestamps: list[float] = []
        rest = line
        while True:
            match = LINE_TAG_RE.match(rest)
            if not match:
                break
            timestamps.append(_tag_to_seconds(match.group(1), match.group(2), match.group(3)))
            rest = match.group(4)
        if not timestamps:
            continue

        words = strip_furigana_words(_split_enhanced_words(rest))
        if words:
            result.has_word_times = True
        lyric = strip_furigana(WORD_TAG_RE.sub("", rest).strip())
        for ts in timestamps:
            entries.append((ts, lyric, [word.model_copy() for word in words]))

    entries.sort(key=lambda item: item[0])
    for index, (start, lyric, words) in enumerate(entries):
        end = entries[index + 1][0] if index + 1 < len(entries) else None
        result.lines.append(Line(text=lyric, start=start, end=end, words=words, source="lrclib"))

    if result.has_word_times:
        for line in result.lines:
            timed = [word for word in line.words if word.start is not None]
            if timed and timed[-1].end is None:
                timed[-1].end = line.end

    if offset_ms:
        _apply_offset(result.lines, offset_ms)
        result.offset_ms = offset_ms
    return result


def _split_enhanced_words(rest: str) -> list[Word]:
    words: list[Word] = []
    matches = list(WORD_TAG_RE.finditer(rest))
    if not matches:
        return words
    lead = rest[: matches[0].start()].strip()
    if lead:
        words.append(Word(text=lead))
    for index, match in enumerate(matches):
        start = _tag_to_seconds(match.group(1), match.group(2), match.group(3))
        next_start = (
            _tag_to_seconds(matches[index + 1].group(1), matches[index + 1].group(2), matches[index + 1].group(3))
            if index + 1 < len(matches)
            else None
        )
        segment_end = matches[index + 1].start() if index + 1 < len(matches) else len(rest)
        segment = rest[match.end():segment_end]
        words.append(Word(text=segment, start=start, end=next_start, source="lrclib"))
    return words


def _apply_offset(lines: list[Line], offset_ms: int) -> None:
    delta = offset_ms / 1000.0
    for line in lines:
        if line.start is not None:
            line.start = max(0.0, line.start + delta)
        if line.end is not None:
            line.end = max(0.0, line.end + delta)
        for word in line.words:
            if word.start is not None:
                word.start = max(0.0, word.start + delta)
            if word.end is not None:
                word.end = max(0.0, word.end + delta)


def parse_plain_lyrics(text: str) -> list[Line]:
    lines: list[Line] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped = strip_furigana(raw_line.strip())
        if not stripped:
            continue
        lines.append(Line(text=stripped, start=None, end=None, words=[], source="lrclib"))
    return lines


def split_lyrics_text(text: str) -> list[str]:
    lines = (strip_furigana(line.strip()) for line in text.replace("\r\n", "\n").split("\n"))
    return [line for line in lines if line]


CREDIT_WORD = (
    r"(?:作词|作曲|编曲|词曲演唱|词曲唱|词曲|词|曲|演唱|制作人|制作|监制|混音母带|混音|母带|"
    r"录音|和声|吉他|贝斯|鼓|键盘|弦乐|出品|发行|策划|统筹|封面|"
    r"lyrics?|music|composer|arranger|producer|mixed|mastered|vocal|guitar|bass|drums?|strings|"
    r"recording|op|sp)"
)
CREDIT_RE = re.compile(
    # "词：…", "混音&母带 : …", "词曲唱：…" — one or more credit words, any of
    # the usual joiners, then a colon.
    rf"^\s*{CREDIT_WORD}(?:\s*[&、/,+·和与]\s*{CREDIT_WORD})*\s*[:：]",
    re.IGNORECASE,
)
"""Studio credits that some sources ship as timed lines (``[00:07.89]词：…``).

They are not sung, so leaving them in would make the aligner hunt for a line
that never exists in the audio. The joiners matter: NetEase writes both
``混音&母带 : …`` and ``词曲唱：…``, and a plain ``混音\\s*:`` pattern misses them.
"""


def strip_credits(lines: list[Line]) -> list[Line]:
    return [line for line in lines if line.text.strip() and not CREDIT_RE.match(line.text)]


def lyrics_from_lrc(raw: str) -> tuple[str, list[Line]]:
    """Parse an LRC from a fallback source into ``(plain text, sung lines)``.

    Credit lines are dropped: they carry timestamps but are never sung, so they
    would become lines the aligner could never place.
    """
    parsed = parse_lrc(raw)
    lines = strip_credits(parsed.lines)
    return lines_to_text(lines), lines


def lines_to_text(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def format_lrc_time(seconds: float | None) -> str:
    """LRC supports centisecond precision; round half-up to 2 decimals."""
    if seconds is None or seconds < 0:
        return "[00:00.00]"
    centis = int(round(seconds * 100))
    minutes, remainder = divmod(centis, 6000)
    secs, cs = divmod(remainder, 100)
    return f"[{minutes:02d}:{secs:02d}.{cs:02d}]"


def format_lrc_word_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "<00:00.00>"
    centis = int(round(seconds * 100))
    minutes, remainder = divmod(centis, 6000)
    secs, cs = divmod(remainder, 100)
    return f"<{minutes:02d}:{secs:02d}.{cs:02d}>"


def export_lrc(timeline: Timeline, *, title: str = "", artist: str = "", album: str = "") -> str:
    header: list[str] = []
    if title:
        header.append(f"[ti:{title}]")
    if artist:
        header.append(f"[ar:{artist}]")
    if album:
        header.append(f"[al:{album}]")
    header.append("[by:LyricTrace]")

    body: list[str] = []
    for line in timeline.lines:
        if line.start is None:
            continue
        stamp = format_lrc_time(line.start)
        if line.words and any(word.start is not None for word in line.words):
            pieces = []
            for word in line.words:
                if word.start is not None:
                    pieces.append(f"{format_lrc_word_time(word.start)}{word.text}")
                else:
                    pieces.append(word.text)
            body.append(stamp + "".join(pieces))
        else:
            body.append(stamp + line.text)
    return "\n".join(header + body) + "\n"


def format_ass_time(seconds: float | None) -> str:
    """ASS supports centisecond precision in the ``H:MM:SS.cc`` format."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    centis = int(round(seconds * 100))
    hours, remainder = divmod(centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def export_ass(
    timeline: Timeline,
    *,
    title: str = "",
    artist: str = "",
    width: int = 1920,
    height: int = 1080,
    karaoke: bool = True,
) -> str:
    header = f"""[Script Info]
; Generated by LyricTrace
Title: {_ass_escape(title)}
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,2,60,60,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    lines = [line for line in timeline.lines if line.start is not None]
    for index, line in enumerate(lines):
        end = line.end
        if end is None and index + 1 < len(lines):
            end = lines[index + 1].start
        if end is None:
            end = (line.start or 0) + 3.0
        if end <= (line.start or 0):
            end = (line.start or 0) + 0.5
        text = _ass_escape(line.text)
        if karaoke and line.words:
            timed = [word for word in line.words if word.start is not None]
            if len(timed) >= 1:
                parts: list[str] = []
                for word_index, word in enumerate(line.words):
                    next_word = line.words[word_index + 1] if word_index + 1 < len(line.words) else None
                    start = word.start if word.start is not None else (line.start or 0)
                    word_end = word.end if word.end is not None else (
                        next_word.start if next_word and next_word.start is not None else end
                    )
                    duration_cs = max(0, int(round(((word_end or end) - start) * 100)))
                    parts.append(f"{{\\k{duration_cs}}}{_ass_escape(word.text)}")
                text = "".join(parts)
        events.append(
            f"Dialogue: 0,{format_ass_time(line.start)},{format_ass_time(end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + ("\n" if events else "")


def export_json(
    timeline: Timeline,
    *,
    project_id: str,
    title: str = "",
    artist: str = "",
    album: str = "",
) -> dict:
    return {
        "project_id": project_id,
        "title": title,
        "artist": artist,
        "album": album,
        "time_unit": "seconds",
        "precision": "milliseconds (JSON float seconds)",
        "timeline_status": timeline.status,
        "notes": list(timeline.notes),
        "lines": [line.model_dump() for line in timeline.lines],
    }
