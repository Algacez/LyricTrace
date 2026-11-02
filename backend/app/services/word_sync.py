"""Word-level (逐字) timings supplied by the lyrics sources.

Two sources can hand us timings finer than one stamp per line:

* **LRCLIB** — an entry whose ``hasWordSync`` is true carries a ``lyricsfile``
  (YAML) with ``lines[].words``; each word has its own ``start_ms``.
* **NetEase** — ``/api/song/lyric/v1`` returns ``yrc.lyric`` for part of its
  catalogue: ``[lineStart,lineDur](wordStart,wordDur,flag)text…`` per line.

Those timings are for *some* recording of the song, not necessarily this audio.
So they are never copied in as-is: the ASR still has to hear the same lines at
roughly the same places, and each line is then re-anchored onto the start the
ASR measured, keeping the source's intra-line word spacing. When the deviation
is too large (a live version, a different edit, a different song entirely) the
whole thing is rejected and the normal local pipeline runs.

This is the only path where a timing that did not come from the ASR is written
into a result, and it is gated: identical line text (≥90 % of lines), an ASR
anchor for most lines, and a small, consistent offset.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ..schemas import Line, Timeline, Word
from .furigana import strip_furigana, strip_furigana_words

logger = logging.getLogger("lyrictrace.word_sync")

#: Share of lyric lines that must have a word-timed counterpart.
MIN_COVERAGE = 0.9
#: Share of lyric lines the ASR must have located, to anchor the source.
MIN_ANCHORS = 0.6
#: Median |source - ASR| line start, *after removing the run's overall offset*,
#: above which the two recordings are not the same edit. A constant offset is
#: normal — a live intro, a longer outro — and each line is anchored on its own
#: ASR start anyway; it is the scatter around that offset that matters.
MAX_MEDIAN_DEVIATION = 4.0
#: ... and no more than this share of lines may deviate more than that.
MAX_DEVIATION = 6.0
MIN_WITHIN_SHARE = 0.8
#: Shortest word width, so a zero-width source word still lights up.
MIN_WORD_SEC = 0.04

#: NetEase yrc line: ``[start,dur](start,dur,flag)text`` repeated.
YRC_LINE_RE = re.compile(r"^\[(\d+),(\d+)\]")
YRC_WORD_RE = re.compile(r"\((\d+),(\d+),-?\d+\)")

SOURCE_LABEL = {"lrclib": "LRCLIB", "lrcapi": "LrcApi", "netease": "网易云"}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_lyricsfile(text: str | None) -> list[Line]:
    """Word-timed lines from an LRCLIB ``lyricsfile`` (YAML).

    Lines without a ``words`` array are kept with line-level times only; the
    caller only trusts the result when at least one line carries words.
    """
    if not text or not text.strip():
        return []
    try:
        import yaml

        document = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - a malformed file is simply no data
        logger.warning("could not parse lyricsfile: %s", exc)
        return []
    if not isinstance(document, dict):
        return []
    metadata = document.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("instrumental"):
        return []

    lines: list[Line] = []
    for item in document.get("lines") or []:
        if not isinstance(item, dict):
            continue
        raw_words = item.get("words") or []
        words: list[Word] = []
        for raw in raw_words:
            if not isinstance(raw, dict):
                continue
            word_text = str(raw.get("text") or "")
            if not word_text:
                continue
            start = _ms(raw.get("start_ms"))
            end = _ms(raw.get("end_ms"))
            words.append(Word(text=word_text, start=start, end=end, source="word-sync"))
        words = strip_furigana_words(words)
        line_start = _ms(item.get("start_ms"))
        line_end = _ms(item.get("end_ms"))
        text_value = strip_furigana(str(item.get("text") or "").strip()) or "".join(
            w.text for w in words
        ).strip()
        if not text_value:
            continue
        _fill_times(words, line_start, line_end)
        lines.append(
            Line(
                text=text_value,
                start=line_start if line_start is not None else (words[0].start if words else None),
                end=line_end if line_end is not None else (words[-1].end if words else None),
                words=words,
                source="word-sync" if words else "reference",
            )
        )
    return lines


def parse_yrc(text: str | None) -> list[Line]:
    """Word-timed lines from a NetEase ``yrc`` blob.

    Credit lines arrive as JSON objects (``{"t":0,"c":[...]}``); they carry no
    word timings and are skipped. Spaces belong to the preceding word and are
    kept verbatim, otherwise mixed CJK/latin lines glue together.
    """
    if not text or not text.strip():
        return []
    lines: list[Line] = []
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("{"):
            continue
        head = YRC_LINE_RE.match(line)
        if not head:
            continue
        line_start = int(head.group(1)) / 1000.0
        line_dur = int(head.group(2)) / 1000.0
        body = line[head.end() :]
        words: list[Word] = []
        markers = list(YRC_WORD_RE.finditer(body))
        lead = body[: markers[0].start()] if markers else body
        if lead.strip():
            # Text before the first marker has no word timing of its own.
            words.append(Word(text=lead, start=line_start, end=None, source="word-sync"))
        for index, match in enumerate(markers):
            end_at = markers[index + 1].start() if index + 1 < len(markers) else len(body)
            chunk = body[match.end() : end_at]
            if not chunk:
                continue
            start = int(match.group(1)) / 1000.0
            duration = int(match.group(2)) / 1000.0
            words.append(
                Word(text=chunk, start=start, end=start + duration if duration > 0 else None, source="word-sync")
            )
        words = strip_furigana_words(words)
        text_value = "".join(word.text for word in words).strip()
        if not text_value:
            continue
        _fill_times(words, line_start, line_start + line_dur if line_dur > 0 else None)
        lines.append(
            Line(
                text=text_value,
                start=line_start,
                end=(line_start + line_dur) if line_dur > 0 else (words[-1].end if words else None),
                words=words,
                source="word-sync",
            )
        )
    return lines


def has_word_times(lines: list[Line]) -> bool:
    return any(line.words for line in lines)


def from_stored(raw: str | None, fmt: str | None) -> list[Line]:
    """Parse the payload as it was stored on the project."""
    if not raw:
        return []
    if fmt == "yrc":
        return parse_yrc(raw)
    return parse_lyricsfile(raw)


def dump_lines(lines: list[Line]) -> str:
    return json.dumps([line.model_dump() for line in lines], ensure_ascii=False)


def load_lines(raw: str | None) -> list[Line]:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        return [Line(**item) for item in payload]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def _ms(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return round(int(value) / 1000.0, 3)
    except (TypeError, ValueError):
        return None


def _fill_times(words: list[Word], line_start: float | None, line_end: float | None) -> None:
    """Give every word a start/end, using the line bounds as the fallback."""
    if not words:
        return
    if line_start is None:
        line_start = words[0].start
    if line_end is None:
        line_end = words[-1].end
    count = len(words)
    fallback_start = line_start if line_start is not None else 0.0
    fallback_end = line_end if line_end is not None else fallback_start + 0.2 * count
    step = (fallback_end - fallback_start) / max(count, 1)
    for index, word in enumerate(words):
        if word.start is None:
            word.start = round(fallback_start + step * index, 3)
        if word.end is None or word.end < word.start:
            word.end = round(fallback_start + step * (index + 1), 3)
    previous: float | None = None
    for word in words:
        if previous is not None and word.start < previous:
            word.start = previous
        if word.end < word.start + MIN_WORD_SEC:
            word.end = word.start + MIN_WORD_SEC
        previous = word.start
        word.start = round(word.start, 3)
        word.end = round(word.end, 3)


# --------------------------------------------------------------------------
# Matching the source's lines to ours
# --------------------------------------------------------------------------


def text_likeness(a: str, b: str) -> float:
    """How alike two lyric texts are, ignoring punctuation and spacing."""
    left, right = _fold(a), _fold(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()


def _fold(text: str) -> str:
    return re.sub(r"[\s　。、，！？!?,.…~〜ー「」『』（）()\[\]【】・\-—–/]+", "", text or "").lower()


def fold(text: str) -> str:
    """Public spelling of :func:`_fold`: text with punctuation and spacing gone."""
    return _fold(text)


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------


@dataclass
class SyncResult:
    """What the word-sync step decided, plus everything the debug page shows."""

    used: bool = False
    reason: str = ""
    timeline: Timeline | None = None
    note: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    extras: list[Line] = field(default_factory=list)
    """Sung content the source lyrics do not cover (live intro, MC, ad-libs)."""
    report: Any = None
    """The line -> ASR span map, so the caller can judge the extras."""


def apply(
    lines: list[Line],
    sync_lines: list[Line],
    asr_words: list[Word],
    *,
    source: str = "",
    edition: str | None = None,
) -> SyncResult:
    """Re-anchor source word timings onto this audio, if they agree with the ASR.

    The gate is deliberately conservative: a wrong word-sync is worse than no
    word-sync, because it looks authoritative. Anything that does not clear it
    falls through to the normal pipeline.
    """
    from .providers.llm_aligner import (
        ABSENT_HEARD_CONFIDENCE,
        AlignReport,
        LineSpan,
        absorb_strays,
        align_spans,
        clamp_overlaps,
        extra_lines,
    )
    from . import partial

    stats: dict[str, Any] = {
        "source": source,
        "source_label": SOURCE_LABEL.get(source, source),
        "edition": edition,
        "available": bool(sync_lines),
        "line_count": len(lines),
        "sync_line_count": len(sync_lines),
        "word_lines": sum(1 for line in sync_lines if line.words),
    }
    if not sync_lines:
        return SyncResult(reason="歌词源没有提供逐字歌词", stats=stats)
    if not has_word_times(sync_lines):
        return SyncResult(reason="歌词源只有行级时间，没有逐字时间", stats=stats)
    if not lines or not asr_words:
        return SyncResult(reason="缺少歌词行或 ASR 时间，无法核对逐字歌词", stats=stats)

    texts = [line.text for line in lines]
    matched = _match_lines(texts, sync_lines)
    coverage = len(matched) / len(lines)
    stats["matched"] = len(matched)
    stats["coverage"] = round(coverage, 3)
    if coverage < MIN_COVERAGE:
        return SyncResult(
            reason=f"逐字歌词只覆盖 {len(matched)}/{len(lines)} 行歌词，按本地流程处理",
            stats=stats,
        )

    spans = align_spans(texts, asr_words)
    report = AlignReport()
    anchors: dict[int, float] = {}
    for index, (first, last, confidence) in enumerate(spans):
        # Any span at all counts here: a song the recogniser transcribes badly
        # (残機: confidences 0.06–0.87 on lines that really are sung) would
        # otherwise lose most of its anchors. The confidence threshold belongs
        # to the partial-cover test below, which asks a different question.
        if first is None or confidence <= 0:
            continue
        if last is not None:
            report.mapped[index] = LineSpan(index, first, last, confidence)
        start = asr_words[first].start
        if start is not None:
            anchors[index] = start

    # A cover may sing only part of the song. Those lines have no anchor at all,
    # and counting them as "the ASR missed this line" would sink the gate below
    # and throw away a perfectly good word-sync. Decide which lines this
    # recording does not contain first (see partial), then judge the run on what
    # it does sing.
    covered = [False] * len(asr_words)
    for index, span in report.mapped.items():
        if span.confidence <= ABSENT_HEARD_CONFIDENCE:
            continue
        for position in range(max(span.first, 0), min(span.last + 1, len(asr_words))):
            covered[position] = True
    found = partial.find_absent(
        heard=[
            index in report.mapped and report.mapped[index].confidence > ABSENT_HEARD_CONFIDENCE
            for index in range(len(lines))
        ],
        spans=[
            (asr_words[report.mapped[index].first].start, asr_words[report.mapped[index].last].end)
            if index in report.mapped
            and report.mapped[index].confidence > ABSENT_HEARD_CONFIDENCE
            and 0 <= report.mapped[index].first < len(asr_words)
            and 0 <= report.mapped[index].last < len(asr_words)
            else None
            for index in range(len(lines))
        ],
        chars=[len(fold(line.text)) or 1 for line in lines],
        words=asr_words,
        covered=covered,
    )
    absent = set(found.indices)
    stats["absent"] = len(absent)
    if absent:
        stats["absent_lines"] = [index + 1 for index in found.indices]
    sung_lines = len(lines) - len(absent)
    located = sum(1 for index in anchors if index not in absent)
    anchor_share = located / max(1, sung_lines)
    stats["anchors"] = located
    stats["anchor_share"] = round(anchor_share, 3)
    if anchor_share < MIN_ANCHORS:
        return SyncResult(
            reason=f"ASR 只定位到 {located}/{sung_lines} 行，无法核对逐字歌词的时间",
            stats=stats,
        )

    report.absent = list(found.indices)
    report.absent_detail = list(found.detail)
    report.unmatched = [index for index in report.unmatched if index not in absent]
    deltas: dict[int, float] = {}
    for index, position in matched.items():
        if index in absent:
            # A line this recording does not sing has no honest offset; a stray
            # anchor inside the cut would drag the run's median away.
            continue
        anchor = anchors.get(index)
        source_start = sync_lines[position].start
        if anchor is None or source_start is None:
            continue
        deltas[index] = anchor - source_start
    if not deltas:
        return SyncResult(reason="逐字歌词与 ASR 没有可比较的行", stats=stats)
    # A live version starts later than the studio cut, so every line is shifted
    # by the same amount. That constant offset is not a mismatch: each line is
    # re-anchored on its own ASR start, and only the *scatter* around the
    # common offset says whether the two recordings line up at all.
    offset = statistics.median(deltas.values())
    residuals = [abs(value - offset) for value in deltas.values()]
    median = statistics.median(residuals)
    within = sum(1 for value in residuals if value <= MAX_DEVIATION) / len(residuals)
    stats.update(
        offset=round(offset, 3),
        median_deviation=round(median, 3),
        max_deviation=round(max(residuals), 3),
        within_share=round(within, 3),
    )
    if median > MAX_MEDIAN_DEVIATION or within < MIN_WITHIN_SHARE:
        return SyncResult(
            reason=(
                f"逐字歌词与这段音频对不上（扣掉整体 {offset:+.2f} 秒的偏移后，"
                f"仍有 {within:.0%} 的行在 {MAX_DEVIATION:.0f} 秒内，偏差中位数 {median:.2f} 秒），"
                "按本地流程处理"
            ),
            stats=stats,
        )

    # A run of tokens the DP left out between two lines is that line's words
    # misheard, not new content — the lyric text is the authority.
    absorb_strays(lines, asr_words, report)

    # A line with no ASR anchor (the recogniser missed it) is moved by the
    # run's median shift; the rest are anchored on what the ASR actually heard.
    fallback = statistics.median(list(deltas.values()))
    out: list[Line] = []
    unverified = 0
    for index, line in enumerate(lines):
        if index in absent:
            # This recording does not sing this line at all (a partial cover):
            # leaving it in the timeline untimed would read like a failure.
            continue
        position = matched.get(index)
        if position is None:
            out.append(
                Line(
                    text=line.text,
                    start=None,
                    end=None,
                    words=[],
                    source="word-sync-missing",
                    note="歌词源的逐字歌词里没有这一行，未给出时间",
                )
            )
            continue
        anchored = index in deltas
        if not anchored:
            unverified += 1
        out.append(_shift_line(line.text, sync_lines[position], deltas.get(index, fallback), anchored=anchored))
    clamp_overlaps(out)

    timed = sum(1 for line in out if line.start is not None)
    # Everything the ASR heard that no lyric line covers: on a live recording
    # the spoken intro, the MC section, the ad-libs. The source lyrics cannot
    # contain them, and dropping them would lose real content, so they are
    # handed back for the caller's usual keep/drop judgement.
    extras = extra_lines(asr_words, report) if report.mapped else []
    timeline = Timeline(status="none", lines=out)
    timeline.notes = [
        f"逐字歌词来自 {SOURCE_LABEL.get(source, source) or '歌词源'}"
        + (f"的另一个版本（{edition}）" if edition else "")
        + f"：{timed}/{sung_lines} 行按其逐字时间，按这段音频里 ASR 听到的行起点对齐"
        f"（与源时间的偏差中位数 {median:.2f} 秒，整体偏移 {offset:+.2f} 秒）"
    ]
    if unverified:
        timeline.notes.append(
            f"其中 {unverified} 行 ASR 没有定位到，逐字时间按整体偏移平移，建议试听确认"
        )
    timeline.notes.extend(found.notes)
    stats["used"] = True
    stats["timed"] = timed
    stats["unverified"] = unverified
    stats["extras"] = len(extras)
    return SyncResult(
        used=True,
        reason="逐字歌词与 ASR 的时间吻合，采用歌词源的逐字时间并按音频对齐",
        timeline=timeline,
        note=timeline.notes[0],
        stats=stats,
        extras=extras,
        report=report,
    )


def _match_lines(texts: list[str], sync_lines: list[Line]) -> dict[int, int]:
    """Map lyric line index -> word-timed source line, by text.

    Exact-after-folding first (the usual case: same entry, same line breaks);
    then a positional map when both sides have the same line count and the
    texts still look alike overall. Anything else is left unmatched rather than
    guessed.
    """
    used: set[int] = set()
    matched: dict[int, int] = {}
    keys = [_fold(text) for text in texts]
    for index, key in enumerate(keys):
        if not key:
            continue
        for position, line in enumerate(sync_lines):
            if position in used or not line.words:
                continue
            if _fold(line.text) == key:
                matched[index] = position
                used.add(position)
                break

    if len(matched) * 2 < len(texts) and len(sync_lines) == len(texts):
        positional = {
            index: index
            for index in range(len(texts))
            if texts[index].strip() and sync_lines[index].words
        }
        if positional:
            likeness = statistics.median(
                text_likeness(texts[index], sync_lines[index].text) for index in positional
            )
            if likeness >= 0.6:
                return positional
    return matched


def line_anchors(lines: list[Line], sync_lines: list[Line]) -> dict[int, tuple[float, float]]:
    """Line times from the source's word-level lyrics, for **anchoring only**.

    Another edition of the same song usually has 逐字 timings even when this
    lyrics entry has none, and those line times are a good anchor for deciding
    which lines are sung in which stretch of audio. They are not the answer: the
    anchor pass decides block membership and nothing else, every timestamp in
    the timeline still comes from the aligner.
    """
    if not lines or not sync_lines:
        return {}
    out: dict[int, tuple[float, float]] = {}
    for index, position in _match_lines([line.text for line in lines], sync_lines).items():
        source = sync_lines[position]
        start = source.start
        if start is None:
            start = next((word.start for word in source.words if word.start is not None), None)
        end = source.end
        if end is None:
            end = next((word.end for word in reversed(source.words) if word.end is not None), None)
        if start is None:
            continue
        out[index] = (float(start), float(end) if end is not None and end > start else float(start) + 3.0)
    return out


def _shift_line(text: str, source: Line, shift: float, *, anchored: bool = True) -> Line:
    """One lyric line with the source's word spacing, moved onto ``shift``.

    ``anchored`` says whether the shift came from this line's own ASR start; a
    line the recogniser never located is only moved by the run's median shift,
    which is a guess worth a listen, so it carries a flag.
    """
    words = [
        Word(
            text=word.text,
            start=round((word.start or 0.0) + shift, 3),
            end=round((word.end or word.start or 0.0) + shift, 3),
            source="word-sync",
        )
        for word in source.words
    ]
    if words:
        words[0].start = min(words[0].start or 0.0, (source.start or 0.0) + shift)
        previous: float | None = None
        for word in words:
            if previous is not None and (word.start or 0.0) < previous:
                word.start = previous
            if (word.end or 0.0) < (word.start or 0.0) + MIN_WORD_SEC:
                word.end = round((word.start or 0.0) + MIN_WORD_SEC, 3)
            previous = word.start
    start = words[0].start if words else ((source.start or 0.0) + shift)
    end = words[-1].end if words else ((source.end or source.start or 0.0) + shift)
    return Line(
        text=text,
        start=start,
        end=end,
        words=words,
        source="word-sync" if anchored else "word-sync-shifted",
        note=None if anchored else "ASR 没有定位到这一行，逐字时间按整体偏移平移，建议试听确认",
    )


def describe(lines: list[Line]) -> dict[str, Any]:
    """Small summary of a stored word-sync payload, for logs and traces."""
    return {
        "lines": len(lines),
        "word_lines": sum(1 for line in lines if line.words),
        "words": sum(len(line.words) for line in lines),
    }
