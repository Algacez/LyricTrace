"""Re-recognising the sentences where the ASR's own timings do not add up."""

import asyncio

import pytest

from app.schemas import Line, Word
from app.services.asr_repair import (
    Gap,
    find_gaps,
    gap_threshold,
    repair_gaps,
    repair_note,
    splice,
    text_share,
    token_duration,
)


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


def _line(text, words):
    return Line(text=text, start=words[0].start, end=words[-1].end, words=words, source="asr")


# The hole used below: 心の中 is sung, then the recogniser loses four seconds,
# and に is stamped correctly. 0.8 s is the floor for a fast song.
HOLE_WORDS = _words(("心", 0.0, 0.2), ("の", 0.2, 0.4), ("に", 4.6, 4.8))
HOLE = Gap(line=0, text="心のに", start=0.4, end=4.6, span_start=0.0, span_end=4.8)
# What a re-recognition of the whole sentence should sound like.
RETRY = _words(("心", 0.0, 0.2), ("の", 0.2, 0.4), ("中", 0.4, 0.6), ("に", 0.6, 0.8))


def test_gap_threshold_scales_with_the_song_pace():
    fast = _words(("あ", 0.0, 0.1), ("い", 0.1, 0.2), ("う", 0.2, 0.3))
    slow = _words(("あ", 0.0, 0.5), ("い", 0.5, 1.0), ("う", 1.0, 1.5))
    assert token_duration(fast) == pytest.approx(0.1)
    assert gap_threshold(fast) == pytest.approx(0.8)  # the floor wins on a fast song
    assert gap_threshold(slow) == pytest.approx(2.5)  # 5x the median token


def test_a_hole_inside_one_line_is_found():
    gaps = find_gaps([_line("心の中に", HOLE_WORDS)], HOLE_WORDS)
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (0.4, 4.6)
    assert gaps[0].duration == pytest.approx(4.2)
    # The sentence is the original lyric line, so the retry window is its span.
    assert (gaps[0].span_start, gaps[0].span_end) == (0.0, 4.8)


def test_a_pause_between_lines_is_not_a_gap():
    # Singing stops between two lines: that is a breath, not a lost phrase.
    first = _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4))
    second = _words(("う", 6.0, 6.2), ("え", 6.2, 6.4))
    assert find_gaps([_line("あい", first), _line("うえ", second)], first + second) == []


def test_splice_replaces_the_whole_sentence_with_the_retry():
    spliced = splice(HOLE_WORDS, HOLE, RETRY, threshold=0.8)
    assert spliced is not None
    merged, remaining = spliced
    # The sentence's timings all come from the retry now, in one consistent
    # pass — not a patchwork of old and new tokens.
    assert [word.text for word in merged] == ["心", "の", "中", "に"]
    assert remaining == pytest.approx(0.0)
    assert merged[0].start == 0.0 and merged[-1].end == 0.8


def test_splice_leaves_the_next_sentence_alone():
    words = HOLE_WORDS + _words(("あ", 6.0, 6.2), ("い", 6.2, 6.4))
    spliced = splice(words, HOLE, RETRY, threshold=0.8)
    assert spliced is not None
    merged = spliced[0]
    assert [word.text for word in merged] == ["心", "の", "中", "に", "あ", "い"]
    assert merged[-2].start == 6.0 and merged[-1].end == 6.4


def test_splice_rejects_a_retry_that_heard_only_part_of_the_sentence():
    # Hearing 中 alone does not tell us where 心 / の / に actually are.
    assert splice(HOLE_WORDS, HOLE, _words(("中", 0.4, 4.4)), threshold=0.8) is None


def test_splice_rejects_a_retry_that_does_not_close_the_hole():
    # Every word is there, but the singer is still silent for seconds in the
    # middle — that is not a fix a listener would hear.
    retry = _words(("心", 0.0, 0.2), ("の", 0.2, 0.4), ("中", 2.4, 2.6), ("に", 4.6, 4.8))
    assert splice(HOLE_WORDS, HOLE, retry, threshold=0.8) is None


def test_text_share_counts_what_the_retry_heard():
    assert text_share("心の中に", "心の中に") == 1.0
    assert text_share("心の中に", "心 の 中 に") == 1.0  # spacing is ignored
    assert text_share("心の中に", "中") == pytest.approx(0.25)
    assert text_share("心の中に", "") == 0.0


def test_repair_gaps_returns_the_same_list_when_nothing_is_accepted():
    lines = [_line("心のに", HOLE_WORDS)]

    def retry(lo, hi):
        return _words(("中", 2.4, 2.6))

    repaired, log = asyncio.run(repair_gaps(HOLE_WORDS, lines, retry))
    assert repaired is HOLE_WORDS
    assert log[0]["accepted"] is False
    assert log[0]["reason"] == "重识别只听到这句的一部分，保留原结果"
    # The timings are untouched, but the run still says the gap was noticed.
    note = repair_note(log) or ""
    assert "第 1 行" in note and "时间保持原样" in note


def test_repair_gaps_accepts_a_retry_that_closes_the_hole():
    lines = [_line("心のに", HOLE_WORDS)]
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        return RETRY

    repaired, log = asyncio.run(repair_gaps(HOLE_WORDS, lines, retry))
    assert [word.text for word in repaired] == ["心", "の", "中", "に"]
    assert log[0]["accepted"] is True
    # The window is the whole sentence plus padding, not just the hole: the
    # recogniser needs the words around the hole to hear them at all.
    assert calls[0][0] == pytest.approx(0.0)
    assert calls[0][1] == pytest.approx(5.2)
    assert "第 1 行" in (repair_note(log) or "")


def test_repair_gaps_retries_a_sentence_once_for_all_its_holes():
    words = _words(("あ", 0.0, 0.2), ("い", 3.0, 3.2), ("う", 6.0, 6.2), ("え", 6.2, 6.4))
    lines = [_line("あいうえ", words)]
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        return _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4), ("う", 0.4, 0.6), ("え", 0.6, 0.8))

    repaired, log = asyncio.run(repair_gaps(words, lines, retry))
    assert len(calls) == 1
    assert log[0]["gaps"] == [[0.2, 3.0], [3.2, 6.0]]
    assert [word.text for word in repaired] == ["あ", "い", "う", "え"]


def test_repair_gaps_survives_a_failed_retry():
    lines = [_line("心のに", HOLE_WORDS)]

    def retry(lo, hi):
        raise RuntimeError("ASR service is down")

    repaired, log = asyncio.run(repair_gaps(HOLE_WORDS, lines, retry))
    assert repaired is HOLE_WORDS
    assert log[0]["error"] == "ASR service is down"


def test_repair_gaps_accepts_an_async_retry():
    lines = [_line("心のに", HOLE_WORDS)]

    async def retry(lo, hi):
        return RETRY

    repaired, _log = asyncio.run(repair_gaps(HOLE_WORDS, lines, retry))
    assert [word.text for word in repaired] == ["心", "の", "中", "に"]


def test_repair_gaps_caps_the_number_of_retries():
    words, lines = [], []
    for index in range(3):
        base = index * 10.0
        chunk = _words((f"あ", base, base + 0.2), (f"い", base + 5.0, base + 5.2))
        words.extend(chunk)
        lines.append(_line("あい", chunk))
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        return []

    asyncio.run(repair_gaps(words, lines, retry, limit=2))
    # Two sentences, and a failed full-sentence retry earns one tighter one,
    # so at most two ASR jobs per sentence in the cap.
    assert len(calls) <= 2 * 2
    # The full retries are the long windows (the whole 10 s sentence).
    full = [call for call in calls if call[1] - call[0] > 5.0]
    assert len(full) == 2


# --------------------------------------------------------------------------
# The hole is not a lost word: the line's span ran into silence
# --------------------------------------------------------------------------

# 心の is sung, then the span runs on for six seconds over an instrumental and
# the recogniser stamps に there. The line is really sung in the first second.
STRETCHED_WORDS = _words(("心", 0.0, 0.4), ("の", 0.4, 0.8), ("に", 6.0, 6.2))
STRETCHED = [_line("心のに", STRETCHED_WORDS)]
# What a tight retry (window ending shortly after the hole starts) hears.
TIGHT = _words(("心", 0.0, 0.4), ("の", 0.4, 0.8), ("に", 0.8, 1.0))


def test_a_stretched_span_is_fixed_by_a_tighter_retry():
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        # The full-sentence window hears the same stretched timings as before.
        if hi - lo > 3.0:
            return STRETCHED_WORDS
        return TIGHT

    repaired, log = asyncio.run(repair_gaps(STRETCHED_WORDS, STRETCHED, retry))
    assert log[0]["accepted"] is True
    assert log[0]["mode"] == "tight"
    assert [(word.text, word.start) for word in repaired] == [
        ("心", 0.0),
        ("の", 0.4),
        ("に", 0.8),
    ]
    # The tight window ends shortly after the hole starts, not at the span end.
    assert calls[-1][1] == pytest.approx(2.3)
    # The run says what happened, and it is not the same note as a full retry.
    note = repair_note(log) or ""
    assert "第 1 行" in note and "人声之外" in note


def test_the_tighter_retry_counts_the_padding_token():
    # The window starts before the line on purpose, so the recogniser hears
    # the first syllable a little before the span begins. That token is this
    # line's, and rejecting it would make a perfect retry look 80% complete.
    padded = _words(("心", 0.3, 0.6), ("の", 0.6, 1.0), ("に", 1.0, 1.4))
    words = _words(("心", 0.6, 1.0), ("の", 1.0, 1.4), ("に", 6.0, 6.2))
    lines = [_line("心のに", words)]

    def retry(lo, hi):
        return words if hi - lo > 3.0 else padded

    repaired, log = asyncio.run(repair_gaps(words, lines, retry))
    assert log[0]["accepted"] is True
    assert log[0]["mode"] == "tight"
    assert [(word.text, word.start) for word in repaired] == [
        ("心", 0.3),
        ("の", 0.6),
        ("に", 1.0),
    ]


def test_the_tighter_retry_must_not_hear_the_next_line():
    # The window runs past the singing and picks up the next line: the whole
    # sentence is there, but so is other content, so it proves nothing.
    def retry(lo, hi):
        if hi - lo > 3.0:
            return STRETCHED_WORDS
        return TIGHT + _words(("あ", 1.0, 1.2), ("い", 1.2, 1.4))

    repaired, log = asyncio.run(repair_gaps(STRETCHED_WORDS, STRETCHED, retry))
    assert log[0]["accepted"] is False
    assert log[0]["tight"]["heard_precision"] == pytest.approx(0.6)
    assert repaired is STRETCHED_WORDS


def test_the_tighter_retry_must_hear_the_whole_line():
    # Hearing only the first half of the line proves nothing: the rest may
    # simply have been cut off by the short window.
    def retry(lo, hi):
        if hi - lo > 3.0:
            return STRETCHED_WORDS
        return _words(("心", 0.0, 0.2))

    repaired, log = asyncio.run(repair_gaps(STRETCHED_WORDS, STRETCHED, retry))
    assert log[0]["accepted"] is False
    assert log[0]["tight"]["accepted"] is False
    assert repaired is STRETCHED_WORDS


def test_the_tighter_retry_is_not_used_when_the_full_retry_works():
    # No second ASR job when the first one already fixed the hole.
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        return _words(("心", 0.0, 0.2), ("の", 0.2, 0.4), ("中", 0.4, 0.6), ("に", 0.6, 0.8))

    repaired, log = asyncio.run(repair_gaps(HOLE_WORDS, [_line("心の中に", HOLE_WORDS)], retry))
    assert log[0]["accepted"] is True
    assert "mode" not in log[0]
    assert len(calls) == 1


# --------------------------------------------------------------------------
# A "new sentence" that is really a mangled lyric line
# --------------------------------------------------------------------------


def test_a_suspect_segment_is_re_heard_and_replaced():
    from app.services.asr_repair import resolve_extra_segments

    words = _words(("あ", 0.0, 0.2), ("きみの", 5.0, 6.0), ("い", 9.0, 9.2))
    suspect = Line(text="きみの", start=5.0, end=6.0, words=[])
    calls = []

    def retry(lo, hi):
        calls.append((lo, hi))
        return _words(("君", 5.0, 5.4), ("の", 5.4, 5.8), ("心", 5.8, 6.4))

    resolved = asyncio.run(resolve_extra_segments([(suspect, 1, "君の心")], words, retry))
    assert resolved is not None
    assert [word.text for word in resolved] == ["あ", "君", "の", "心", "い"]
    # The window is the segment plus context, like the sentence retries.
    assert calls[0] == (pytest.approx(4.6), pytest.approx(6.4))


def test_a_suspect_segment_whose_retry_hears_nothing_changes_nothing():
    from app.services.asr_repair import resolve_extra_segments

    words = _words(("あ", 0.0, 0.2), ("きみの", 5.0, 6.0))
    suspect = Line(text="きみの", start=5.0, end=6.0, words=[])
    resolved = asyncio.run(resolve_extra_segments([(suspect, 1, "君の心")], words, lambda lo, hi: []))
    assert resolved is None


def test_a_failed_suspect_retry_never_fails_the_run():
    from app.services.asr_repair import resolve_extra_segments

    words = _words(("きみの", 5.0, 6.0))
    suspect = Line(text="きみの", start=5.0, end=6.0, words=[])

    def retry(lo, hi):
        raise RuntimeError("ASR service is down")

    assert asyncio.run(resolve_extra_segments([(suspect, 0, "君の心")], words, retry)) is None
