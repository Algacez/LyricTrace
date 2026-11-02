"""A cover that sings only part of the song: the unsung lines are dropped.

The lyrics source describes the whole song; a 翻唱 may sing one verse and the
chorus, or be a TV-size edit. Those lines are still in the lyrics, so the run
nobody heard comes back untimed — indistinguishable, row by row, from a line the
recogniser lost. The decision is made from the audio: a run of unheard lines
whose neighbours leave no room for them was not sung.
"""

from app.schemas import Line, Word
from app.services import partial, word_sync
from app.services.providers.llm_aligner import AlignReport, LineSpan, build_timeline


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


def _heard(*flags):
    return list(flags)


def test_a_run_without_room_in_the_audio_is_absent():
    # Four sung lines, then four lines nobody heard, then two more sung ones.
    # The 1 s between the neighbours cannot hold four lines.
    spans = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), None, None, None, None, (9.0, 11.0), (11.0, 13.0)]
    heard = _heard(True, True, True, True, False, False, False, False, True, True)
    words = _words(*[(text, index * 0.5, index * 0.5 + 0.5) for index, text in enumerate("あ" * 26)])
    found = partial.find_absent(heard=heard, spans=spans, chars=[10] * 10, words=words)
    assert found.indices == [4, 5, 6, 7]
    assert found.runs == [(4, 7)]
    assert found.notes and "第 5–8 行" in found.notes[0]


def test_a_single_missed_line_inside_a_sung_verse_is_kept():
    # One line the recogniser lost is a recognition miss, not a cut: a cover
    # cuts whole verses, and dropping a sung line loses content for good.
    spans = [(0.0, 2.0), (2.0, 4.0), None, (6.0, 8.0), (8.0, 10.0)]
    heard = _heard(True, True, False, True, True)
    words = _words(*[(text, index * 0.5, index * 0.5 + 0.5) for index, text in enumerate("あ" * 20)])
    found = partial.find_absent(heard=heard, spans=spans, chars=[10] * 5, words=words)
    assert found.indices == []
    assert found.notes == []


def test_a_long_silence_with_no_speech_drops_the_run():
    # There is room, but the recogniser heard nothing at all across it: the
    # lines were not sung, the gap is an instrumental break.
    spans = [(0.0, 2.0), (2.0, 4.0), None, None, None, (60.0, 62.0)]
    heard = _heard(True, True, False, False, False, True)
    words = _words(("あ", 0.5, 1.0), ("い", 1.5, 2.0), ("う", 60.5, 61.0), ("え", 61.5, 62.0))
    found = partial.find_absent(heard=heard, spans=spans, chars=[5, 5, 5, 5, 5, 5], words=words)
    assert found.indices == [2, 3, 4]
    assert "没有识别到任何演唱" in found.detail[0]["reason"]


def test_a_run_at_the_very_start_or_end_is_dropped_too():
    # A cover that starts at the second verse: the opening lines are never sung.
    spans = [None, None, None, (10.0, 12.0), (12.0, 14.0), (14.0, 16.0)]
    heard = _heard(False, False, False, True, True, True)
    words = _words(("あ", 10.5, 11.0), ("い", 11.5, 12.0), ("う", 12.5, 13.0), ("え", 15.5, 16.0))
    found = partial.find_absent(heard=heard, spans=spans, chars=[6] * 6, words=words)
    assert found.indices == [0, 1, 2]


def test_nothing_is_dropped_when_too_little_was_heard_to_judge():
    spans = [None, None, None, None]
    found = partial.find_absent(heard=_heard(False, False, False, False), spans=spans, chars=[5] * 4, words=[])
    assert found.indices == []


def test_build_timeline_leaves_the_unsung_lines_out():
    lines = [Line(text="あ"), Line(text="い"), Line(text="う")]
    report = AlignReport(
        mapped={0: LineSpan(0, 0, 0, 1.0), 2: LineSpan(2, 2, 2, 1.0)},
        unmatched=[1],
        absent=[1],
    )
    words = _words(("あ", 1.0, 2.0), ("ん", 2.0, 3.0), ("う", 9.0, 10.0))
    timeline = build_timeline(lines, words, report)
    assert [line.text for line in timeline.lines] == ["あ", "う"]


def test_an_absent_line_does_not_count_as_a_missed_one():
    """The run note must not say "N lines were not recognised" for a cut."""
    from app.services.providers.llm_aligner import coverage_note

    words = _words(("あ", 0.0, 1.0))
    report = AlignReport(mapped={0: LineSpan(0, 0, 0)}, unmatched=[], absent=[1])
    note = coverage_note(report, words)
    assert note is None or "没有被识别到" not in note


# --------------------------------------------------------------------------
# the word-sync path: a cover must not sink the gate that validates the source
# --------------------------------------------------------------------------

COVER_TEXTS = ["あかい", "あおい", "きいろ", "みどり", "しろい", "くろい", "あかるい", "くらい"]


def _source_lines():
    return [
        Line(
            text=text,
            start=10.0 + index * 3,
            end=13.0 + index * 3,
            words=[Word(text=text, start=10.0 + index * 3, end=13.0 + index * 3, source="word-sync")],
            source="word-sync",
        )
        for index, text in enumerate(COVER_TEXTS)
    ]


def _sung(texts, offset):
    words: list[Word] = []
    for position, text in enumerate(texts):
        start = offset + position * 3
        step = 3 / len(text)
        words.extend(
            Word(text=char, start=start + i * step, end=start + (i + 1) * step, source="asr")
            for i, char in enumerate(text)
        )
    return words


def test_a_cover_that_sings_half_the_song_keeps_only_the_sung_lines():
    words = _sung(COVER_TEXTS[3:], offset=20.0)
    result = word_sync.apply(
        [Line(text=text) for text in COVER_TEXTS], _source_lines(), words, source="lrclib"
    )
    assert result.used, result.reason
    assert [line.text for line in result.timeline.lines] == COVER_TEXTS[3:]
    assert all(line.start is not None for line in result.timeline.lines)
    assert result.stats["absent"] == 3
    assert any("只唱了歌词的一部分" in note for note in result.timeline.notes)


def test_a_cover_does_not_trip_the_anchor_gate():
    """The unsung lines have no anchor; they must not count as missed ones."""
    words = _sung(COVER_TEXTS[3:], offset=20.0)
    result = word_sync.apply(
        [Line(text=text) for text in COVER_TEXTS], _source_lines(), words, source="lrclib"
    )
    assert result.used, result.reason
    assert result.stats["anchor_share"] >= word_sync.MIN_ANCHORS


def test_a_whole_song_still_keeps_every_line():
    words = _sung(COVER_TEXTS, offset=20.0)
    result = word_sync.apply(
        [Line(text=text) for text in COVER_TEXTS], _source_lines(), words, source="lrclib"
    )
    assert result.used, result.reason
    assert [line.text for line in result.timeline.lines] == COVER_TEXTS
    assert result.stats["absent"] == 0


def test_align_drops_the_unsung_run_and_says_so():
    """End to end through ``LLMLyricAligner.align`` with no LLM answers."""
    import asyncio

    from app.services.providers.llm_aligner import LLMLyricAligner

    class Quiet:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            if '只输出 JSON：{"keep"' in prompt:
                return {"keep": [], "lyric": []}
            if '只输出 JSON：{"lines"' in prompt:
                return {"lines": []}
            return {"picks": []}

    # Seven lines; only the first two and the last two were sung, with a long
    # instrumental in between where the middle three should have been.
    texts = ["あかい", "あおい", "きいろ", "みどり", "しろい", "くろい", "くらい"]
    words = _sung(texts[:2], offset=0.0) + _sung(texts[5:], offset=46.0)
    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = Quiet()
    timeline = asyncio.run(
        aligner.align(None, [Line(text=text) for text in texts], asr_words=words, detect_extras=False)
    )
    assert [line.text for line in timeline.lines] == ["あかい", "あおい", "くろい", "くらい"]
    assert any("第 3–5 行" in note for note in timeline.notes)
