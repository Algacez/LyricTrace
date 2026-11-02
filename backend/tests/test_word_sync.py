"""Word-level (逐字) lyrics from a source, used only when they fit this audio."""

from app.schemas import Line, Word
from app.services import word_sync

LYRICSFILE = """
version: '1.0'
metadata:
  title: 残機
  artist: ずっと真夜中でいいのに。
  instrumental: false
lines:
  - text: しょっぱいぜ
    start_ms: 10000
    end_ms: 12000
    words:
      - text: しょ
        start_ms: 10000
        end_ms: 10400
      - text: っぱ
        start_ms: 10400
        end_ms: 10800
      - text: いぜ
        start_ms: 10800
        end_ms: 12000
  - text: 自暴自棄です
    start_ms: 12000
    end_ms: 14000
    words:
      - text: 自暴
        start_ms: 12000
        end_ms: 12800
      - text: 自棄
        start_ms: 12800
        end_ms: 13400
      - text: です
        start_ms: 13400
        end_ms: 14000
"""

YRC = """{"t":0,"c":[{"tx":"作词: "},{"tx":"唐恬"}]}
[22310,4300](22310,2880,0)都 (25190,310,0)是(25500,290,0)勇(25790,230,0)敢(26020,590,0)的
[28570,2000](28570,600,0)你(29170,400,0)额(29570,500,0)头
"""


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


def _lyric_lines(*texts):
    return [Line(text=text) for text in texts]


def test_lyricsfile_words_are_parsed_with_their_times():
    lines = word_sync.parse_lyricsfile(LYRICSFILE)
    assert [line.text for line in lines] == ["しょっぱいぜ", "自暴自棄です"]
    assert [word.text for word in lines[0].words] == ["しょ", "っぱ", "いぜ"]
    assert lines[0].words[0].start == 10.0
    assert lines[1].words[-1].end == 14.0
    assert word_sync.has_word_times(lines)


def test_lyricsfile_without_words_has_no_word_times():
    plain = """
metadata:
  instrumental: false
lines:
  - text: Hello world
    start_ms: 1000
"""
    lines = word_sync.parse_lyricsfile(plain)
    assert [line.text for line in lines] == ["Hello world"]
    assert not word_sync.has_word_times(lines)


def test_malformed_lyricsfile_is_simply_no_data():
    assert word_sync.parse_lyricsfile("not: [valid") == []
    assert word_sync.parse_lyricsfile("") == []


def test_yrc_keeps_spaces_and_skips_credit_rows():
    lines = word_sync.parse_yrc(YRC)
    # The space after 都 is part of the source text and must survive.
    assert [line.text for line in lines] == ["都 是勇敢的", "你额头"]
    # The space belongs to 都 and must survive: without it the line glues.
    assert lines[0].words[0].text == "都 "
    assert lines[0].words[1].start == 25.19
    assert lines[0].start == 22.31
    assert lines[1].words[-1].text == "头"


def test_matching_word_sync_is_reanchored_onto_the_audio():
    """The source says 10.0 s; this recording sings it at 12.5 s."""
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    words = _words(
        ("しょ", 12.5, 12.9),
        ("っぱ", 12.9, 13.3),
        ("いぜ", 13.3, 14.5),
        ("自暴", 14.5, 15.3),
        ("自棄", 15.3, 15.9),
        ("です", 15.9, 16.5),
    )
    result = word_sync.apply(_lyric_lines("しょっぱいぜ", "自暴自棄です"), sync_lines, words, source="lrclib")
    assert result.used, result.reason
    timeline = result.timeline
    assert timeline is not None
    # Every line keeps the source's intra-line spacing, moved by its own shift.
    assert timeline.lines[0].start == 12.5
    assert timeline.lines[0].words[1].start == 12.9
    assert timeline.lines[1].start == 14.5
    assert all(word.source == "word-sync" for line in timeline.lines for word in line.words)
    assert timeline.notes and "逐字歌词来自 LRCLIB" in timeline.notes[0]


def test_word_sync_is_rejected_when_it_does_not_fit_the_audio():
    """Different edits: after removing the common offset the lines still scatter."""
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    words = _words(
        ("しょ", 40.0, 40.4),
        ("っぱ", 40.4, 40.8),
        ("いぜ", 40.8, 42.0),
        ("自暴", 60.0, 60.8),  # 48 s later than the source says, not 30
        ("自棄", 60.8, 61.4),
        ("です", 61.4, 62.0),
    )
    result = word_sync.apply(_lyric_lines("しょっぱいぜ", "自暴自棄です"), sync_lines, words, source="lrclib")
    assert not result.used
    assert "对不上" in result.reason
    assert result.timeline is None


def test_a_constant_offset_does_not_reject_the_source():
    """A live cut starts 35 s later than the studio one; every line moves alike."""
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    words = _words(
        ("しょ", 45.0, 45.4),
        ("っぱ", 45.4, 45.8),
        ("いぜ", 45.8, 47.0),
        ("自暴", 47.0, 47.8),
        ("自棄", 47.8, 48.4),
        ("です", 48.4, 49.0),
    )
    result = word_sync.apply(_lyric_lines("しょっぱいぜ", "自暴自棄です"), sync_lines, words, source="netease")
    assert result.used, result.reason
    # Each line is re-anchored on what the ASR heard, not shifted by the median.
    assert result.timeline.lines[0].start == 45.0
    assert result.timeline.lines[1].start == 47.0
    assert result.stats["offset"] == 35.0


def test_word_sync_is_rejected_when_it_covers_too_few_lines():
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    texts = _lyric_lines("しょっぱいぜ", "自暴自棄です", "いらっしゃいませ", "腹歌満たしてる")
    words = _words(("しょ", 10.0, 10.4), ("自暴", 12.0, 12.8))
    result = word_sync.apply(texts, sync_lines, words, source="lrclib")
    assert not result.used
    assert "覆盖" in result.reason


def test_word_sync_needs_asr_anchors():
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    result = word_sync.apply(_lyric_lines("しょっぱいぜ", "自暴自棄です"), sync_lines, [], source="lrclib")
    assert not result.used


def test_a_line_the_asr_missed_is_moved_by_the_median_shift_and_flagged():
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE) + [
        Line(
            text="いらっしゃいませ",
            start=14.0,
            end=16.0,
            words=[
                Word(text="いらっ", start=14.0, end=15.0, source="word-sync"),
                Word(text="しゃいませ", start=15.0, end=16.0, source="word-sync"),
            ],
            source="word-sync",
        )
    ]
    # The ASR heard lines 1 and 3 but lost line 2, which is moved by the same
    # 2.5 s shift as its neighbours — and flagged, because that is a guess.
    words = _words(
        ("しょ", 12.5, 12.9),
        ("っぱ", 12.9, 13.3),
        ("いぜ", 13.3, 14.5),
        ("いらっ", 16.5, 17.5),
        ("しゃいませ", 17.5, 18.5),
    )
    result = word_sync.apply(
        _lyric_lines("しょっぱいぜ", "自暴自棄です", "いらっしゃいませ"), sync_lines, words, source="lrclib"
    )
    assert result.used, result.reason
    second = result.timeline.lines[1]
    assert second.start == 14.5
    assert second.source == "word-sync-shifted"
    assert second.note
    assert any("试听" in note for note in result.timeline.notes)


def test_a_line_absent_from_the_source_is_left_untimed():
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    words = _words(("しょ", 10.0, 10.4), ("っぱ", 10.4, 10.8), ("いぜ", 10.8, 12.0))
    texts = _lyric_lines("しょっぱいぜ", "自暴自棄です", "いらっしゃいませ")
    result = word_sync.apply(texts, sync_lines, words, source="lrclib")
    assert not result.used  # 2 of 3 lines covered is below the floor
    assert "覆盖" in result.reason


def test_stored_payload_round_trips_by_format():
    lines = word_sync.from_stored(YRC, "yrc")
    # The space after 都 is part of the source text and must survive.
    assert [line.text for line in lines] == ["都 是勇敢的", "你额头"]
    assert word_sync.from_stored(YRC, "lyricsfile") == []  # wrong parser, no lines
    assert word_sync.from_stored(None, "yrc") == []


def test_describe_counts_words_for_the_debug_page():
    summary = word_sync.describe(word_sync.parse_lyricsfile(LYRICSFILE))
    assert summary == {"lines": 2, "word_lines": 2, "words": 6}


def test_word_sync_hands_back_sung_content_the_lyrics_do_not_cover():
    """A live intro is real content: the 逐字 path must not drop it."""
    sync_lines = word_sync.parse_lyricsfile(LYRICSFILE)
    words = _words(
        ("大家好", 1.0, 2.0),
        ("我是", 2.0, 2.5),
        ("某某", 2.5, 3.2),
        ("しょ", 12.5, 12.9),
        ("っぱ", 12.9, 13.3),
        ("いぜ", 13.3, 14.5),
        ("自暴", 14.5, 15.3),
        ("自棄", 15.3, 15.9),
        ("です", 15.9, 16.5),
    )
    result = word_sync.apply(_lyric_lines("しょっぱいぜ", "自暴自棄です"), sync_lines, words, source="lrclib")
    assert result.used, result.reason
    assert [extra.text for extra in result.extras] == ["大家好我是某某"]
    assert result.extras[0].start == 1.0
    assert result.stats["extras"] == 1
    # The lyric lines themselves are still timed from the source's word data.
    assert result.timeline.lines[0].start == 12.5


def test_yrc_reading_chunk_is_dropped_and_its_time_goes_to_the_kanji():
    """The reading is its own ``(start,dur)`` chunk in ``yrc``; it is not sung."""
    raw = "[1000,3000](1000,500,0)残機(1500,500,0)（ざんき）(2000,1000,0)を数える\n"
    lines = word_sync.parse_yrc(raw)
    assert [line.text for line in lines] == ["残機を数える"]
    assert [word.text for word in lines[0].words] == ["残機", "を数える"]
    assert lines[0].words[0].start == 1.0
    assert lines[0].words[0].end == 2.0


def test_attach_extras_honours_the_live_extras_switch():
    """「识别现场加唱」 off: the word-sync path adds nothing either."""
    from app.schemas import Line, Timeline
    from app.services.pipeline import _attach_extras
    from app.services.providers.llm_aligner import AlignReport, LineSpan
    from app.services.word_sync import SyncResult

    class FakeAligner:
        def filter_extras(self, extras, lines, report, trace=None, words=None, lyrics_only=False):
            return ([] if lyrics_only else list(extras)), []

    timeline = Timeline(lines=[Line(text="あの日", start=0.0, end=1.0, source="asr")])
    sync = SyncResult(
        used=True,
        timeline=timeline,
        extras=[Line(text="嗨各位朋友们好", start=5.0, end=7.0, words=[], source="asr-extra")],
        report=AlignReport(mapped={0: LineSpan(0, 0, 0)}),
    )

    kept = _attach_extras(FakeAligner(), sync, [Line(text="あの日")], [], timeline, None, detect=False)
    assert [line.text for line in kept] == ["あの日"]

    kept = _attach_extras(FakeAligner(), sync, [Line(text="あの日")], [], timeline, None, detect=True)
    assert [line.text for line in kept] == ["あの日", "嗨各位朋友们好"]
