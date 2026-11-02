from app.schemas import Line, Timeline, Word
from app.services.lyrics import (
    export_ass,
    export_lrc,
    format_ass_time,
    format_lrc_time,
    parse_lrc,
    parse_plain_lyrics,
)
from app.services.projects import compute_timeline_status


def test_parse_plain_lrc():
    parsed = parse_lrc("[ti:Song]\n[ar:Artist]\n[00:01.00]First\n[00:03.50]Second\n")
    assert parsed.metadata["ti"] == "Song"
    assert [line.text for line in parsed.lines] == ["First", "Second"]
    assert parsed.lines[0].start == 1.0
    assert parsed.lines[0].end == 3.5
    assert parsed.lines[1].end is None
    assert not parsed.has_word_times


def test_parse_multiple_timestamps_duplicate_line():
    parsed = parse_lrc("[00:01.00][00:10.00]Chorus\n[00:05.00]Verse\n")
    assert [line.start for line in parsed.lines] == [1.0, 5.0, 10.0]
    assert parsed.lines[2].text == "Chorus"


def test_parse_enhanced_lrc_word_tags():
    parsed = parse_lrc("[00:01.00]<00:01.00>Hello <00:01.60>world\n[00:03.00]Plain\n")
    assert parsed.has_word_times
    line = parsed.lines[0]
    assert line.text == "Hello world"
    assert [word.text for word in line.words] == ["Hello ", "world"]
    assert line.words[0].start == 1.0
    assert line.words[0].end == 1.6
    assert line.words[1].end == line.end == 3.0


def test_parse_lrc_offset():
    parsed = parse_lrc("[offset:500]\n[00:02.00]Late\n")
    assert parsed.lines[0].start == 2.5


def test_timeline_status_progression():
    untimed = Timeline(lines=[Line(text="a"), Line(text="b")])
    assert compute_timeline_status(untimed) == "untimed"

    line_only = Timeline(lines=[Line(text="a", start=1.0, end=2.0)])
    assert compute_timeline_status(line_only) == "line_only"

    partial = Timeline(
        lines=[
            Line(text="a", start=1.0, end=2.0, words=[Word(text="a", start=1.0, end=2.0)]),
            Line(text="b", start=2.0, end=3.0),
        ]
    )
    assert compute_timeline_status(partial) == "partial"

    complete = Timeline(
        lines=[Line(text="a", start=1.0, end=2.0, words=[Word(text="a", start=1.0, end=2.0)])]
    )
    assert compute_timeline_status(complete) == "complete"
    assert compute_timeline_status(Timeline()) == "none"


def test_format_time_rounding():
    assert format_lrc_time(1.234) == "[00:01.23]"
    assert format_lrc_time(1.235) == "[00:01.24]"
    assert format_lrc_time(61.5) == "[01:01.50]"
    assert format_ass_time(3661.234) == "1:01:01.23"


def test_export_lrc_and_ass_round_trip():
    timeline = Timeline(
        lines=[
            Line(
                text="Hello world",
                start=1.0,
                end=2.5,
                words=[Word(text="Hello ", start=1.0, end=1.5), Word(text="world", start=1.5, end=2.5)],
            )
        ],
        status="complete",
    )
    lrc = export_lrc(timeline, title="T", artist="A")
    assert "[ti:T]" in lrc
    assert "[00:01.00]<00:01.00>Hello <00:01.50>world" in lrc

    reparsed = parse_lrc(lrc)
    assert reparsed.lines[0].text == "Hello world"
    assert reparsed.lines[0].words[1].start == 1.5

    ass = export_ass(timeline)
    assert "Dialogue: 0,0:00:01.00,0:00:02.50" in ass
    assert "{\\k50}Hello {\\k100}world" in ass


def test_export_skips_untimed_lines():
    timeline = Timeline(lines=[Line(text="no time"), Line(text="timed", start=1.0, end=2.0)])
    lrc = export_lrc(timeline)
    assert "no time" not in lrc
    assert "timed" in lrc


def test_parse_plain_lyrics_never_invents_time():
    lines = parse_plain_lyrics("a\n\nb\n")
    assert [line.text for line in lines] == ["a", "b"]
    assert all(line.start is None for line in lines)


def test_credit_lines_are_dropped_from_fallback_lrc():
    """LrcApi/NetEase ship ``[00:07.89]词：…`` lines that are never sung."""
    from app.services.lyrics import lyrics_from_lrc

    raw = (
        "[ti:我的故事未写完]\n"
        "[00:00.00] 作词 : 泠鸢yousa\n"
        "[00:01.00] 作曲 : 泠鸢yousa\n"
        "[00:25.40]记忆就像是画片\n"
        "[00:30.94]被风吹起又飞远\n"
    )
    text, lines = lyrics_from_lrc(raw)
    assert text == "记忆就像是画片\n被风吹起又飞远"
    assert all(line.start is not None for line in lines)


def test_parse_lrc_drops_readings_from_the_line_and_its_words():
    """``残機（ざんき）`` is one sung word plus its reading, not two words."""
    parsed = parse_lrc("[00:01.00]<00:01.00>残機（ざんき）<00:01.60>を数える\n")
    line = parsed.lines[0]
    assert line.text == "残機を数える"
    assert [word.text for word in line.words] == ["残機", "を数える"]


def test_parse_plain_lyrics_drops_readings_but_keeps_sung_parentheses():
    lines = parse_plain_lyrics("運命(さだめ)\nHello (Wow)\n")
    assert [line.text for line in lines] == ["運命", "Hello (Wow)"]
