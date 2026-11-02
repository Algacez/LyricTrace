"""Deterministic parts of the aligner: kana DP mapping, tokenizing, interpolation."""

from app.schemas import Line, Word
from app.services.providers.llm_aligner import (
    LLMLyricAligner,
    AlignReport,
    LineSpan,
    align_spans,
    build_timeline,
    key_text,
    tokenize,
)


def test_tokenize_round_trips_and_splits_cjk():
    text = "君の 心が 見えない。 Yeah!"
    tokens = tokenize(text)
    assert "".join(tokens) == text
    assert tokens[0] == "君"
    assert tokens[-1] == "Yeah!"


def test_tokenize_keeps_latin_words_whole():
    assert tokenize("hello world") == ["hello ", "world"]


def test_key_text_uses_the_reading_not_the_written_form():
    # Kanji the ASR spells differently must still compare equal on reading.
    assert key_text("残機") == key_text("ざんき")
    assert key_text("残機") != ""


def _words(pairs):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in pairs]


def test_align_spans_maps_lines_to_word_spans():
    words = _words(
        [
            ("おはよう", 1.0, 1.6),
            ("おつかれ", 1.6, 2.2),
            ("さようなら", 5.0, 6.0),
        ]
    )
    spans = align_spans(["おはよう おつかれ", "さようなら"], words)
    assert spans[0][:2] == (0, 1)
    assert spans[0][2] > 0.5
    assert spans[1][:2] == (2, 2)
    assert spans[1][2] > 0.5


def test_align_spans_leaves_unheard_lines_empty():
    words = _words([("あ", 1.0, 1.2), ("い", 1.2, 1.4)])
    spans = align_spans(["全然違う歌詞です"], words)
    # A monotonic alignment will still attach the line, but the confidence is
    # low enough for the caller to know the ASR never really sang it.
    assert spans[0][2] < 0.5


def test_align_spans_ignores_a_second_match_far_away():
    """A line matched twice (its own singing + the spoken intro) takes one place.

    The global DP is monotonic, so the title was matched onto the intro and onto
    the chorus; the span from one to the other swallowed the whole intro, and
    the intro disappeared from the output.
    """
    words = _words(
        [
            ("这", 1.0, 1.1),
            ("首", 1.1, 1.2),
            ("歌", 1.2, 1.3),
            ("叫", 1.3, 1.4),
            ("我", 1.5, 1.6),
            ("的", 1.6, 1.7),
            ("唱", 3.0, 3.1),
            ("给", 3.1, 3.2),
            ("大", 3.2, 3.3),
            ("家", 3.3, 3.4),
            ("听", 3.4, 3.5),
            ("我", 30.0, 30.1),
            ("的", 30.1, 30.2),
            ("故", 30.2, 30.3),
            ("事", 30.3, 30.4),
            ("未", 30.4, 30.5),
            ("写", 30.5, 30.6),
            ("完", 30.6, 30.7),
        ]
    )
    spans = align_spans(["我的故事未写完", "唱给大家听"], words)
    first = spans[0][:2]
    # The group with the most matches wins: the sung line at 30 s, not the
    # partly-matching mention in the intro at 1.5 s.
    assert first[0] >= 11
    # The intro's words stay uncovered, so they can be kept as their own content.
    assert spans[1][:2][0] >= 6


def test_build_timeline_marks_unmapped_lines():
    lines = [Line(text="残機"), Line(text="残機")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 0)})
    timeline = build_timeline(lines, _words([("ざんき", 1.0, 1.2)]), report)
    assert timeline.lines[0].start == 1.0
    assert timeline.lines[1].start is None
    assert timeline.lines[1].source == "unmatched"
    assert timeline.lines[1].words == []


def test_matched_characters_keep_asr_timing():
    lines = [Line(text="あいう")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 2)})
    words = _words([("あ", 1.0, 1.5), ("い", 1.5, 2.0), ("う", 2.0, 2.5)])
    timeline = build_timeline(lines, words, report)
    line = timeline.lines[0]
    assert [word.start for word in line.words] == [1.0, 1.5, 2.0]
    assert all(word.source == "asr" for word in line.words)
    assert line.start == 1.0 and line.end == 2.5


def test_missing_characters_are_interpolated_not_invented():
    lines = [Line(text="あXい")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)})
    words = _words([("あ", 1.0, 1.5), ("い", 2.0, 2.5)])
    timeline = build_timeline(lines, words, report)
    words_out = timeline.lines[0].words
    assert [word.source for word in words_out] == ["asr", "interpolated", "asr"]
    # The interpolated slot sits strictly between its timed neighbours.
    assert 1.5 <= words_out[1].start <= words_out[1].end <= 2.0


def test_untimed_edges_start_at_the_edge_of_the_span():
    # Characters the ASR never heard still start where the line was sung: the
    # leading ones spread from the start of the mapped span, so a misheard
    # first syllable does not drag the line's start onto the first match.
    lines = [Line(text="XXあ")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)})
    words = _words([("ら", 3.0, 3.2), ("あ", 3.2, 3.4)])
    timeline = build_timeline(lines, words, report)
    words_out = timeline.lines[0].words
    assert [word.source for word in words_out] == ["interpolated", "asr"]
    assert words_out[0].start == 3.0
    assert words_out[0].end == words_out[1].start
    assert words_out[1].end == 3.4


def test_sparse_matches_spread_the_line_over_its_span():
    # One stray matching character must not drag the whole line onto it: with
    # too few matches the line is spread over the span the ASR sang instead.
    lines = [Line(text="あいうえおかきく")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 2)})
    words = _words([("あ", 10.0, 11.0), ("ん", 12.0, 13.0), ("ん", 14.0, 15.0)])
    timeline = build_timeline(lines, words, report)
    line = timeline.lines[0]
    assert line.start == 10.0
    assert line.end == 15.0
    assert all(word.source == "interpolated" for word in line.words)


def test_unmatched_line_is_spread_over_its_span_not_left_at_zero():
    # A line whose characters the ASR never heard still gets the timing of the
    # span it was mapped to; nothing lands at t=0.
    lines = [Line(text="全然違う")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 0)})
    timeline = build_timeline(lines, _words([("あ", 10.0, 11.0)]), report)
    words_out = timeline.lines[0].words
    assert all(word.source == "interpolated" for word in words_out)
    assert words_out[0].start == 10.0
    assert words_out[-1].end == 11.0


# --------------------------------------------------------------------------
# LLM verification must not flag repeated lyric lines
# --------------------------------------------------------------------------


class _PickingClient:
    """Replies with the label whose candidate text matches a wanted reading.

    ``wanted`` maps the 1-based sentence number in the prompt to the heard text
    the reply should point at, so a test can say "line 1 picks line 2's span"
    without knowing how the candidates were shuffled.
    """

    def __init__(self, wanted):
        self.wanted = wanted

    def available(self):
        return True

    def chat_json(self, prompt, max_tokens=0):
        picks = []
        sentence = None
        for raw in prompt.splitlines():
            line = raw.strip()
            if line.startswith("句"):
                sentence = int(line[1 : line.index(" ")])
            elif line[:2] in ("A)", "B)", "C)", "D)") and sentence in self.wanted:
                if line[3:].strip() == self.wanted[sentence]:
                    picks.append({"line": sentence, "pick": line[0]})
        return {"picks": picks}


def _verify(lines, words, report, wanted):
    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = _PickingClient(wanted)
    aligner.verify_lines(lines, words, report)
    return lines


def test_repeated_lyric_lines_are_not_flagged_when_the_spans_read_alike():
    # A chorus line appears twice; the model picks the other occurrence's span.
    # Both spans read the same, so that pick says nothing about correctness and
    # must not push the line into the review list.
    lines = [Line(text="間違ってないよな"), Line(text="間違ってないよな"), Line(text="違う歌詞")]
    words = _words(
        [("間違ってないよな", 1.0, 2.0), ("間違ってないよな", 3.0, 4.0), ("違う歌詞", 5.0, 6.0)]
    )
    report = AlignReport(mapped={0: LineSpan(0, 0, 0, 1.0), 1: LineSpan(1, 1, 1, 1.0), 2: LineSpan(2, 2, 2, 1.0)})
    _verify(lines, words, report, {1: key_text("間違ってないよな")})
    assert lines[0].source == "asr"
    assert report.weak == []


def test_repeated_lines_with_different_readings_are_still_flagged():
    # Same repeated lyric text, but the two spans really are different singing:
    # the pick is informative again and the line stays flagged for review.
    lines = [Line(text="ああ"), Line(text="ああ")]
    words = _words([("ああ", 1.0, 2.0), ("いい", 3.0, 4.0)])
    report = AlignReport(mapped={0: LineSpan(0, 0, 0, 1.0), 1: LineSpan(1, 1, 1, 1.0)})
    _verify(lines, words, report, {1: key_text("いい")})
    assert lines[0].source == "asr-bad"
    assert report.weak == [0]


def test_mutual_picks_between_repeated_lines_are_not_flagged():
    lines = [Line(text="らら"), Line(text="らら")]
    words = _words([("らら", 1.0, 2.0), ("らら", 3.0, 4.0)])
    report = AlignReport(mapped={0: LineSpan(0, 0, 0, 1.0), 1: LineSpan(1, 1, 1, 1.0)})
    _verify(lines, words, report, {1: key_text("らら"), 2: key_text("らら")})
    assert [line.source for line in lines] == ["asr", "asr"]
    assert report.weak == []
