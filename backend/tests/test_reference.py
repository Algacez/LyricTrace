"""LRCLIB's timeline as a positional hint, and ASR extras that get kept."""

from app.schemas import Line, Word
from app.services.providers.llm_aligner import (
    AlignReport,
    LineSpan,
    align_spans,
    build_timeline,
    extra_lines,
    fold,
)
from app.services.reference import match_reference, reference_lines


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


SYNCED = """[00:01.00]あいうえお
[00:05.00]かきくけこ
[00:09.00]さしすせそ
"""


def test_reference_lines_carry_the_api_timestamps():
    lines = reference_lines(SYNCED)
    assert [line.text for line in lines] == ["あいうえお", "かきくけこ", "さしすせそ"]
    assert [line.start for line in lines] == [1.0, 5.0, 9.0]


def test_reference_lines_empty_without_stamps():
    assert reference_lines("ああ\nいい") == []
    assert reference_lines(None) == []


def test_match_reference_pairs_lines_by_text_ignoring_punctuation():
    reference = reference_lines(SYNCED)
    matched = match_reference(["あいうえお。", "さしすせそ！"], reference)
    assert matched[0].start == 1.0
    assert matched[1].start == 9.0


def test_match_reference_falls_back_to_positions_for_edited_lyrics():
    reference = reference_lines(SYNCED)
    matched = match_reference(["完全に別の歌詞", "これも別", "これも"], reference)
    # Same number of lines: positions still say roughly where each one goes.
    assert sorted(matched) == [0, 1, 2]
    assert matched[2].start == 9.0


def test_match_reference_discards_mismatched_shapes():
    reference = reference_lines(SYNCED)
    assert match_reference(["全然違う一行だけ"], reference) == {}


class _Aligner:
    """The aligner without the network: only the deterministic steps."""

    from app.services.providers.llm_aligner import LLMLyricAligner as _Real

    recover_with_reference = _Real.recover_with_reference


def test_reference_recovery_places_a_line_the_dp_could_not():
    # The DP maps the first line only; the second is far from its neighbours in
    # ASR text, but its LRCLIB stamp says it is sung at ~10 s.
    words = _words(
        ("あ", 1.0, 1.5), ("い", 1.5, 2.0),
        ("か", 10.0, 10.5), ("き", 10.5, 11.0),
    )
    lines = [Line(text="あい"), Line(text="かき")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)}, unmatched=[1])
    reference = {1: Line(text="かき", start=9.0)}

    _Aligner().recover_with_reference(lines, words, report, reference)

    assert 1 in report.mapped
    assert (report.mapped[1].first, report.mapped[1].last) == (2, 3)
    assert report.recovered == [1]
    assert 1 in report.weak


def test_reference_recovery_leaves_the_line_alone_when_nothing_matches():
    words = _words(("あ", 1.0, 1.5), ("ん", 10.0, 10.5), ("ん", 11.0, 11.5))
    lines = [Line(text="あ"), Line(text="全然違う歌詞です")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)}, unmatched=[1])

    _Aligner().recover_with_reference(lines, words, report, {1: Line(text="x", start=10.0)})

    assert 1 not in report.mapped
    assert report.unmatched == [1]


def test_extra_lines_keep_uncovered_asr_text_with_its_timing():
    words = _words(
        ("あ", 0.0, 0.5),
        ("い", 0.5, 1.0),
        ("ラララ", 5.0, 6.0),
        ("ラララ", 6.0, 7.0),
    )
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)})
    extras = extra_lines(words, report)
    assert len(extras) == 1
    extra = extras[0]
    assert extra.text == "ララララララ"
    assert extra.start == 5.0 and extra.end == 7.0
    assert extra.source == "asr-extra"
    assert extra.note


def test_extra_lines_splits_at_pauses():
    words = _words(("あ", 0.0, 0.5), ("ラララ", 5.0, 6.0), ("ルルル", 9.0, 10.0))
    extras = extra_lines(words, AlignReport(mapped={0: LineSpan(0, 0, 0)}))
    assert [extra.text for extra in extras] == ["ラララ", "ルルル"]


def test_extra_lines_empty_when_everything_is_covered():
    words = _words(("あ", 0.0, 0.5))
    assert extra_lines(words, AlignReport(mapped={0: LineSpan(0, 0, 0)})) == []


def test_extras_are_inserted_in_time_order_between_lyric_lines():
    words = _words(("あ", 0.0, 0.5), ("い", 10.0, 10.5))
    report = AlignReport(mapped={0: LineSpan(0, 0, 1), 1: LineSpan(1, 1, 1)})
    extras = [Line(text="ララ", start=5.0, end=6.0, words=[], source="asr-extra")]
    timeline = build_timeline(
        [Line(text="あ"), Line(text="い")], words, report, extras=extras
    )
    assert [line.text for line in timeline.lines] == ["あ", "ララ", "い"]


def test_recovered_line_is_labelled_and_flagged():
    words = _words(("か", 10.0, 10.5))
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)}, recovered=[0], weak=[0])
    timeline = build_timeline([Line(text="か", source="asr-ref")], words, report)
    line = timeline.lines[0]
    assert line.source == "asr-ref"
    assert line.note and "LRCLIB" in line.note


def test_fold_is_the_noise_threshold_helper():
    assert fold("ララ、ラ！") == "ラララ"


def test_flagged_lines_carry_what_the_asr_heard():
    words = _words(("か", 10.0, 10.5), ("き", 10.5, 11.0))
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)}, weak=[0])
    timeline = build_timeline([Line(text="かき", source="asr-partial")], words, report)
    assert timeline.lines[0].heard == "かき"


def test_clean_lines_stay_uncluttered():
    words = _words(("あ", 1.0, 1.5))
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)})
    timeline = build_timeline([Line(text="あ")], words, report)
    assert timeline.lines[0].heard is None


def test_filter_extras_keeps_only_substantial_segments_without_the_llm():
    from app.services.providers.llm_aligner import LLMLyricAligner

    class NoLLM:
        def available(self):
            return False

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = NoLLM()
    extras = [
        Line(text="ララララ", start=5.0, end=6.0, words=[]),
        Line(text="あ", start=9.0, end=9.2, words=[]),  # a stray syllable
    ]
    kept, suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert [extra.text for extra in kept] == ["ララララ"]
    assert suspects == []


def test_filter_extras_keeps_a_short_but_dense_phrase():
    """A phrase spoken quickly is short in time, not in content.

    "唱给大家听" is five different characters in 0.72 s. The old rule required
    a full second, so the tail of the live intro was dropped even though the
    recogniser cannot invent five characters out of an instrument.
    """
    from app.services.providers.llm_aligner import LLMLyricAligner, is_substantial

    assert is_substantial(Line(text="唱给大家听", start=30.16, end=30.88))
    # A sustained instrument misheard as one syllable stays noise.
    assert not is_substantial(Line(text="嗯", start=360.0, end=360.06))
    assert not is_substantial(Line(text="啊啊", start=10.0, end=11.0))

    class NoLLM:
        def available(self):
            return False

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = NoLLM()
    extras = [Line(text="唱给大家听", start=30.16, end=30.88, words=[])]
    kept, _suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert [extra.text for extra in kept] == ["唱给大家听"]


def test_filter_extras_uses_the_llm_verdict_when_available():
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            assert "ララララ" in prompt
            return {"keep": [2]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [
        Line(text="ララララ", start=5.0, end=6.0, words=[]),
        Line(text="ルルルル", start=40.0, end=41.0, words=[]),
    ]
    kept, suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert [extra.text for extra in kept] == ["ルルルル"]
    assert suspects == []


def test_filter_extras_keeps_the_tail_of_a_sentence_split_at_a_breath():
    """``extra_lines`` splits at pauses, so an intro arrives in pieces.

    The short tail ("行") must not be dropped just because it is short — it is
    the end of the sentence right before it.
    """
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [1]}  # only the long clause survives on its own

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [
        Line(text="原本今天有我的一场线下演唱会很可惜现在无法正常举", start=5.52, end=11.76, words=[]),
        Line(text="行", start=11.76, end=12.0, words=[]),
    ]
    kept, _suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert [extra.text for extra in kept] == ["原本今天有我的一场线下演唱会很可惜现在无法正常举", "行"]


def test_filter_extras_keeps_long_speech_the_llm_called_noise():
    """A spoken intro is not a lyric line and not noise either — keep it.

    The recogniser does not invent a whole sentence over an instrumental, so a
    long, varied segment is real content even when the model says drop it.
    """
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": []}  # the model drops everything, as it did live

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [
        Line(text="嗨各位朋友们好我是林渊", start=2.32, end=4.56, words=[]),
        Line(text="原本今天有我的一场线下演唱会很可惜现在无法正常举", start=5.52, end=11.76, words=[]),
    ]
    kept, suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert [extra.text for extra in kept] == [extra.text for extra in extras]
    assert suspects == []


def test_filter_extras_keeps_a_segment_named_as_a_lyric_line():
    """'It is really line 5' must never mean 'drop it'.

    An intro that names the song, or a repeated chorus, is real content. The
    lyric answer only decides whether the stretch is re-heard; if that line
    already has its timing, the segment stays as a sung repetition.
    """
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            # The model keeps nothing and calls segment 1 "really line 1".
            return {"keep": [], "lyric": [{"segment": 1, "line": 1}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text="哪怕结局是平凡")]
    extras = [Line(text="哪怕结局", start=289.28, end=290.4, words=[])]
    # Line 0 already has a span, so this cannot be a mangled line: it is a
    # repeat of that line, and it is kept as its own line.
    kept, suspects = aligner.filter_extras(
        extras, lines, AlignReport(mapped={0: LineSpan(0, 0, 3)})
    )
    assert [extra.text for extra in kept] == ["哪怕结局"]
    assert suspects == []


def test_filter_extras_only_offers_lines_near_the_segment_in_time():
    """A line from another verse reading alike is a coincidence, not this one."""
    from app.services.providers.llm_aligner import LLMLyricAligner

    seen = {}

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            seen["prompt"] = prompt
            # The model insists the intro is line 1 — 200 s away. Ignore it.
            return {"keep": [1], "lyric": [{"segment": 1, "line": 1}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text="记忆就像是画片")]
    words = [Word(text="记", start=200.0, end=200.5, source="asr")]
    extras = [Line(text="嗨各位朋友们好", start=2.0, end=4.0, words=[])]
    kept, suspects = aligner.filter_extras(
        extras, lines, AlignReport(mapped={0: LineSpan(0, 0, 0)}), words=words
    )
    assert "时间上附近没有歌词行" in seen["prompt"]
    assert suspects == []
    assert [extra.text for extra in kept] == ["嗨各位朋友们好"]


def test_filter_extras_keeps_the_title_spoken_in_the_intro():
    """The intro names the song; that is a line, not a duplicate to drop."""
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [], "lyric": [{"segment": 1, "line": 5}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text=f"第 {n} 行") for n in range(1, 5)] + [Line(text="我的故事未写完")]
    extras = [Line(text="我的故事未写完", start=26.24, end=28.64, words=[])]
    kept, suspects = aligner.filter_extras(extras, lines, AlignReport(mapped={4: LineSpan(4, 0, 3)}))
    assert [extra.text for extra in kept] == ["我的故事未写完"]
    assert suspects == []


def test_filter_extras_still_drops_a_long_stretch_of_one_syllable():
    """A sustained instrument misheard as text is a dozen ラ, not a sentence."""
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": []}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [Line(text="ララララララララララ", start=5.0, end=9.0, words=[])]
    kept, suspects = aligner.filter_extras(extras, [Line(text="あ")], AlignReport())
    assert kept == [] and suspects == []


def test_separate_spans_gives_a_shared_boundary_word_to_one_line():
    from app.services.providers.llm_aligner import separate_spans

    report = AlignReport(
        mapped={0: LineSpan(0, 0, 5), 1: LineSpan(1, 5, 8)}  # word 5 in both
    )
    separate_spans(report)
    assert report.mapped[1].first == 6
    assert report.mapped[0].last == 5


def test_separate_spans_leaves_a_lost_line_for_the_human_flag():
    from app.services.providers.llm_aligner import separate_spans

    report = AlignReport(mapped={0: LineSpan(0, 0, 3), 1: LineSpan(1, 0, 3)})
    separate_spans(report)
    assert report.mapped[1] == LineSpan(1, 0, 3)


def test_a_short_stray_run_is_folded_into_the_line_it_touches():
    """「有人悻悻地离开」 followed by a misheard 「心情的离开」 is one line."""
    from app.services.providers.llm_aligner import absorb_strays

    words = _words(
        ("有人", 195.0, 195.5),
        ("悻悻地", 195.5, 196.2),
        ("离开", 196.2, 196.8),
        ("心情的", 196.8, 197.5),
        ("离开", 197.5, 198.2),
        ("有人", 210.0, 210.5),
        ("添一笔", 210.5, 211.2),
        ("色彩", 211.2, 211.9),
    )
    lines = [Line(text="有人悻悻地离开"), Line(text="有人添一笔色彩")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 2), 1: LineSpan(1, 5, 7)})
    absorb_strays(lines, words, report)
    # The line now runs to the last token the ASR really sang there.
    assert report.mapped[0] == LineSpan(0, 0, 4)
    # The lyric text is untouched: only the span grew.
    assert lines[0].text == "有人悻悻地离开"
    assert report.mapped[1] == LineSpan(1, 5, 7)


def test_a_stray_run_far_from_any_line_stays_an_extra():
    from app.services.providers.llm_aligner import absorb_strays

    words = _words(
        ("有人", 10.0, 10.5),
        ("悻悻地", 10.5, 11.2),
        ("离开", 11.2, 11.8),
        ("谢谢大家", 40.0, 41.5),
    )
    lines = [Line(text="有人悻悻地离开")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 2)})
    absorb_strays(lines, words, report)
    assert report.mapped[0] == LineSpan(0, 0, 2)


def test_a_long_stray_run_is_a_complete_new_sentence_not_a_mishearing():
    from app.services.providers.llm_aligner import EXTEND_MAX_WORDS, absorb_strays

    strays = [("字", 12.0 + index * 0.3, 12.3 + index * 0.3) for index in range(EXTEND_MAX_WORDS + 1)]
    words = _words(("有人", 10.0, 10.5), ("离开", 10.5, 11.2), *strays)
    lines = [Line(text="有人离开")]
    report = AlignReport(mapped={0: LineSpan(0, 0, 1)})
    absorb_strays(lines, words, report)
    assert report.mapped[0] == LineSpan(0, 0, 1)


def test_clamp_overlaps_keeps_lines_in_order():
    from app.services.providers.llm_aligner import clamp_overlaps

    lines = [
        Line(text="a", start=10.0, end=12.0, words=[Word(text="a", start=10.0, end=12.0)]),
        Line(text="b", start=11.0, end=14.0, words=[Word(text="b", start=11.0, end=14.0)]),
    ]
    clamp_overlaps(lines)
    assert lines[0].end == 11.0
    assert lines[0].words[0].end == 11.0
    assert lines[1].start == 11.0


def test_zero_width_asr_tokens_get_a_usable_duration():
    from app.services.asr_lines import repair_word_times

    words = _words(("あ", 1.0, 1.0), ("い", 2.0, 2.0), ("う", 2.5, 3.0))
    fixed = repair_word_times(words)
    # The start is never moved (it is the timing source); only the end grows.
    assert [word.start for word in fixed] == [1.0, 2.0, 2.5]
    assert fixed[0].end == 2.0  # stretched to the next token's start
    assert fixed[1].end > fixed[1].start
    assert fixed[1].end <= 2.5
    assert fixed[2].end == 3.0


def test_repair_word_times_does_not_mutate_the_input():
    from app.services.asr_lines import repair_word_times

    words = _words(("あ", 1.0, 1.0))
    repair_word_times(words)
    assert words[0].end == 1.0


def test_zero_width_characters_are_given_a_width_and_kept_in_order():
    from app.services.providers.llm_aligner import _enforce_word_times

    words = [
        Word(text="あ", start=1.0, end=1.0),
        Word(text="い", start=1.0, end=1.02),
        Word(text="う", start=0.98, end=1.5),
    ]
    _enforce_word_times(words)
    assert words[0].end >= words[0].start
    assert all(word.end > word.start for word in words)
    starts = [word.start for word in words]
    assert starts == sorted(starts)
    assert words[-1].end == 1.5


def test_matched_reference_aligns_the_source_timeline_onto_the_lyrics():
    from app.services.reference import matched_reference

    lines = matched_reference("あいうえお。\n（间奏）\nさしすせそ！", SYNCED)
    # The interlude has no counterpart in the source, so it gets no hint at all.
    assert [(item.index, item.start) for item in lines] == [(0, 1.0), (2, 9.0)]


def test_matched_reference_is_empty_without_a_source_timeline():
    from app.services.reference import matched_reference

    assert matched_reference("ああ\nいい", None) == []
    assert matched_reference("", SYNCED) == []


def test_filter_extras_flags_a_segment_that_is_really_a_lost_lyric_line():
    """A mangled line looks like new content; the LLM names the line it is."""
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            # The prompt must show the lyrics by line number and the readings.
            assert "2. 君の心" in prompt
            assert "读音" in prompt
            return {"keep": [1], "lyric": [{"segment": 1, "line": 2}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text="あの日"), Line(text="君の心"), Line(text="遠い")]
    extras = [Line(text="きみのこころ", start=5.0, end=7.0, words=[])]
    report = AlignReport(unmatched=[1])
    kept, suspects = aligner.filter_extras(extras, lines, report)

    assert [extra.text for extra in kept] == ["きみのこころ"]
    assert suspects == [(extras[0], 1, "君の心")]


def test_a_suspect_line_that_already_has_a_span_is_left_alone():
    """Moving a placed line is a different, riskier fix — only unmatched lines."""
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [1], "lyric": [{"segment": 1, "line": 1}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text="君の心")]
    extras = [Line(text="きみのこころ", start=5.0, end=7.0, words=[])]
    kept, suspects = aligner.filter_extras(extras, lines, AlignReport(mapped={0: LineSpan(0, 0, 0)}))
    assert [extra.text for extra in kept] == ["きみのこころ"]
    assert suspects == []


def test_filter_extras_ignores_out_of_range_lyric_numbers():
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [1], "lyric": [{"segment": 1, "line": 9}, {"segment": 4, "line": 1}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [Line(text="きみのこころ", start=5.0, end=7.0, words=[])]
    kept, suspects = aligner.filter_extras(extras, [Line(text="君の心")], AlignReport(unmatched=[0]))
    assert [extra.text for extra in kept] == ["きみのこころ"]
    assert suspects == []


def test_filter_extras_with_lyrics_only_keeps_nothing_but_still_names_the_line():
    """「识别现场加唱」 off: no new line, but the mangled-line answer survives.

    The switch exists so the result holds the original lyrics and nothing else.
    It must not switch off the *other* job of this step — noticing that a
    segment is really a lyric line the recogniser mangled, which the caller
    re-hears so that line gets a time.
    """
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [1], "lyric": [{"segment": 1, "line": 2}]}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    lines = [Line(text="あの日"), Line(text="君の心")]
    extras = [Line(text="きみのこころ", start=5.0, end=7.0, words=[])]
    kept, suspects = aligner.filter_extras(
        extras, lines, AlignReport(unmatched=[1]), lyrics_only=True
    )
    assert kept == []
    assert suspects == [(extras[0], 1, "君の心")]


def test_filter_extras_with_lyrics_only_drops_an_ordinary_extra():
    from app.services.providers.llm_aligner import LLMLyricAligner

    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"keep": [1], "lyric": []}

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    extras = [Line(text="嗨各位朋友们好", start=2.0, end=4.0, words=[])]
    kept, suspects = aligner.filter_extras(
        extras, [Line(text="君の心")], AlignReport(mapped={0: LineSpan(0, 0, 0)}), lyrics_only=True
    )
    assert kept == []
    assert suspects == []


class _ExtrasClient:
    """Answers every aligner prompt: verify picks nothing, extras keep segment 1.

    ``named_line`` (0-based) makes the model call the extra segment "really that
    lyric line", which is the ASR-misrecognition route.
    """

    def __init__(self, keep=(), named_line=None):
        self.keep = list(keep)
        self.named_line = named_line
        self.prompts: list[str] = []

    def available(self):
        return True

    def chat_json(self, prompt, max_tokens=0):
        self.prompts.append(prompt)
        if "只输出 JSON：{\"keep\"" in prompt:
            lyric = [] if self.named_line is None else [{"segment": 1, "line": self.named_line + 1}]
            return {"keep": self.keep, "lyric": lyric}
        if "只输出 JSON：{\"lines\"" in prompt:
            return {"lines": []}
        return {"picks": []}


def _run_align(detect_extras, *, named_line=None, resolved=None):
    import asyncio

    from app.services.providers.llm_aligner import LLMLyricAligner

    words = _words(("あの日", 0.0, 1.0), ("嗨各位朋友们好", 5.0, 7.0))
    # The second lyric line is not sung at all, so the stray tokens stay stray.
    lines = [Line(text="あの日"), Line(text="XYZ")]
    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = _ExtrasClient(keep=[1], named_line=named_line)
    calls: list[object] = []

    async def resolve_extra(suspects, current):
        calls.append(list(suspects))
        return resolved

    timeline = asyncio.run(
        aligner.align(
            None,
            lines,
            asr_words=words,
            detect_extras=detect_extras,
            resolve_extra=resolve_extra,
        )
    )
    return timeline, calls


def test_detect_extras_off_outputs_the_lyric_lines_only():
    """The default: the recogniser may not add a line of its own."""
    timeline, calls = _run_align(detect_extras=False)
    assert [line.text for line in timeline.lines] == ["あの日", "XYZ"]
    assert all(line.source != "asr-extra" for line in timeline.lines)
    # No suspect was named, so nothing was re-heard either.
    assert calls == []
    assert any("未开启「识别现场加唱」" in note for note in timeline.notes)


def test_detect_extras_off_still_maps_a_misrecognised_segment_onto_its_line():
    """ASR heard a lyric line badly: the LLM names it, the audio is re-heard."""
    timeline, calls = _run_align(detect_extras=False, named_line=1)
    assert [line.text for line in timeline.lines] == ["あの日", "XYZ"]
    assert all(line.source != "asr-extra" for line in timeline.lines)
    # The mangled-line route survives the switch: re-hearing is how the line
    # gets a time at all.
    assert len(calls) == 1
    assert calls[0][0][1] == 1


def test_detect_extras_on_keeps_the_sung_but_unwritten_content():
    timeline, _calls = _run_align(detect_extras=True)
    assert sorted(line.text for line in timeline.lines) == sorted(["あの日", "嗨各位朋友们好", "XYZ"])
    extra = next(line for line in timeline.lines if line.text == "嗨各位朋友们好")
    assert extra.source == "asr-extra"
    assert extra.start == 5.0 and extra.end == 7.0
