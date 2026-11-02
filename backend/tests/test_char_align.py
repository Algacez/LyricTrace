"""Forced-aligner per-character timings: text folding, guarding, and the retime pass."""

from pathlib import Path

import pytest

from app.schemas import Line, Timeline, Word
from app.services import char_align


@pytest.fixture(autouse=True)
def aligner_in_the_run(monkeypatch):
    """These cases are about the aligner's placement, so it has to take part.

    It is off by default now (the recogniser's own table is the only source);
    the ASR-only path has its own cases in ``test_char_align_asr.py``.
    """
    from app.config import settings

    monkeypatch.setattr(settings, "char_hfa", True)


def _chars(pairs):
    """Build the aligner's ``chars`` list from (char, start, end) triples."""
    return [{"c": char, "start": start, "end": end} for char, start, end in pairs]


def test_language_codes_cover_what_the_service_supports():
    assert char_align.language_code("Japanese") == "ja"
    assert char_align.language_code("Chinese") == "zh"
    assert char_align.language_code("Cantonese") == "zh"
    assert char_align.language_code("English") == "en"
    # Korean is a supported ASR language but not an aligner one.
    assert char_align.language_code("Korean") is None
    assert char_align.language_code(None) is None


def test_characters_fold_back_into_the_lyric_tokens():
    chars = _chars(
        [
            ("ど", 99.3, 99.55),
            ("う", 99.55, 99.57),
            ("に", 99.57, 99.74),
            ("も", 99.74, 100.09),
        ]
    )
    words = char_align.words_from_chars("どうにも", chars)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("ど", 99.3, 99.55),
        ("う", 99.55, 99.57),
        ("に", 99.57, 99.74),
        ("も", 99.74, 100.09),
    ]
    # The aligner placed these characters itself, so they are not "asr": the
    # word stream's estimate is a different, much coarser thing.
    assert all(word.source == "aligner" for word in words)


def test_latin_runs_stay_one_word():
    chars = _chars(
        [
            ("h", 0.0, 0.1),
            ("e", 0.1, 0.2),
            ("y", 0.2, 0.3),
            (" ", None, None),
            ("y", 0.4, 0.5),
            ("o", 0.5, 0.6),
            ("u", 0.6, 0.7),
        ]
    )
    words = char_align.words_from_chars("hey you", chars)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("hey ", 0.0, 0.3),
        ("you", 0.4, 0.7),
    ]


def test_attached_kana_gets_a_usable_width():
    # ょ is zero-width by the service's own convention (attach), and a
    # zero-width character can never light up in karaoke.
    chars = _chars(
        [("し", 0.0, 0.5), ("ょ", 0.5, 0.5), ("っ", 0.6, 0.7)]
    )
    words = char_align.words_from_chars("しょっ", chars)
    assert words[1].start == 0.5
    assert words[1].end >= words[1].start + char_align.MIN_WORD_SEC
    assert words[1].end <= words[2].start


def test_an_untimed_leading_character_starts_where_the_line_started():
    chars = _chars([("お", None, None), ("は", 10.35, 10.6), ("よ", 10.6, 10.9), ("う", 10.9, 11.2)])
    words = char_align.words_from_chars("おはよう", chars, fallback=(10.3, 11.2))
    assert words[0].start == 10.3
    assert words[0].source == "interpolated"
    assert words[1].start == 10.35


def test_alignment_of_other_text_is_rejected():
    # A clip that also holds the next line can align to the wrong stretch.
    chars = _chars([("血", 0.0, 0.4), ("の", 0.4, 0.8), ("温", 0.8, 1.4)])
    assert char_align.words_from_chars("有色素の距離も", chars) is None


def test_times_stay_ascending_and_non_zero():
    chars = _chars([("あ", 1.0, 1.2), ("い", 1.19, 1.19), ("う", 1.4, 1.6)])
    words = char_align.words_from_chars("あいう", chars)
    starts = [word.start for word in words]
    ends = [word.end for word in words]
    assert starts == sorted(starts)
    assert all(end >= start for start, end in zip(starts, ends))
    assert starts[1] >= starts[0]


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


@pytest.fixture()
def synth_audio(tmp_path):
    """A silent WAV long enough to be sliced per line (the aligner is faked)."""
    import wave

    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 20)
    return path


@pytest.fixture()
def voiced_audio(tmp_path):
    """A WAV that is loud until ``voiced_until`` seconds and silent after it."""

    def build(voiced_until: float, total: float = 20.0) -> Path:
        import array
        import wave

        rate = 16000
        loud = min(int(voiced_until * rate), int(total * rate))
        samples = array.array("h", [9000] * loud + [0] * (int(total * rate) - loud))
        path = tmp_path / f"voiced-{voiced_until}.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())
        return path

    return build


def _timeline():
    return Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=2.0,
                words=[Word(text="あ", start=1.0, end=1.5), Word(text="い", start=1.5, end=2.0)],
            ),
            Line(
                text="うえ",
                start=3.0,
                end=4.0,
                words=[Word(text="う", start=3.0, end=3.5), Word(text="え", start=3.5, end=4.0)],
            ),
        ],
    )


def test_retime_replaces_each_line_with_the_aligners_timings(synth_audio, tmp_path, monkeypatch):
    calls = []

    def fake_post(payload):
        calls.append(payload)
        # Characters spaced through most of the line, starting earlier than the
        # ASR had said, so the replacement is visible.
        lo = payload["offset"]
        text = payload["text"]
        return {"chars": _chars([(char, lo + 0.05 + i * 0.4, lo + 0.4 + i * 0.4) for i, char in enumerate(text)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")

    timeline = _timeline()
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    assert len(calls) == 2
    assert calls[0]["lang"] == "ja"
    assert "强制对齐" in notes[0]
    # The first line starts where the aligner put it — the clip's first
    # character — not at the ASR's 1.0.
    assert timeline.lines[0].start == pytest.approx(calls[0]["offset"] + 0.05, abs=0.001)
    assert timeline.lines[0].start != pytest.approx(1.0, abs=0.001)
    assert [word.start for word in timeline.lines[0].words] == sorted(
        word.start for word in timeline.lines[0].words
    )
    # A character lasts until the next one starts, so the highlight never blinks
    # off mid-line; the line itself keeps the ASR's end.
    first = timeline.lines[0]
    assert first.words[0].end == pytest.approx(first.words[1].start, abs=0.001)
    assert first.end == pytest.approx(2.0, abs=0.001)


def test_a_line_that_was_only_half_matched_keeps_its_timings(synth_audio, tmp_path, monkeypatch):
    # The characters are squeezed into the first half of the line: the aligner
    # matched part of the phrase and stopped, so their positions are not this
    # line's and the word stream's own spans are the better answer.
    def fake_post(payload):
        lo = payload["offset"]
        text = payload["text"]
        return {"chars": _chars([(char, lo + 0.05 + i * 0.7, lo + 0.75 + i * 0.7) for i, char in enumerate(text)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=4.0,
                words=[Word(text="あ", start=1.0, end=2.0), Word(text="い", start=2.0, end=4.0)],
            )
        ],
    )
    before = [(line.start, line.end, [(w.start, w.end) for w in line.words]) for line in timeline.lines]
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    assert [(line.start, line.end, [(w.start, w.end) for w in line.words]) for line in timeline.lines] == before
    assert notes and "没有为任何一句给出结果" in notes[0]


def test_a_line_covered_over_half_way_keeps_the_aligners_timings(synth_audio, tmp_path, monkeypatch):
    # Same shape as the test above, but the aligner covered 1.8 s of the line's
    # 3.0 s: the ASR's line end runs into silence often enough that a line
    # covered this well is normally placed correctly (measured on /root/yrc:
    # 27 of 31 lines a 0.75 gate rejected were closer to the official 逐字 times
    # than the word-stream spread they fell back to), so the looser gate keeps
    # the aligner's answer instead of throwing it away.
    def fake_post(payload):
        lo = payload["offset"]
        text = payload["text"]
        return {"chars": _chars([(char, lo + 0.05 + i * 0.9, lo + 0.95 + i * 0.9) for i, char in enumerate(text)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=4.0,
                words=[Word(text="あ", start=1.0, end=2.0), Word(text="い", start=2.0, end=4.0)],
            )
        ],
    )
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    line = timeline.lines[0]
    assert [word.source for word in line.words] == ["aligner", "aligner"]
    assert line.start == pytest.approx(1.05, abs=0.01)
    assert notes and "强制对齐" in notes[0]


def test_lines_timed_by_the_lyrics_source_are_left_alone(synth_audio, tmp_path, monkeypatch):
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align, "_post", lambda payload: pytest.fail("should not be called"))
    timeline = _timeline()
    for line in timeline.lines:
        line.source = "word-sync"
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    assert notes == []


def test_tightly_spaced_characters_do_not_overlap():
    # Two characters 20 ms apart would be one highlight frame: the later start is
    # pushed forward until it has a frame of its own, so both light up and the
    # highlight never shows two characters at once.
    words = [Word(text="あ", start=1.0, end=1.5), Word(text="い", start=1.02, end=1.5), Word(text="う", start=1.5, end=1.6)]
    char_align._fill(words, 2.0)
    assert words[0].end == pytest.approx(1.04)
    assert words[1].start == pytest.approx(1.04)
    assert words[1].end == pytest.approx(1.5)
    assert words[2].end == pytest.approx(2.0)
    assert all(word.end > word.start for word in words)


def test_retime_skips_a_language_the_service_cannot_handle(synth_audio, tmp_path, monkeypatch):
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align, "_post", lambda payload: pytest.fail("should not be called"))
    notes = char_align.retime(
        _timeline(), audio_path=synth_audio, work_dir=tmp_path, language="Korean"
    )
    assert notes and "只支持" in notes[0]


def test_a_line_the_aligner_could_not_match_keeps_its_timings(synth_audio, tmp_path, monkeypatch):
    def fake_post(payload):
        # Echo unrelated text: this says nothing about the line's timings.
        return {"chars": _chars([("謎", 0.0, 0.5), ("の", 0.5, 0.9)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = _timeline()
    before = [(line.start, line.end) for line in timeline.lines]
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    assert [(line.start, line.end) for line in timeline.lines] == before
    assert notes and "没有为任何一句给出结果" in notes[0]


def test_a_line_the_aligner_left_alone_still_gets_usable_widths(synth_audio, tmp_path, monkeypatch):
    """Its characters keep the word stream's spans, which can overlap each other."""
    def fake_post(payload):
        # This line's alignment lost its tail, so it keeps the word stream's
        # own characters — three of which came from one recognised word.
        return {"chars": _chars([("あ", 0.0, 0.1)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あいう",
                start=1.0,
                end=3.0,
                words=[
                    Word(text="あ", start=1.0, end=3.0),
                    Word(text="い", start=1.0, end=3.0),
                    Word(text="う", start=2.0, end=3.0),
                ],
            )
        ],
    )
    char_align.retime(timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese")
    words = timeline.lines[0].words
    # Stacked starts are spread in order, and nothing overlaps.
    assert [word.start for word in words] == sorted(word.start for word in words)
    for first, second in zip(words, words[1:]):
        assert first.end <= second.start + 0.001
    assert words[-1].end == pytest.approx(3.0, abs=0.001)


class _Recorder:
    """Stands in for ``trace.Trace``: keeps the payloads for assertions."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, dict]] = []

    def step(self, name: str, payload) -> None:
        self.steps.append((name, payload))


def test_the_last_character_carries_the_held_final_vowel(voiced_audio, tmp_path, monkeypatch):
    """The aligner never holds the final vowel; that is not a lost phrase."""
    audio = voiced_audio(20.0)

    def fake_post(payload):
        lo = payload["offset"]
        # Both characters placed, the last one stopping 0.5 s before the line.
        return {"chars": _chars([("あ", lo + 0.05, lo + 1.4), ("い", lo + 1.4, lo + 1.95)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=3.0,
                words=[Word(text="あ", start=1.0, end=2.0), Word(text="い", start=2.0, end=3.0)],
            )
        ],
    )
    notes = char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[0]
    assert [word.source for word in line.words] == ["aligner", "aligner"]
    assert line.words[0].start == pytest.approx(1.05, abs=0.01)
    assert line.words[0].end == pytest.approx(line.words[1].start, abs=0.001)
    # The held vowel is singing: the last character carries it, up to the line end.
    assert line.end == pytest.approx(3.0, abs=0.02)
    assert line.words[-1].end == pytest.approx(line.end, abs=0.001)
    assert "未成功" not in notes[0]


def test_a_line_ends_where_the_singing_stops_not_where_the_asr_said(voiced_audio, tmp_path, monkeypatch):
    """An ASR line end can reach a second into the silence after the singing."""
    audio = voiced_audio(3.0)

    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([("あ", lo + 0.05, lo + 0.95), ("い", lo + 0.95, lo + 1.85)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=5.0,
                words=[Word(text="あ", start=1.0, end=3.0), Word(text="い", start=3.0, end=5.0)],
            )
        ],
    )
    char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[0]
    # The singing stops at 3.0; the last character is not stretched over 2 s of
    # silence just because the ASR put the line end there. The envelope is
    # smoothed over 120 ms, so the tail can be that much late.
    assert line.end == pytest.approx(3.0, abs=0.15)
    assert line.words[-1].end == pytest.approx(line.end, abs=0.001)


def test_a_line_ends_before_the_next_phrase_not_inside_it(voiced_audio, tmp_path, monkeypatch):
    """The ASR's line end can reach into the next phrase; that is not this line's tail.

    The vocal is loud over 1.0–2.0 s and again over 4.0–6.0 s. The ASR puts the
    line at 1.0–4.5 s, so the last loud frame before its end belongs to the
    *next* phrase — stretching the last character to it would hold the line
    highlight through 2.5 s of silence and into singing that is not this line.
    """
    import array
    import wave

    rate = 16000
    total = 8.0
    loud = array.array("h", [0] * int(total * rate))
    for begin, end in ((1.0, 2.0), (4.0, 6.0)):
        for index in range(int(begin * rate), int(end * rate)):
            loud[index] = 9000
    audio = tmp_path / "two-phrases.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(loud.tobytes())

    def fake_post(payload):
        lo = payload["offset"]
        # Both characters placed inside the first phrase.
        return {"chars": _chars([("あ", lo + 0.05, lo + 0.5), ("い", lo + 0.5, lo + 0.95)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=4.5,
                words=[Word(text="あ", start=1.0, end=4.5)],
            )
        ],
    )
    char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[0]
    assert line.words[-1].end == pytest.approx(2.0, abs=0.15)
    assert line.end == pytest.approx(2.0, abs=0.15)


def test_a_clip_is_cut_where_its_phrase_stops(tmp_path):
    """One over-long ASR token must not let the line spread across the silence.

    あいつら L27: the ASR gave one token sixteen seconds (78.48–94.96) while the
    line is sung 77.06–79.11. The clip then holds the phrase and fifteen seconds
    of silence, and the aligner puts the first characters at the head and the
    last ones at the far edge — the line is accepted, nine seconds wrong, and
    its characters are highlighted across the whole silence. Cutting the clip
    where the first phrase stops leaves it nowhere to spread.
    """
    import array
    import wave

    rate = 16000
    total = 100.0
    samples = array.array("h", [0] * int(total * rate))
    for begin, end in ((77.0, 79.1), (94.9, 97.0)):
        for index in range(int(begin * rate), int(end * rate)):
            samples[index] = 9000
    audio = tmp_path / "long-token.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    line = Line(text="あい", start=78.48, end=94.96)
    lo, hi = char_align._window([line], 0, line, char_align.QuietTrack(audio))
    assert lo == pytest.approx(78.48, abs=0.05)
    assert hi == pytest.approx(79.1, abs=0.15)


def test_a_breath_inside_a_line_does_not_cut_the_clip(tmp_path):
    """A short pause is part of the line; only a phrase break ends the clip.

    The singer takes a breath mid-phrase all the time. Cutting at every silence
    would throw away the rest of the line — measured: a 1.2 s cut costs Dear
    Mr 「F」 0.11 → 0.17 s per character.
    """
    import array
    import wave

    rate = 16000
    total = 10.0
    samples = array.array("h", [0] * int(total * rate))
    for begin, end in ((1.0, 2.0), (2.5, 4.0)):
        for index in range(int(begin * rate), int(end * rate)):
            samples[index] = 9000
    audio = tmp_path / "one-breath.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    line = Line(text="あい", start=1.0, end=4.0)
    _lo, hi = char_align._window([line], 0, line, char_align.QuietTrack(audio))
    assert hi > 3.5


def test_a_line_that_starts_before_its_phrase_is_not_cut_at_the_break(tmp_path):
    """The ASR can start a line before the phrase it is sung in; that head is silence.

    土星 L6: 24 characters, the ASR started the line early, and the only singing
    before the phrase break is 0.46 s. Cutting there (the two-phrase rule) leaves
    the aligner a window it cannot place 24 characters in — score 0.67, coverage
    0.06 — while the untruncated clip places the whole line at 0.95. A head too
    short to hold one ``MIN_WORD_SEC`` per character is not this line's phrase.
    """
    import array
    import wave

    rate = 16000
    total = 12.0
    samples = array.array("h", [0] * int(total * rate))
    for begin, end in ((1.0, 1.46), (2.8, 5.0)):
        for index in range(int(begin * rate), int(end * rate)):
            samples[index] = 9000
    audio = tmp_path / "early-start.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    line = Line(text="あ" * 24, start=1.0, end=5.0)
    _lo, hi = char_align._window([line], 0, line, char_align.QuietTrack(audio))
    assert hi > 4.9

    # The same audio with a short line: the 0.46 s head can hold two characters,
    # so the two-phrase rule still applies and the clip stops at the break.
    short = Line(text="あい", start=1.0, end=5.0)
    _lo, hi = char_align._window([short], 0, short, char_align.QuietTrack(audio))
    assert hi == pytest.approx(1.46, abs=0.15)


def test_a_line_the_asr_collapsed_grows_back_to_its_phrase(voiced_audio, tmp_path, monkeypatch):
    """A collapsed line leaves no clip; the phrase it belongs to is still there.

    The ASR sometimes gives a line a zero-length span whose start equals the
    next line's start (measured: 14 lines across the six songs in ``/root/yrc``).
    The clip is then empty and the line used to fall back to the word stream —
    the worst path measured. The vocal is loud over 4.0–6.0 s, the line sits at
    6.0–6.0 s, and the next line starts at 6.0 s too, so the window must grow
    back to the start of that loud run.
    """
    import array
    import wave

    rate = 16000
    total = 10.0
    samples = array.array("h", [0] * int(total * rate))
    for index in range(int(4.0 * rate), int(6.0 * rate)):
        samples[index] = 9000
    audio = tmp_path / "one-phrase.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    calls = []

    def fake_post(payload):
        calls.append(payload["offset"])
        lo = payload["offset"]
        return {"chars": _chars([("あ", lo + 0.05, lo + 0.9), ("い", lo + 0.9, lo + 1.9)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", False)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="うえ", start=1.0, end=2.0, words=[Word(text="うえ", start=1.0, end=2.0)]),
            Line(text="あい", start=6.0, end=6.0, words=[Word(text="あい", start=6.0, end=6.0)]),
            Line(text="おか", start=6.0, end=8.0, words=[Word(text="おか", start=6.0, end=8.0)]),
        ],
    )
    char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[1]
    assert len(calls) >= 2, "the collapsed line must still be aligned"
    assert calls[1] == pytest.approx(4.0, abs=0.15)
    assert line.start == pytest.approx(4.05, abs=0.15)
    assert line.words[0].source == "aligner"


def test_a_collapsed_line_starts_where_the_aligner_scores_it(tmp_path, monkeypatch):
    """Growing to the loud run's start is a guess; the aligner can score a start.

    我的故事 L11: the ASR gives 85.76–85.92 s (and 85.92 is the next line's start),
    while the phrase is sung from 80.12 s. The loud run the window grows into
    starts at 85.84 s — a breath at 85.75 stops the walk — so the clip holds
    only the phrase's tail. Starting the same clip further back scores 1.00 and
    places the line on the phrase. The scan must find that start, and it must
    keep the *earliest* of the tied starts: the score is flat across a phrase.
    """
    import array
    import wave

    rate = 16000
    total = 95.0
    samples = array.array("h", [0] * int(total * rate))
    # Loud only from 81.0 s: the phrase's head is softer than the level that
    # decides where a loud run starts, so the run the window grows into begins
    # in the middle of the line, not at its first syllable.
    for index in range(int(81.0 * rate), int(82.1 * rate)):
        samples[index] = 9000
    audio = tmp_path / "collapsed.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    def fake_post(payload):
        # A clip that starts at the phrase's head answers with the whole line;
        # one that starts later has lost it and places the line on nothing.
        lo = payload["offset"]
        if lo <= 79.3:
            return {"chars": _chars([("あ", 79.2, 80.6), ("い", 80.6, 82.0)])}
        return {"chars": _chars([("あ", lo + 0.05, lo + 0.1), ("い", lo + 0.1, lo + 0.15)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="うえ", start=78.0, end=79.0, words=[Word(text="うえ", start=78.0, end=79.0)]),
            Line(text="あい", start=82.0, end=82.16, words=[Word(text="あい", start=82.0, end=82.16)]),
            Line(text="おか", start=82.16, end=86.0, words=[Word(text="おか", start=82.16, end=86.0)]),
        ],
    )
    trace = _Recorder()
    char_align.retime(
        timeline, audio_path=audio, work_dir=tmp_path, language="Japanese", trace=trace
    )
    line = timeline.lines[1]
    assert line.words[0].source == "aligner", "the collapsed line must be aligned, not kept"
    assert line.start == pytest.approx(79.2, abs=0.4)
    windows = {entry["line"]: entry for entry in dict(trace.steps)["char_align"]["windows"]}
    assert windows[2]["scanned"] == pytest.approx(79.0, abs=0.6)
    # The next line must not be dragged back onto the same phrase.
    assert timeline.lines[2].start >= 82.1


def test_a_line_placed_seconds_late_is_found_by_scanning_back(tmp_path, monkeypatch):
    """The shifted search reaches ±1.5 s; an ASR line 3.6 s late needs more.

    Dear Mr 「F」 L35 is given 187.12–195.04 s while the phrase is sung
    183.5–191.9. Every shifted window still starts inside the silence after the
    phrase's head, the aligner places only the first two characters, and the
    coverage gate throws the line away. Started where the phrase does, the same
    clip scores 0.84 with 0.68 coverage. The rescue must take that window and
    leave a window that merely scores a little higher with low coverage alone.
    """
    import array
    import wave

    rate = 16000
    total = 200.0
    samples = array.array("h", [0] * int(total * rate))
    for index in range(int(183.5 * rate), int(192.0 * rate)):
        samples[index] = 9000
    audio = tmp_path / "late.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    def fake_post(payload):
        lo = payload["offset"]
        if lo <= 183.6:
            return {"chars": _chars([("あ", 183.6, 187.0), ("い", 187.0, 191.8)])}
        # Late windows match only the phrase's tail: the line is squeezed into
        # the first 1.8 s of a 7.9 s line, which the coverage gate rejects.
        return {"chars": _chars([("あ", lo + 0.05, lo + 0.9), ("い", lo + 0.9, lo + 1.8)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="うえ", start=178.8, end=187.12, words=[Word(text="うえ", start=178.8, end=187.12)]),
            Line(text="あい", start=187.12, end=195.04, words=[Word(text="あい", start=187.12, end=195.04)]),
            Line(text="おか", start=195.04, end=198.0, words=[Word(text="おか", start=195.04, end=198.0)]),
        ],
    )
    notes = char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[1]
    assert line.words[0].source == "aligner", "the late line must be rescued, not kept"
    assert line.start == pytest.approx(183.6, abs=0.7)
    assert any("把片段起点往前找过" in note for note in notes)


def test_a_line_left_with_no_clip_is_found_by_scanning_back(tmp_path, monkeypatch):
    """A collapsed line can leave a clip too short to send; the scan still finds it.

    我的故事 L11 is given 85.76–85.92 s, the next line starts at 85.92, and the
    only loud frame just before that point is a 0.22 s breath — so growing the
    window back to the loud run's start finds 85.90 s and the clip is still
    0.16 s long. The phrase itself is sung from 78.0 s. The line must be found by
    the scan rather than handed to the word stream as "too short".
    """
    import array
    import wave

    rate = 16000
    total = 95.0
    samples = array.array("h", [0] * int(total * rate))
    for index in range(int(78.0 * rate), int(84.0 * rate)):
        samples[index] = 9000
    # The breath right before the ASR's point: long enough to be a loud frame,
    # far too short to be a phrase the window can grow into.
    for index in range(int(85.54 * rate), int(85.76 * rate)):
        samples[index] = 9000
    audio = tmp_path / "point-only.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    def fake_post(payload):
        lo = payload["offset"]
        if payload["text"] != "あい":
            return {"chars": _chars([("う", 74.2, 75.0), ("え", 75.0, 76.0)])}
        if lo <= 80.5:
            return {"chars": _chars([("あ", 78.0, 81.0), ("い", 81.0, 84.0)])}
        # A window starting inside the phrase's tail hears almost nothing.
        return {"chars": _chars([("あ", lo + 0.02, lo + 0.05), ("い", lo + 0.05, lo + 0.08)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="うえ", start=74.0, end=84.0, words=[Word(text="うえ", start=74.0, end=84.0)]),
            Line(text="あい", start=85.76, end=85.92, words=[Word(text="あい", start=85.76, end=85.92)]),
            Line(text="おか", start=85.92, end=90.0, words=[Word(text="おか", start=85.92, end=90.0)]),
        ],
    )
    trace = _Recorder()
    notes = char_align.retime(
        timeline, audio_path=audio, work_dir=tmp_path, language="Japanese", trace=trace
    )
    line = timeline.lines[1]
    assert line.words[0].source == "aligner", "the point-only line must be aligned"
    assert line.start == pytest.approx(78.0, abs=0.6)
    assert line.end == pytest.approx(84.0, abs=0.6)
    windows = {entry["line"]: entry for entry in dict(trace.steps)["char_align"]["windows"]}
    assert "scanned" in windows[2]
    assert any("把片段起点往前找过" in note for note in notes)


def test_a_line_with_no_singing_is_reported_as_such(synth_audio, tmp_path, monkeypatch):
    """A hallucinated line over an instrumental intro is not an alignment failure."""
    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([("は", lo + 0.05, lo + 0.1), ("い", lo + 0.1, lo + 0.13)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[Line(text="はい", start=6.88, end=18.0, words=[Word(text="はい", start=6.88, end=18.0)])],
    )
    trace = _Recorder()
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese", trace=trace
    )
    # The note can only say nothing was placed; the debug page carries the why.
    assert notes and "没有为任何一句给出结果" in notes[0]
    payload = dict(trace.steps)["char_align"]
    assert "几乎对不上人声" in payload["windows"][0]["reason"]


def test_an_extra_line_with_no_singing_is_dropped(synth_audio, tmp_path, monkeypatch):
    """An extra is speculative: if nothing is sung there, the line is a hallucination."""
    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([("は", lo + 0.05, lo + 0.08), ("い", lo + 0.08, lo + 0.1)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="はい", start=6.88, end=18.0, source="asr-extra",
                 words=[Word(text="はい", start=6.88, end=18.0)]),
            Line(text="夜に浮かんでいた", start=33.56, end=36.64,
                 words=[Word(text="夜", start=33.56, end=36.64)]),
        ],
    )

    def per_line(payload):
        lo = payload["offset"]
        if payload["text"] == "はい":
            return {"chars": _chars([("は", lo + 0.05, lo + 0.08), ("い", lo + 0.08, lo + 0.1)])}
        text = payload["text"]
        step = 2.9 / len(text)
        return {"chars": _chars([(c, lo + 0.05 + i * step, lo + 0.05 + (i + 1) * step)
                                 for i, c in enumerate(text)])}

    monkeypatch.setattr(char_align, "_post", per_line)
    notes = char_align.retime(timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese")
    # The hallucinated line is gone; the lyric line is timed by the aligner.
    assert [line.text for line in timeline.lines] == ["夜に浮かんでいた"]
    assert all(word.source == "aligner" for word in timeline.lines[0].words)
    assert "已删掉" in notes[0]
    assert "はい" in notes[0]


def _clean_chars(text, lo, step=0.5):
    return _chars([(c, lo + 0.05 + i * step, lo + 0.05 + (i + 1) * step) for i, c in enumerate(text)])


def test_a_line_squeezed_into_the_head_of_its_clip_moves_with_the_clip(tmp_path, monkeypatch):
    """A clip that stops before the singing does squeezes the line into its head.

    The ASR gives 100–110 s and the phrase is sung 106–116 s, but the aligner
    only hears the part of the clip that holds singing, so its answer covers 3 s
    of the 10 s the line is sung over and the coverage gate rejects it. Moving
    the start inside the clip cannot help — coverage is capped by the clip's own
    length — so the whole window has to move. The window at 106 s passes the
    gates, and the line must be taken from it instead of falling back to the
    word stream.
    """
    import array
    import wave

    rate = 16000
    total = 130.0
    samples = array.array("h", [0] * int(total * rate))
    for index in range(int(100.0 * rate), int(119.0 * rate)):
        samples[index] = 9000
    audio = tmp_path / "squeezed.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())

    calls = []

    def fake_post(payload):
        lo = payload["offset"]
        calls.append(lo)
        if lo >= 105.9:
            # Where the phrase is: the whole line, over the line's own span.
            return {"chars": _chars([("あ", lo + 0.05, lo + 3.35),
                                     ("い", lo + 3.35, lo + 6.65),
                                     ("う", lo + 6.65, lo + 9.95)])}
        # Anywhere earlier the aligner only matches the clip's head.
        return {"chars": _chars([("あ", lo + 0.05, lo + 1.05),
                                 ("い", lo + 1.05, lo + 2.05),
                                 ("う", lo + 2.05, lo + 3.05)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="えお", start=90.0, end=100.0, words=[Word(text="えお", start=90.0, end=100.0)]),
            Line(text="あいう", start=100.0, end=110.0,
                 words=[Word(text="あいう", start=100.0, end=110.0)]),
            Line(text="かき", start=120.0, end=126.0, words=[Word(text="かき", start=120.0, end=126.0)]),
        ],
    )
    notes = char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[1]
    assert any(lo >= 105.9 for lo in calls), calls
    assert line.words[0].source == "aligner", "the squeezed line must be rescued, not kept"
    assert line.start == pytest.approx(106.05, abs=0.2)
    assert "识别放偏了" in notes[0]


def _guessing_chars(text, lo):
    """What a wrong window looks like: every character interpolated, no width."""
    return [{"c": c, "start": lo + 0.05, "end": lo + 0.06, "interpolated": True} for c in text]


def test_a_line_the_asr_placed_on_the_wrong_stretch_is_moved(voiced_audio, tmp_path, monkeypatch):
    """The ASR says 1.0; the line is really sung at 2.2. The aligner finds it."""
    audio = voiced_audio(20.0)
    truth = 2.2
    calls = []

    def fake_post(payload):
        lo = payload["offset"]
        calls.append(lo)
        near = abs(lo - truth) < 0.25
        # A window that holds the line: every character placed, over the whole span.
        chars = (
            _clean_chars(payload["text"], lo, step=1.0)
            if near
            else _guessing_chars(payload["text"], lo)
        )
        return {"chars": chars}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(
                text="あい",
                start=1.0,
                end=3.0,
                words=[Word(text="あ", start=1.0, end=2.0), Word(text="い", start=2.0, end=3.0)],
            )
        ],
    )
    notes = char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    line = timeline.lines[0]
    assert calls[0] == pytest.approx(1.0, abs=0.01)
    assert any(abs(lo - truth) < 0.01 for lo in calls[1:]), calls
    assert line.words[0].start == pytest.approx(truth + 0.05, abs=0.01)
    assert "识别放偏了" in notes[0]


def test_a_clean_line_is_aligned_once(voiced_audio, tmp_path, monkeypatch):
    """No search unless the first window is doubtful: one request per good line."""
    audio = voiced_audio(20.0)
    calls = []

    def fake_post(payload):
        calls.append(payload["offset"])
        return {"chars": _clean_chars(payload["text"], payload["offset"], step=1.0)}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[Line(text="あい", start=1.0, end=3.0, words=[Word(text="あい", start=1.0, end=3.0)])],
    )
    char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    assert len(calls) == 1


def test_the_search_never_reaches_into_the_next_line(voiced_audio, tmp_path, monkeypatch):
    """Two lines cannot be moved onto the same stretch of audio."""
    audio = voiced_audio(20.0)
    calls = []

    def fake_post(payload):
        calls.append((payload["text"], payload["offset"]))
        return {"chars": _guessing_chars(payload["text"], payload["offset"])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", True)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="あい", start=1.0, end=2.0, words=[Word(text="あい", start=1.0, end=2.0)]),
            Line(text="うえ", start=2.4, end=3.4, words=[Word(text="うえ", start=2.4, end=3.4)]),
        ],
    )
    char_align.retime(timeline, audio_path=audio, work_dir=tmp_path, language="Japanese")
    # Every window tried for the first line begins before the next line starts:
    # that stretch belongs to the next line and cannot be taken from it.
    windows = [lo for text, lo in calls if text == "あい"]
    assert len(windows) > 1
    assert all(lo < 2.4 for lo in windows), windows


def _fuse_setup(monkeypatch, second):
    """Make the second-opinion pass available and return fixed tokens."""
    from app.services import char_fuse

    monkeypatch.setattr(char_fuse, "available", lambda: True)
    monkeypatch.setattr(char_fuse, "second_words", lambda *a, **k: [Word(**w) for w in second])


def test_a_character_both_backends_agree_on_takes_the_recognisers_time(
    synth_audio, tmp_path, monkeypatch
):
    """The ASR word stream is the per-character source; agreement is what licenses it."""

    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([(c, lo + 0.1 + i * 0.4, lo + 0.5 + i * 0.4) for i, c in enumerate(payload["text"])])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    # The recogniser says each character is 0.1 s later than the aligner does:
    # inside the tolerance, so the recogniser's own boundaries are kept.
    _fuse_setup(
        monkeypatch,
        [
            {"text": "あ", "start": 1.2, "end": 1.5, "source": "asr"},
            {"text": "い", "start": 1.6, "end": 1.9, "source": "asr"},
        ],
    )
    timeline = _timeline()
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    first = timeline.lines[0]
    assert first.words[0].start == pytest.approx(1.2, abs=0.01)
    assert first.words[1].start == pytest.approx(1.6, abs=0.01)
    # ... and each character says where its time came from.
    assert [word.source for word in first.words] == ["asr", "asr"]
    assert any("识别词流" in note for note in notes)


def test_a_character_the_recogniser_only_interpolated_keeps_the_aligner(
    synth_audio, tmp_path, monkeypatch
):
    """An interpolated time is not a measurement: it cannot displace a real one."""

    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([(c, lo + 0.1 + i * 0.4, lo + 0.5 + i * 0.4) for i, c in enumerate(payload["text"])])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    # The recogniser heard only the first character: the second is interpolated
    # between its neighbours, which is not a measurement of anything.
    _fuse_setup(
        monkeypatch,
        [
            {"text": "あ", "start": 1.2, "end": 1.5, "source": "asr"},
        ],
    )
    timeline = _timeline()
    char_align.retime(timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese")
    first = timeline.lines[0]
    assert first.words[0].start == pytest.approx(1.2, abs=0.01)
    assert first.words[1].start == pytest.approx(1.5, abs=0.01)
    assert first.words[1].source == "aligner"


def test_a_character_the_backends_disagree_on_keeps_the_aligner(synth_audio, tmp_path, monkeypatch):
    """Disagreement is evidence one of them is on the wrong stretch, not a tie."""

    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([(c, lo + 0.1 + i * 0.4, lo + 0.5 + i * 0.4) for i, c in enumerate(payload["text"])])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    # The second backend is 2 s late on every character: outside the tolerance.
    _fuse_setup(
        monkeypatch,
        [
            {"text": "あ", "start": 3.1, "end": 3.4, "source": "asr"},
            {"text": "い", "start": 3.5, "end": 3.8, "source": "asr"},
        ],
    )
    timeline = _timeline()
    char_align.retime(timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese")
    first = timeline.lines[0]
    assert first.words[0].start == pytest.approx(1.1, abs=0.01)
    assert first.words[1].start == pytest.approx(1.5, abs=0.01)


def test_a_recogniser_that_stacks_two_characters_still_leaves_them_visible(
    synth_audio, tmp_path, monkeypatch
):
    """Adopting the recogniser's boundaries can put neighbours milliseconds apart.

    The aligner separates あ and い by 0.06 s; the recogniser heard both at 1.04 s.
    Adopting its boundaries is the point — but two characters on one instant are
    one highlight block, so the same spreading an aligned line gets has to run
    here too (土星: 24 of 411 characters below 20 ms, most on fused lines).
    """

    def fake_post(payload):
        lo = payload["offset"]
        return {"chars": _chars([("あ", lo + 0.0, lo + 0.03), ("い", lo + 0.06, lo + 0.9)])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    _fuse_setup(
        monkeypatch,
        [
            {"text": "あ", "start": 1.04, "end": 1.05, "source": "asr"},
            {"text": "い", "start": 1.04, "end": 1.9, "source": "asr"},
        ],
    )
    timeline = _timeline()
    notes = char_align.retime(
        timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese"
    )
    first = timeline.lines[0]
    assert any("识别词流" in note for note in notes)
    assert first.words[1].start - first.words[0].start >= char_align.MIN_WORD_SEC
    assert first.words[0].end > first.words[0].start
    assert first.words[1].end > first.words[1].start


def test_fusion_leaves_a_line_the_aligner_could_not_place_alone(synth_audio, tmp_path, monkeypatch):
    """A kept line is on the word stream's spread, not on a measurement."""

    def fake_post(payload):
        # Half the text echoed back: the line is rejected and keeps its times.
        text = payload["text"]
        lo = payload["offset"]
        return {"chars": _chars([(c, lo + 0.05, lo + 0.1) for c in text[:1]])}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    _fuse_setup(
        monkeypatch,
        [
            {"text": "あ", "start": 1.9, "end": 2.0, "source": "asr"},
            {"text": "い", "start": 2.0, "end": 2.1, "source": "asr"},
        ],
    )
    timeline = _timeline()
    char_align.retime(timeline, audio_path=synth_audio, work_dir=tmp_path, language="Japanese")
    assert [(w.start, w.end) for w in timeline.lines[0].words] == [(1.0, 1.5), (1.5, 2.0)]


def test_the_second_opinion_is_cached_per_audio(monkeypatch, tmp_path):
    """A cache from another audio (another take, another stem) is not reused."""
    from app.services import char_fuse

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not really audio")
    calls = []

    class FakeProvider:
        def available(self):
            return True

        def with_align(self, align):
            return self

        async def transcribe(self, path, language=None):
            from app.services.providers.base import ASRResult

            calls.append(path.name)
            return ASRResult(words=[Word(text="あ", start=0.0, end=0.5, source="asr")])

    monkeypatch.setattr(char_fuse.settings, "char_fuse", True)
    monkeypatch.setattr(char_fuse.settings, "asr_url", "http://asr")
    monkeypatch.setattr(char_fuse.registry, "asr", lambda name=None: FakeProvider())
    monkeypatch.setattr(char_fuse, "probe_audio", lambda path: type("I", (), {"duration": 0.0})())
    monkeypatch.setattr(char_fuse, "cut_audio", lambda source, target, start, length: target)

    words = char_fuse.second_words(audio, tmp_path, "Japanese")
    assert [word.text for word in words] == ["あ"]
    assert calls

    # Same audio: the cache answers, no second pass.
    calls.clear()
    assert char_fuse.second_words(audio, tmp_path, "Japanese")
    assert calls == []

    # Another audio under the same name: the cached times do not describe it.
    audio.write_bytes(b"different audio")
    assert char_fuse.second_words(audio, tmp_path, "Japanese")
    assert calls

    # Another language: same story.
    calls.clear()
    char_fuse.second_words(audio, tmp_path, "Chinese")
    assert calls


def test_a_token_covering_two_characters_is_split_between_them():
    """No character is merged into a multi-character time slot."""
    words = char_align._per_character(
        [
            Word(text="この", start=7.0, end=7.4, source="asr"),
            Word(text="に", start=7.4, end=7.6, source="asr"),
        ]
    )
    assert [(word.text, word.start, word.end) for word in words] == [
        ("こ", 7.0, 7.2),
        ("の", 7.2, 7.4),
        ("に", 7.4, 7.6),
    ]
    assert all(word.source == "asr" for word in words)


def test_a_latin_run_stays_one_token():
    """The official 逐字 timings keep latin words whole; so do we."""
    words = char_align._per_character([Word(text="Hip ", start=102.71, end=103.1, source="asr")])
    assert [(word.text, word.start, word.end) for word in words] == [("Hip ", 102.71, 103.1)]


def test_trailing_punctuation_stays_with_its_character():
    words = char_align._per_character([Word(text="あい、", start=0.0, end=0.6, source="asr")])
    assert [word.text for word in words] == ["あ", "い、"]
    assert words[-1].end == 0.6


def test_kana_lyrics_are_aligned_as_japanese_whatever_the_label(voiced_audio, tmp_path, monkeypatch):
    """A project labelled Chinese whose lyrics are kana is a Japanese song.

    土星 is sung in Japanese and was labelled Chinese from its title (the
    Chinese reading of the same characters). Asked to align Japanese as
    Chinese the aligner does not place the text badly, it refuses it: on that
    song 17/28 lines placed under ``zh`` against 25/28 under ``ja``, and the
    characters it did place were crammed (57 below 20 ms against 23). Kana
    only ever appear in Japanese, so the line's own text is the evidence.
    """
    audio = voiced_audio(20.0)
    langs = []

    def fake_post(payload):
        langs.append(payload["lang"])
        return {"chars": _clean_chars(payload["text"], payload["offset"], step=0.5)}

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_align.settings, "char_aligner_url", "http://aligner")
    monkeypatch.setattr(char_align.settings, "char_align_search", False)
    timeline = Timeline(
        status="none",
        lines=[
            Line(text="君の心", start=1.0, end=2.5, words=[Word(text="君の心", start=1.0, end=2.5)]),
            Line(text="看不见", start=3.0, end=4.0, words=[Word(text="看不见", start=3.0, end=4.0)]),
        ],
    )
    trace = _Recorder()
    char_align.retime(
        timeline, audio_path=audio, work_dir=tmp_path, language="Chinese", trace=trace
    )
    # The kana line goes out as Japanese; a line with no kana keeps the label.
    assert langs == ["ja", "zh"], langs
    assert dict(trace.steps)["char_align"]["japanese_lines"] == 1
