"""Per-character times from the recogniser alone: the aligner off, which is the default.

``retime`` has two sources. The forced aligner placing every character is one of
them (``hfa_chars`` on); the recogniser's own per-character table is the other,
and it is the default — no aligner request is made at all. These cases cover the
default, so a change that quietly goes back to requiring the aligner fails here.
"""

from __future__ import annotations

import pytest

from app.schemas import Line, Timeline, Word
from app.services import char_align, char_fuse


@pytest.fixture(autouse=True)
def recogniser_available(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "char_hfa", False)
    monkeypatch.setattr(settings, "char_aligner_url", "http://aligner.test")
    monkeypatch.setattr(settings, "asr_url", "http://asr.test")


def _timeline():
    return Timeline(
        status="none",
        lines=[
            Line(
                text="きみの",
                start=1.0,
                end=2.0,
                words=[
                    Word(text="き", start=1.0, end=1.3, source="asr"),
                    Word(text="み", start=1.3, end=1.6, source="asr"),
                    Word(text="の", start=1.6, end=2.0, source="asr"),
                ],
            ),
        ],
    )


def test_aligner_is_not_called_at_all_by_default(monkeypatch, tmp_path):
    """Off is the default: not one request goes to the aligner."""

    def must_not_call(*_args, **_kwargs):
        raise AssertionError("the aligner was asked for per-character times while it is off")

    monkeypatch.setattr(char_align, "_post", must_not_call)
    monkeypatch.setattr(char_align, "align_text", must_not_call)
    monkeypatch.setattr(char_fuse, "second_words", lambda *_a, **_k: [])
    notes = char_align.retime(
        _timeline(), audio_path=tmp_path / "a.wav", work_dir=tmp_path, language="Japanese"
    )
    # No tokens either: the note has to say the times are estimates, and it
    # must not claim an alignment happened.
    assert notes and "识别器" in notes[0]


def test_the_recognisers_tokens_are_the_per_character_times(monkeypatch, tmp_path):
    tokens = [
        Word(text="きみ", start=0.9, end=1.4, source="asr"),
        Word(text="の", start=1.5, end=1.9, source="asr"),
    ]
    seen: dict[str, object] = {}

    def fake_second_words(audio_path, work_dir, language, *, on_progress=None, honour_switch=True):
        seen["honour_switch"] = honour_switch
        return tokens

    monkeypatch.setattr(char_fuse, "second_words", fake_second_words)
    timeline = _timeline()
    notes = char_align.retime(
        timeline, audio_path=tmp_path / "a.wav", work_dir=tmp_path, language="Japanese"
    )

    # The fusion switch does not apply here: this table is not a second opinion,
    # it is the only one, so the pass asks for it regardless.
    assert seen["honour_switch"] is False
    words = timeline.lines[0].words
    assert "".join(word.text for word in words) == "きみの"
    # ``きみ`` came back as one token for two characters, and each still gets a
    # time of its own — one highlight block per character, as required.
    assert len(words) == 3
    assert all(word.start is not None and word.end is not None for word in words)
    assert all(word.end - word.start >= char_align.MIN_WORD_SEC for word in words)
    assert [word.start for word in words] == sorted(word.start for word in words)
    assert notes and "识别器" in notes[0]


def test_a_line_the_recogniser_never_heard_keeps_its_word_stream_spans(monkeypatch, tmp_path):
    monkeypatch.setattr(char_fuse, "second_words", lambda *_a, **_k: [])
    timeline = _timeline()
    notes = char_align.retime(
        timeline, audio_path=tmp_path / "a.wav", work_dir=tmp_path, language="Japanese"
    )
    # Nothing heard over the line at all: the times are still there (the word
    # stream's), and the note says they are not a measurement.
    assert len(timeline.lines[0].words) == 3
    assert notes and "识别器" in notes[0]


def test_a_word_synced_line_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(
        char_fuse,
        "second_words",
        lambda *_a, **_k: [Word(text="きみの", start=0.1, end=0.6, source="asr")],
    )
    timeline = _timeline()
    timeline.lines[0].source = "word-sync"
    before = [(word.text, word.start, word.end) for word in timeline.lines[0].words]
    char_align.retime(
        timeline, audio_path=tmp_path / "a.wav", work_dir=tmp_path, language="Japanese"
    )
    # The lyrics source's own 逐字 times passed the gate; a recogniser cannot
    # improve a verified measurement.
    assert [(word.text, word.start, word.end) for word in timeline.lines[0].words] == before


def test_a_language_the_recogniser_cannot_read_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(char_fuse, "second_words", lambda *_a, **_k: [])
    notes = char_align.retime(
        _timeline(), audio_path=tmp_path / "a.wav", work_dir=tmp_path, language="Korean"
    )
    assert notes and "韩" not in notes[0] and "Korean" in notes[0]


def test_switching_the_aligner_on_asks_it_for_each_character(monkeypatch, tmp_path):
    def fake_post(payload):
        lo = payload["offset"]
        return {
            "chars": [
                {"c": char, "start": lo + 0.05 + i * 0.4, "end": lo + 0.4 + i * 0.4}
                for i, char in enumerate(payload["text"])
            ]
        }

    monkeypatch.setattr(char_align, "_post", fake_post)
    monkeypatch.setattr(char_fuse, "second_words", lambda *_a, **_k: [])
    timeline = _timeline()
    char_align.retime(
        timeline,
        audio_path=_wav(tmp_path),
        work_dir=tmp_path,
        language="Japanese",
        use_hfa=True,
    )
    # Placed by the aligner, so the starts are its own (offset + 0.05 ...).
    assert timeline.lines[0].words[0].start == pytest.approx(1.05, abs=0.01)


def _wav(tmp_path):
    import wave

    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000 * 20)
    return path
