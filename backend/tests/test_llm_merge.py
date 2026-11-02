"""The last step: the LLM says where the words end, the characters keep the times.

It is asked for boundaries only, never for a time. Whatever it answers has to
tile the line exactly (every character accounted for, in order, none invented);
anything else is discarded and the line stays per-character. That is the whole
safety property of this pass, so most of these cases are about a bad answer
leaving the timeline alone.
"""

from __future__ import annotations

import pytest

from app.schemas import Line, Timeline, Word
from app.services import llm_merge
from app.services.llm import LLMClient, LLMError


def _line(text: str = "君の心が見えない") -> Line:
    chars = list(text)
    words = [
        Word(text=char, start=round(1.0 + index * 0.25, 3), end=round(1.25 + index * 0.25, 3), source="asr")
        for index, char in enumerate(chars)
    ]
    return Line(text=text, start=words[0].start, end=words[-1].end, words=words)


class FakeLLM(LLMClient):
    def __init__(self, answer):
        super().__init__(base_url="http://llm.test", api_key="", model="test-model")
        self.answer = answer
        self.prompts: list[str] = []

    def chat_json(self, prompt: str, *, max_tokens: int = 1500):
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_the_units_take_the_span_of_the_characters_they_cover():
    timeline = Timeline(status="none", lines=[_line()])
    notes = llm_merge.apply(timeline, client=FakeLLM({"1": ["君の", "心が", "見えない"]}))

    words = timeline.lines[0].words
    assert [word.text for word in words] == ["君の", "心が", "見えない"]
    # 君の covers characters 0–1 (1.00 → 1.50): first start to last end.
    assert words[0].start == pytest.approx(1.0)
    assert words[0].end == pytest.approx(1.5)
    # 心が covers 2–3 (1.50 → 2.00), 見えない covers 4–7 (2.00 → 3.00).
    assert words[1].start == pytest.approx(1.5)
    assert words[2].end == pytest.approx(3.0)
    # Contiguous: no dead air between units, no overlap.
    assert words[0].end == pytest.approx(words[1].start)
    assert words[1].end == pytest.approx(words[2].start)
    # The line still spans what it did.
    assert timeline.lines[0].start == pytest.approx(1.0)
    assert timeline.lines[0].end == pytest.approx(3.0)
    # Each unit says its time is a span, not a measurement of its own.
    assert all(word.source == "merged" for word in words)
    assert notes and "按语意合并" in notes[0]


def test_units_that_do_not_tile_the_line_are_discarded():
    timeline = Timeline(status="none", lines=[_line()])
    before = [(word.text, word.start, word.end) for word in timeline.lines[0].words]
    # "心" is missing: concatenating the units would drop a character, which
    # would silently move the highlight onto the wrong text.
    notes = llm_merge.apply(timeline, client=FakeLLM({"1": ["君の", "が見えない"]}))

    assert [(word.text, word.start, word.end) for word in timeline.lines[0].words] == before
    assert notes and "没有给出可用的语意切分" in notes[0]


def test_an_invented_character_is_discarded():
    timeline = Timeline(status="none", lines=[_line()])
    before = [word.text for word in timeline.lines[0].words]
    llm_merge.apply(timeline, client=FakeLLM({"1": ["君の", "心が", "見えないよ"]}))
    assert [word.text for word in timeline.lines[0].words] == before


def test_one_unit_for_the_whole_line_is_not_a_merge():
    """A single unit lights the whole line at once — coarser than what it had."""
    timeline = Timeline(status="none", lines=[_line()])
    before = [word.text for word in timeline.lines[0].words]
    llm_merge.apply(timeline, client=FakeLLM({"1": ["君の心が見えない"]}))
    assert [word.text for word in timeline.lines[0].words] == before


def test_a_failed_request_leaves_every_line_alone():
    timeline = Timeline(status="none", lines=[_line()])
    before = [word.text for word in timeline.lines[0].words]
    notes = llm_merge.apply(timeline, client=FakeLLM(LLMError("boom")))
    assert [word.text for word in timeline.lines[0].words] == before
    assert notes and "没有给出可用的语意切分" in notes[0]


def test_whitespace_does_not_have_to_match():
    """Latin lyrics are spaced in the text and not in the units (or the reverse)."""
    timeline = Timeline(status="none", lines=[_line("hey you")])
    llm_merge.apply(timeline, client=FakeLLM({"1": ["hey you"]}))
    # One unit for the whole line: nothing to merge, kept per character.
    assert len(timeline.lines[0].words) == len("hey you")

    timeline = Timeline(status="none", lines=[_line("hey you")])
    llm_merge.apply(timeline, client=FakeLLM({"1": ["hey", "you"]}))
    assert [word.text for word in timeline.lines[0].words] == ["hey", "you"]


def test_lines_without_times_or_too_short_are_not_sent():
    llm = FakeLLM({})
    untimed = Line(text="君の", start=None, end=None, words=[Word(text="君"), Word(text="の")])
    single = _line("あ")
    timeline = Timeline(status="none", lines=[untimed, single])
    notes = llm_merge.apply(timeline, client=llm)
    assert llm.prompts == []
    assert notes == []


def test_switching_it_off_makes_no_request():
    llm = FakeLLM({"1": ["君の", "心が", "見えない"]})
    timeline = Timeline(status="none", lines=[_line()])
    assert llm_merge.apply(timeline, client=llm, enabled=False) == []
    assert llm.prompts == []


def test_the_prompt_asks_for_boundaries_and_never_for_times():
    llm = FakeLLM({"1": ["君の", "心が", "見えない"]})
    llm_merge.apply(Timeline(status="none", lines=[_line()]), client=llm)
    prompt = llm.prompts[0]
    assert "只决定在哪里断开" in prompt
    assert "必须与这一行原文完全一致" in prompt
    assert "1. 君の心が見えない" in prompt
