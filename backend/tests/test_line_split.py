"""LLM line splitting for a run with no lyrics (``services/line_split``).

The rule the module exists to hold: the LLM may only decide **where the lines
break**. It never invents a character and never invents a time — an answer that
does not reproduce the recogniser's transcript exactly is thrown away and the
run keeps the pause-based split.
"""

import pytest

from app.schemas import Word
from app.services import line_split
from app.services.asr_lines import group_words


def _words(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


class FakeLLM:
    """``LLMClient`` stand-in: a canned answer, or an error."""

    def __init__(self, answer=None, *, error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def available(self):
        return True

    def chat_json(self, prompt, max_tokens=1500):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


class Unavailable:
    def available(self):
        return False


def _chinese(n=24, start=0.0):
    """n single-character tokens, back to back — no pause anywhere."""
    return [
        Word(text=chr(ord("あ") + (index % 20)), start=start + index * 0.2, end=start + index * 0.2 + 0.2, source="asr")
        for index in range(n)
    ]


def test_max_chars_is_per_language():
    assert line_split.max_chars_for("Chinese") == 12
    assert line_split.max_chars_for("Japanese") == 18
    # An unknown language gets the fallback, not a neighbour's number.
    assert line_split.max_chars_for("English") == 24
    assert line_split.max_chars_for(None) == 24


def test_the_configured_table_wins():
    assert line_split.max_chars_for("Chinese", {"chinese": 6}) == 6
    # ``default`` in a caller's table overrides the fallback, not a named language.
    assert line_split.max_chars_for("Japanese", {"default": 9}) == 18
    assert line_split.max_chars_for("Klingon", {"default": 9}) == 9


def test_a_faithful_answer_regroups_the_tokens():
    words = _chinese(24)
    answer = {"lines": ["".join(word.text for word in words[:12]), "".join(word.text for word in words[12:])]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    assert [len(group) for group in groups] == [12, 12]
    # Times and text are the recogniser's: only the grouping changed.
    assert groups[0][0].start == words[0].start
    assert groups[1][-1].end == words[-1].end


def test_an_answer_that_drops_a_character_is_discarded():
    words = _chinese(24)
    honest = "".join(word.text for word in words[:12])
    answer = {"lines": [honest[:-1], "".join(word.text for word in words[12:])]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False
    assert groups is None


def test_an_answer_that_invents_a_character_is_discarded():
    words = _chinese(24)
    answer = {"lines": ["".join(word.text for word in words[:12]) + "x", "".join(word.text for word in words[12:])]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False


def test_a_line_over_the_languages_limit_is_cut_to_fit():
    # 13 characters in one line, and the limit for Chinese is 12. The answer is
    # not thrown away: the line is cut into pieces that fit, so the ceiling holds
    # without the model having to be exact.
    words = _chinese(26)
    answer = {"lines": ["".join(word.text for word in words[:13]), "".join(word.text for word in words[13:])]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    assert [len(group) for group in groups] == [6, 10, 10]


def test_japanese_allows_a_longer_line_than_chinese():
    words = _chinese(36)
    answer = {"lines": ["".join(word.text for word in words[:18]), "".join(word.text for word in words[18:])]}
    # The same 18/18 cut fits Japanese as it stands and is cut in half for Chinese.
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    assert [len(group) for group in groups] == [9, 9, 9, 9]
    groups, used = line_split.split_lines(words, language="Japanese", client=FakeLLM(answer))
    assert used is True
    assert [len(group) for group in groups] == [18, 18]


def test_a_sentence_the_model_left_too_long_is_split_automatically():
    # What the user asked for: a line that runs past the ceiling is split for
    # him, and the meaning is kept by keeping the whole of it in order — the
    # split is the only thing that changes.
    words = _chinese(30)
    answer = {"lines": ["".join(word.text for word in words)]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    assert [len(group) for group in groups] == [10, 10, 10]
    assert [word.text for group in groups for word in group] == [word.text for word in words]


def test_a_cut_inside_a_token_is_discarded():
    # One token holds two characters; the model's answer splits it.
    words = _words(("残機", 0.0, 0.6), ("だ", 0.6, 0.9), ("から", 0.9, 1.2), ("の", 1.2, 1.4))
    answer = {"lines": ["残機だか", "らの"]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False
    assert group_words(words)  # the pause-based split is still available


def test_a_failed_request_keeps_the_pause_split():
    words = _words(("あ", 0.0, 0.3), ("い", 0.3, 0.6), ("う", 3.0, 3.3))
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(error=Exception("down")))
    assert used is False
    assert groups is None
    assert [len(group) for group in group_words(words)] == [2, 1]


def test_without_an_llm_there_is_nothing_to_ask():
    words = _chinese(24)
    groups, used = line_split.split_lines(words, language="Chinese", client=Unavailable())
    assert used is False
    assert groups is None


def test_a_pause_still_breaks_a_line_the_model_did_not_break():
    # The model wants one line, but the recogniser left 3 seconds of silence in
    # the middle of it: the pause wins, so the "line" does not straddle it.
    words = _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4), ("う", 3.0, 3.2), ("え", 3.2, 3.4))
    answer = {"lines": ["あいうえ"]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False


def test_one_long_line_with_no_cut_is_not_an_answer():
    # The model is asked to split a whole song and answers with one line: there
    # is no split in it, so nothing was decided. The ceiling would still be met
    # by cutting it, but every character would end up in a line the model never
    # chose — that is the pause-split's job, not this one's.
    words = _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4))
    answer = {"lines": ["あい"]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False
    assert group_words(words)  # the pause-based split is still available


def test_a_short_transcript_is_not_worth_a_request():
    client = FakeLLM({"lines": ["あいう"]})
    groups, used = line_split.split_lines(_words(("あ", 0.0, 0.2)), language="Chinese", client=client)
    assert used is False
    assert client.calls == 0


def test_an_answer_that_never_cuts_anything_is_still_rejected():
    # Not a ceiling case: two characters in two lines, both far under it. There
    # is no line break in the answer at all, so nothing was decided and the run
    # keeps the pause split. (A line over the ceiling is cut — see the test
    # above — but a line that fits is never cut apart.)
    words = _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4))
    answer = {"lines": ["あ", "い"]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False


def test_a_bare_list_answer_is_accepted_too():
    words = _chinese(24)
    answer = ["".join(word.text for word in words[:12]), "".join(word.text for word in words[12:])]
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True


def test_describe_names_the_language_and_its_limit():
    assert line_split.describe("Chinese") == "Chinese：每行最多 12 字"
    assert line_split.describe(None) == "（自动）：每行最多 24 字"


def test_the_prompt_never_asks_for_a_time():
    prompt = line_split.PROMPT.format(text="あいう", limit=12)
    assert "时间" not in prompt or "不改动" in prompt
    # It asks for cuts and forbids changing anything.
    assert "只决定在哪里断行" in prompt
    assert "必须与原文完全一致" in prompt


def test_a_two_character_fragment_does_not_discard_the_whole_answer():
    # What actually happened on a real run: the answer was 99 good lines, two of
    # them short (「对白」 and 「嗯」). Under the old all-or-nothing rule those
    # two fragments threw the answer away and the song fell back to
    # pause-splitting, with 23 of 40 lines over the limit.
    words = _chinese(20)
    answer = {
        "lines": [
            "".join(word.text for word in words[:4]),
            "".join(word.text for word in words[4:6]),  # 「对白」 alone
            "".join(word.text for word in words[6:14]),
            "".join(word.text for word in words[14:]),
        ]
    }
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    # The fragment keeps the answer alive and stays with the line after it,
    # rather than costing the whole song its semantic split.
    lengths = [len("".join(word.text for word in group)) for group in groups]
    assert lengths == [4, 10, 6]


def test_a_fragment_at_the_head_merges_into_the_next_line():
    words = _chinese(20)
    answer = {
        "lines": [
            "".join(word.text for word in words[:2]),  # 「嗯」 alone
            "".join(word.text for word in words[2:10]),
            "".join(word.text for word in words[10:]),
        ]
    }
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is True
    lengths = [len("".join(word.text for word in group)) for group in groups]
    assert lengths == [10, 10]


def test_an_answer_nobody_cut_at_all_is_still_rejected():
    words = _words(("あ", 0.0, 0.2), ("い", 0.2, 0.4))
    answer = {"lines": ["あい"]}
    groups, used = line_split.split_lines(words, language="Chinese", client=FakeLLM(answer))
    assert used is False
    assert groups is None
