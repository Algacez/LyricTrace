"""Bracketed kana after a kanji is a reading, not a second thing to sing."""

import pytest

from app.schemas import Word
from app.services.furigana import strip_furigana, strip_furigana_text, strip_furigana_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("残機（ざんき）を数える", "残機を数える"),
        ("運命(さだめ)ってやつ", "運命ってやつ"),
        ("東京【とうきょう】へ", "東京へ"),
        ("今日（きょう）は", "今日は"),
        ("残機（ザンキ）", "残機"),
        ("漢字（かんじ）と漢字（かんじ）", "漢字と漢字"),
        ("[00:12.34]残機（ざんき）", "[00:12.34]残機"),
    ],
)
def test_kana_bracket_attached_to_a_kanji_is_removed(text, expected):
    assert strip_furigana(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hello (Wow)",
        "君の（もう一度）",  # preceded by kana: sung content, not a reading
        "漢字 （かんじ）",  # the space separates it from the kanji
        "【※注】",  # kanji inside: an annotation, but not a reading
        "（CV：花澤香菜）",
        "中文（笑）",
        "（ららら）",  # nothing before it to annotate
    ],
)
def test_parenthetical_that_is_sung_content_survives(text):
    assert strip_furigana(text) == text


def test_explicit_ruby_marker_is_unwrapped():
    assert strip_furigana("｜東京《とうきょう》へ") == "東京へ"
    assert strip_furigana("|残機《ざんき》") == "残機"


def test_text_level_keeps_the_line_structure():
    assert strip_furigana_text("残機（ざんき）\r\n運命(さだめ)\n\n最後の行") == "残機\n運命\n\n最後の行"


def test_word_chunk_holding_only_the_reading_is_folded_into_the_kanji():
    """``yrc`` gives the reading a chunk of its own; it is not sung twice."""
    words = [
        Word(text="残機", start=1.0, end=1.5),
        Word(text="（ざんき）", start=1.5, end=2.0),
        Word(text="を", start=2.0, end=2.5),
    ]
    out = strip_furigana_words(words)
    assert [word.text for word in out] == ["残機", "を"]
    assert out[0].end == 2.0
    assert out[1].start == 2.0


def test_leading_bracket_without_a_kanji_before_it_stays():
    words = [Word(text="（ららら）", start=1.0, end=2.0)]
    assert [word.text for word in strip_furigana_words(words)] == ["（ららら）"]
