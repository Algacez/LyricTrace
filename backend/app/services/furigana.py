"""Strip Japanese furigana (reading annotations) from lyric text.

Japanese lyric sources routinely put the reading of a kanji inline:

    残機（ざんき）を数える
    運命(さだめ)ってやつ
    ｜東京《とうきょう》へ

Those kana are not sung a second time — they annotate the kanji before them.
Leaving them in doubles the character count of the line, which breaks the
kana DP (it matches the reading twice, so the line's span stretches over the
next line) and shows up in every export. The same text is what the LLM sees
when it judges a line, so the annotation also makes verification noisier.

The rule is deliberately structural rather than "any parenthetical":

* the bracket must sit **directly after a kanji** (no space between them), and
* everything inside must be **kana only**.

So ``（Wow）``, ``(Oh yeah)``, ``（もう一度）`` after hiragana and
``【※注】`` (kanji inside) survive: they are sung content, not a reading.
``｜漢字《かんじ》`` is the explicit ruby form and is unwrapped to ``漢字``.
"""

from __future__ import annotations

import re

from ..schemas import Word

#: Kana that can make up a reading. ``・`` is excluded on purpose: it separates
#: words, so a bracket containing it is not a single reading.
_KANA = "\u3041-\u3096\u30a1-\u30fa\u30fc\u309d\u309e\u30fd\u30fe"
#: Kanji (plus the iteration mark) that a reading can follow.
_KANJI = "\u3400-\u4dbf\u4e00-\u9fff\u3005"

KANA_ONLY_RE = re.compile(rf"[{_KANA}]+")

#: ``漢字（かんじ）`` / ``漢字(かんじ)`` / ``漢字【かんじ】`` — the lookbehind
#: pins the bracket to a kanji, so ``（かんじ）`` alone is never touched.
_PAREN_RE = re.compile(
    rf"(?<=[{_KANJI}])"
    rf"(（[^（）\n]{{1,20}}）|\([^()\n]{{1,20}}\)|【[^【】\n]{{1,20}}】"
    rf"|〔[^〔〕\n]{{1,20}}〕|〈[^〈〉\n]{{1,20}}〉)"
)

#: ``｜漢字《かんじ》`` — the explicit ruby marker; ``｜`` goes with it.
_RUBY_RE = re.compile(rf"[｜|]([^｜|《》\n]{{1,20}})《[{_KANA}]{{1,30}}》")


def strip_furigana(text: str) -> str:
    """Remove kana-only brackets that annotate the kanji before them."""
    if not text or not text.strip():
        return text

    def keep_or_drop(match: re.Match[str]) -> str:
        bracketed = match.group(1)
        inner = bracketed[1:-1].strip()
        return "" if KANA_ONLY_RE.fullmatch(inner) else bracketed

    out = _RUBY_RE.sub(lambda match: match.group(1), text)
    return _PAREN_RE.sub(keep_or_drop, out)


def strip_furigana_text(text: str) -> str:
    """Apply :func:`strip_furigana` line by line, keeping the line structure."""
    if not text:
        return text
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(strip_furigana(line) for line in normalized.split("\n"))


#: A bracket that opens a chunk, e.g. the ``（ざんき）`` in ``yrc`` where the
#: reading is its own word-timed chunk.
_LEADING_READING_RE = re.compile(
    rf"^[（(【〔〈][{_KANA}]{{1,20}}[）)】〕〉]"
)


def _ends_with_kanji(text: str) -> bool:
    return bool(text) and re.fullmatch(rf"[{_KANJI}]", text[-1]) is not None


def strip_furigana_words(words: list[Word]) -> list[Word]:
    """Remove readings from a word-timed line, across word boundaries.

    ``yrc`` splits a line into word-timed chunks, and the annotation often lands
    in a chunk of its own (``残機`` | ``（ざんき）``). There the bracket has no
    kanji in front of it *inside* the chunk, so the per-text rule cannot see it;
    here the previous chunk's last character provides the adjacency. A chunk
    that held nothing but the reading is dropped, and its time is handed to the
    kanji chunk — the reading is not sung twice, it annotates that kanji's
    duration.
    """
    out: list[Word] = []
    for word in words:
        text = strip_furigana(word.text)
        if out and _ends_with_kanji(out[-1].text) and _LEADING_READING_RE.match(text):
            text = _LEADING_READING_RE.sub("", text, count=1)
        if not text:
            if out and word.end is not None:
                previous = out[-1]
                out[-1] = previous.model_copy(
                    update={"end": max(previous.end or 0.0, word.end)}
                )
            continue
        out.append(word.model_copy(update={"text": text}) if text != word.text else word)
    return out
