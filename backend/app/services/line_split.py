"""Split an ASR transcript into lyric-like lines with the LLM, per language.

A run with no lyrics has only the recogniser's tokens. ``asr_lines`` cuts them
at pauses, which is honest but reads badly: a pause is where the singer
breathed, not where the line ended, so one "line" can hold two sung phrases and
the next can be a single word. A sung line is a phrase, and how long a phrase
may be depends on the language — 12 Chinese characters is already a full line,
while 18 kana or a handful of English words is one.

So the transcript goes to the LLM once, which says where the line breaks go.
It is asked for **cuts only**, never for text or times: it answers with the same
characters regrouped, and the answer is used only when concatenating its groups
reproduces the transcript exactly (whitespace aside). A grouping that invents or
drops a character is discarded, because the transcript is the only text we have
— anything else would be a lyric nobody sang.

Every cut must also land on a real token boundary: a line is a run of whole
tokens, so a cut inside a token means the answer is not the transcript. The
character limit is a ceiling, not a target — the LLM is free to cut earlier, and
a line it left too long is cut here rather than thrown away, so the limit holds
without the model having to be exact.

Nothing here invents a time. The tokens keep their own start/end; only which
tokens share a line changes, exactly as in ``asr_lines.group_words``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas import Word
from .asr_lines import LINE_GAP_SEC
from .llm import LLMClient, LLMError

logger = logging.getLogger("lyrictrace.line_split")

#: A line needs at least this many characters, or it is a fragment nobody would
#: write down. It is a **soft** floor: a cut that leaves a line shorter than
#: this is dropped and the characters join the line after them, never a reason
#: to throw the answer away (see ``_cuts_from_lines``).
MIN_LINE_CHARS = 4

PROMPT = """下面是一首歌的识别文本，没有任何原始歌词。请按唱出来的句子把它断成一行一行。

要求：
1. 只决定在哪里断行，不改动、不增删任何字符。把每一行按顺序拼起来，必须与原文完全一致（空白不算）。
2. 每一行是一个完整的演唱句子，长度控制在 {limit} 个字符以内；宁可断得短一些，也不要超过。
3. 句子太长就拆开：在自然的语义边界断——一个短语唱完、主语谓语完整、一个意象结束的地方。不要把一个词拆开。
4. 说话、掌声、哼唱如果被识别成文字，单独成行。
5. 只输出 JSON：{{"lines": ["第一行", "第二行"]}}。不要解释。

文本：
{text}"""


def max_chars_for(
    language: str | None, table: dict[str, int] | None = None, override: int | None = None
) -> int:
    """The character ceiling for this language, from the configured table.

    ``override`` is one project's (or one request's) own number and wins over
    the table — ``0``/``None`` means "use the table", so the default keeps the
    per-language behaviour.
    """
    if override:
        return max(int(override), MIN_LINE_CHARS)
    from ..config import settings

    merged = dict(settings.line_split_max_chars)
    if table:
        merged.update({str(key).lower(): value for key, value in table.items()})
    key = (language or "").strip().lower()
    if key in merged:
        return merged[key]
    return merged.get("default", 24)


def _flat(text: str) -> str:
    return "".join((text or "").split())


def _token_chars(word: Word) -> int:
    return len(_flat(word.text))


def _ask(client: LLMClient, text: str, limit: int) -> list[str] | None:
    """One request. A malformed or unfaithful answer is ``None``, never a guess."""
    try:
        answer = client.chat_json(PROMPT.format(text=text, limit=limit), max_tokens=len(text) + 400)
    except LLMError as exc:
        logger.info("LLM line split request failed: %s", exc)
        return None
    items: Any = answer
    if isinstance(answer, dict):
        items = answer.get("lines")
    if not isinstance(items, list):
        return None
    lines = [item.strip() for item in items if isinstance(item, str) and item.strip()]
    return lines or None


def _cuts_from_lines(lines: list[str], words: list[Word], limit: int) -> list[int] | None:
    """Token indices where a new line starts, or ``None`` if unusable.

    Two things can make the answer unusable, and only two: it does not
    reproduce the transcript (a character invented or dropped, or a cut inside a
    token), or nothing was cut at all. Everything else is fixed in place:

    * a cut that leaves a line **over** ``limit`` is repaired rather than
      honoured: the line is cut again into pieces that fit, so the ceiling holds
      without the model having to be exact. A line the model left one character
      too long must not cost the whole answer, and a line it left ten characters
      too long is exactly the "sentence too long, split it" case the user asked
      for — the split is automatic and the meaning is kept;
    * a cut that leaves a line **under** ``MIN_LINE_CHARS`` is dropped, because
      "嗯" or "对白" on its own line is a fragment nobody writes down; its
      characters join the line after it instead.

    Dropping one cut can leave its neighbour's line too long; that is handled by
    the same loop, which keeps cutting until every line fits. A fragment at the
    very start or end of the song is merged with its only neighbour rather than
    left dangling.

    If dropping leaves nothing cut at all (the model offered one cut and it was
    unusable), the answer is unusable too — one line is not a split.
    """
    target = [_flat(word.text) for word in words]
    total = len("".join(target))
    boundaries: list[int] = []
    index = 0
    for line in lines:
        length = len(_flat(line))
        if length == 0 or length > total - index:
            return None
        covered = 0
        while index < len(target) and covered < length:
            covered += len(target[index])
            index += 1
        if covered != length:
            # The model split inside a token: the text is not the transcript.
            return None
        boundaries.append(index)
    if index != len(target):
        return None
    if not boundaries:
        return None
    # Each boundary closes the span that starts at the cut before it, the last
    # one included — the final line of the answer is judged like any other, so a
    # line the model left too long at the end of the song is cut too.
    cuts: list[int] = []
    for end in boundaries:
        start = cuts[-1] if cuts else 0
        if end <= start:
            continue
        length = _span_chars(target, start, end)
        if length < MIN_LINE_CHARS:
            # A fragment: its characters stay with the line after it.
            continue
        if length <= limit:
            if end < len(target):
                cuts.append(end)
            continue
        cuts.extend(_chunk_within(target, start, end, limit))
    if not cuts:
        return None
    # The tail after the last kept cut: too short, it belongs to the line above.
    if _span_chars(target, cuts[-1], len(target)) < MIN_LINE_CHARS:
        cuts.pop()
    return cuts or None


def _span_chars(target: list[str], start: int, end: int) -> int:
    return sum(len(item) for item in target[start:end])


def _chunk_within(target: list[str], start: int, end: int, limit: int) -> list[int]:
    """Cuts inside ``(start, end)`` so every line fits ``limit``.

    Called only for a span already known to be over the ceiling — the case the
    user asked for: a sentence too long gets split automatically, and only the
    meaning decides where. This function cannot know the meaning, so it cuts the
    span into as few pieces as the ceiling allows and puts each cut as close to
    the even mark as a token boundary manages, which is the closest thing to a
    fair split available without reading the words. A leftover shorter than
    ``MIN_LINE_CHARS`` is never left standing: the candidate is chosen only where
    the rest of the span still has a share of its own. A single token longer
    than ``limit`` cannot be cut at all and is left whole.
    """
    span = _span_chars(target, start, end)
    parts = max(1, -(-span // limit))
    cuts: list[int] = []
    position = start
    for part in range(1, parts):
        wanted = span * part / parts - _span_chars(target, start, position)
        length = 0
        chosen: int | None = None
        gap: float | None = None
        for index in range(position, end):
            length += len(target[index])
            if length > limit or _span_chars(target, index + 1, end) < MIN_LINE_CHARS:
                break
            if length < MIN_LINE_CHARS:
                continue
            distance = abs(length - wanted)
            if gap is None or distance < gap:
                gap = distance
                chosen = index + 1
        if chosen is None or chosen <= position:
            break
        cuts.append(chosen)
        position = chosen
    return cuts


def split_lines(
    words: list[Word],
    *,
    language: str | None,
    gap: float = 0.9,
    client: LLMClient | None = None,
    max_chars: int | None = None,
) -> tuple[list[list[Word]] | None, bool]:
    """``words`` regrouped into lines, or ``(None, False)`` when nothing helped.

    ``None`` is the normal, expected outcome when there is no LLM, when the
    request fails, or when the answer does not reproduce the transcript; the
    caller then keeps ``asr_lines``' pause-based split. The words themselves are
    returned unchanged — only the grouping the caller makes of them differs.

    ``max_chars`` overrides the per-language ceiling for this call (one
    project's setting, or one API request's); omitted means the table.
    """
    usable = [word for word in words if word.start is not None and word.end is not None]
    if sum(_token_chars(word) for word in usable) < 2 * MIN_LINE_CHARS:
        return None, False
    limit = max_chars_for(language, override=max_chars)
    llm = client or LLMClient()
    if not llm.available():
        return None, False
    transcript = "".join(word.text for word in usable).strip()
    if not transcript:
        return None, False
    lines = _ask(llm, transcript, limit)
    cuts = _cuts_from_lines(lines, usable, limit) if lines else None
    if not cuts:
        return None, False
    # The answer is faithful: group at its cuts, then keep the pause rule too —
    # a cut that lands inside a long silence the model did not notice is not a
    # line break, so the pause wins there.
    groups: list[list[Word]] = []
    current: list[Word] = []
    wanted = set(cuts)
    for index, word in enumerate(usable):
        if current and (index in wanted or _after_pause(current[-1], word, gap)):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    if len(groups) <= 1:
        return None, False
    return groups, True


def _after_pause(previous: Word, word: Word, gap: float) -> bool:
    """Whether the recogniser left room between two tokens (see ``LINE_GAP_SEC``)."""
    if word.start is None or previous.end is None:
        return False
    return word.start - previous.end > gap


def describe(language: str | None, max_chars: int | None = None) -> str:
    """Human-readable limit, for the debug page and the run notes."""
    name = (language or "").strip() or "（自动）"
    return f"{name}：每行最多 {max_chars_for(language, override=max_chars)} 字"


__all__ = [
    "MIN_LINE_CHARS",
    "PROMPT",
    "describe",
    "max_chars_for",
    "split_lines",
]
