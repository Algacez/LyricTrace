"""Group the finished per-character timeline into units that read as words.

The per-character pass hands out one time per character, and that is what the
times are measured at. It is not always what a highlight should follow: a
character is the *unit of measurement*, a word or a phrase is the *unit of
meaning*, and a karaoke line that lights up one kana at a time is hard to read
even when every kana's time is right. So the finished timeline goes to the LLM
once, which says where the semantic boundaries are — 君の心が見えない is 君の /
心が / 見えない, not six characters — and each unit then takes the span of the
characters it covers: the first character's start to the last one's end.

The model is asked for boundaries, never for times: it answers with the same
characters split into groups, and a group is used only when concatenating the
groups reproduces the line exactly (whitespace aside). Anything the model
answers that does not tile the line is discarded and that line keeps its
per-character times — a merge that invents or drops a character would silently
move the highlight onto the wrong text, which is worse than a fine-grained one.

The units are contiguous by construction: characters were already filled to
run until the next one starts, so a unit ends exactly where the next begins.
Nothing else changes — line boundaries, the text and every character time the
aligner produced stay as they were; only how many highlight blocks the line is
drawn in changes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from ..schemas import Line, Timeline, Word
from .llm import LLMClient, LLMError

logger = logging.getLogger("lyrictrace.merge")

#: Lines per request. A whole song in one prompt invites a dropped or merged
#: line number, and a batch of twelve is still one request per minute of song.
BATCH_LINES = 12
#: A one-character line has nothing to merge.
MIN_LINE_CHARS = 2
#: A unit this much of the line or more is not a merge, it is the whole line:
#: nothing to draw differently, so the line is left per-character.
SPACE_RE = re.compile(r"\s+")


def available(enabled: bool = True) -> bool:
    """Whether this pass runs at all: the project asked for it, and LLM exists."""
    return bool(enabled) and LLMClient().available()


def unavailable_detail(enabled: bool = True) -> str:
    if not enabled:
        return "没有开启「按语意合并成词」：逐字结果按每个字给出时间"
    return "没有配置 LLM（LYRICTRACE_LLM_BASE_URL / LYRICTRACE_LLM_MODEL），无法按语意合并"


def _flat(text: str) -> str:
    """The text without whitespace: what has to match, not the spacing."""
    return SPACE_RE.sub("", text or "")


def _prompt(items: list[tuple[int, str]]) -> str:
    lines = "\n".join(f"{number}. {text}" for number, text in items)
    return (
        "下面是一首歌的歌词，每行前面是行号。请把每一行按语意切成完整的单元："
        "一个词、一个词组或一个短语。\n"
        '例如「君の心が見えない」切成 ["君の", "心が", "見えない"]。\n'
        "规则：\n"
        "1. 只决定在哪里断开，不改动任何字符。把一行的单元按顺序拼起来，"
        "必须与这一行原文完全一致（空白不算）。\n"
        "2. 单元按语意切：助词跟前一个词，修饰语跟被修饰的词，不要把一个词拆开。\n"
        "3. 一两个字这样太短的行，整行原样作为一个单元。\n"
        "4. 不要跨行合并，每一行只切它自己的字。\n"
        "5. 只输出 JSON，键是行号，值是这一行的单元数组，例如 "
        '{"12": ["君の", "心が", "見えない"], "13": ["見えない"]}。\n\n'
        f"{lines}"
    )


def _ask(client: LLMClient, items: list[tuple[int, str]]) -> dict[int, list[str]]:
    """One request; a malformed answer is an empty answer, never a guess."""
    try:
        answer = client.chat_json(_prompt(items), max_tokens=64 * len(items) + 256)
    except LLMError as exc:
        logger.warning("LLM merge request failed: %s", exc)
        return {}
    if not isinstance(answer, dict):
        return {}
    out: dict[int, list[str]] = {}
    for key, value in answer.items():
        try:
            number = int(str(key).strip())
        except ValueError:
            continue
        if not isinstance(value, list):
            continue
        units = [item for item in value if isinstance(item, str) and item.strip()]
        if units:
            out[number] = units
    return out


def _merge_line(line: Line, units: list[str]) -> int:
    """Replace the line's characters with the units; returns the units made.

    Zero means nothing changed: the units do not tile the line, the line is too
    short, or its characters have no times to merge.
    """
    words = line.words or []
    if len(_flat(line.text)) < MIN_LINE_CHARS or len(words) < 2:
        return 0
    if any(word.start is None or word.end is None for word in words):
        return 0
    if _flat("".join(word.text for word in words)) != _flat(line.text):
        return 0
    merged: list[dict[str, Any]] = []
    index = 0
    for unit in units:
        target = len(_flat(unit))
        if target == 0:
            return 0
        group = []
        covered = 0
        while index < len(words) and covered < target:
            group.append(words[index])
            covered += len(_flat(words[index].text))
            index += 1
        if covered != target or not group:
            return 0
        merged.append(
            {
                # Whitespace is layout, not content: a unit that starts after a
                # space keeps the space in the line, not in the highlight.
                "text": "".join(word.text for word in group).strip(),
                "start": group[0].start,
                "end": group[-1].end,
            }
        )
    if index != len(words):
        return 0
    if len(merged) < 2:
        # One unit for the whole line: the highlight would cover the entire
        # line at once, which is coarser than what the line already has.
        return 0
    line.words = [
        # ``merged``: the time is the span of the characters this unit covers,
        # not a measurement of its own — the characters underneath are.
        Word(text=item["text"], start=item["start"], end=item["end"], source="merged")
        for item in merged
    ]
    line.start = line.words[0].start
    line.end = line.words[-1].end
    return len(line.words)


def apply(
    timeline: Timeline,
    *,
    language: str | None = None,
    trace: Any = None,
    on_progress: Callable[[float, str], None] | None = None,
    client: LLMClient | None = None,
    enabled: bool = True,
) -> list[str]:
    """Merge each line's characters into semantic units; returns run notes.

    ``enabled`` is the project's 「按语意合并成词」 switch: off means the
    finished timeline keeps one time per character and no request is made.

    Never raises: the per-character timeline is already complete and usable, so
    a failed or unhelpful answer only means the timeline stays per-character.
    """
    if not enabled:
        return []
    llm = client or LLMClient()
    if not llm.available():
        return [f"{unavailable_detail()}；逐字结果仍按每个字给出时间"]

    report = on_progress or (lambda progress, message: None)
    targets = [
        (index, line)
        for index, line in enumerate(timeline.lines)
        if (line.text or "").strip()
        and len(_flat(line.text)) >= MIN_LINE_CHARS
        and len(line.words or []) >= 2
        and all(word.start is not None and word.end is not None for word in line.words)
    ]
    if not targets:
        return []

    merged_lines = 0
    merged_units = 0
    chars_merged = 0
    detail: list[dict[str, Any]] = []
    for start in range(0, len(targets), BATCH_LINES):
        batch = targets[start : start + BATCH_LINES]
        items = [(index + 1, line.text) for index, line in batch]
        answer = _ask(llm, items)
        for (index, line), (number, text) in zip(batch, items):
            units = answer.get(number)
            if not units:
                continue
            made = _merge_line(line, units)
            if not made:
                logger.info("line %d: the LLM's units do not tile the line, kept per-character", index + 1)
                continue
            merged_lines += 1
            merged_units += made
            chars_merged += len(_flat(text))
            detail.append({"line": index + 1, "text": text, "units": [word.text for word in line.words]})
        report(
            min(1.0, (start + len(batch)) / len(targets)),
            f"按语意合并 {min(start + len(batch), len(targets))}/{len(targets)} 句",
        )

    if trace is not None:
        trace.step(
            "llm_merge",
            {
                "model": llm.model,
                "lines": len(targets),
                "merged_lines": merged_lines,
                "units": merged_units,
                "detail": detail,
                "note": "LLM 只决定在哪里断词，时间仍是逐字结果：每个单元取它第一个字的起点和最后一个字的终点",
            },
        )
    if not merged_lines:
        return ["LLM 没有给出可用的语意切分（返回的单元与歌词对不上），逐字结果仍按每个字给出时间"]
    return [
        f"逐字结果已按语意合并成词：{merged_lines}/{len(targets)} 句共 {merged_units} 个单元"
        "（LLM 判断在哪里断词），每个单元的时间取自它第一个字的起点和最后一个字的终点；"
        "其余句子仍按每个字给出时间"
    ]
