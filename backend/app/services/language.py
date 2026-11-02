"""Which language is being sung?

The ASR accepts an explicit language and is more accurate with one than in
auto-detect mode, but a wrong guess is worse than none. So: the LLM reads what
we already know (title, artist, album, a slice of the lyrics) and answers with
one of the model's language names; if it cannot tell, or the answer is not a
language we can name, the caller passes nothing and the service auto-detects.

A script heuristic sits underneath the LLM: kana means Japanese, hangul Korean,
Han characters without kana Chinese. It only ever answers those three, because
latin-script songs are exactly the case where a guess is not worth it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .llm import LLMClient, LLMError

logger = logging.getLogger("lyrictrace.language")

#: Language names the Qwen3-ASR service understands (its own display names).
SUPPORTED = (
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "Cantonese",
    "French",
    "German",
    "Spanish",
    "Russian",
    "Italian",
    "Portuguese",
    "Thai",
    "Vietnamese",
    "Arabic",
)

#: Values that mean "let the ASR decide".
AUTO_VALUES = {"", "auto", "自动", "自动识别", "unknown", "none"}

LYRIC_SAMPLE_CHARS = 1200

PROMPT = """你是音乐语言判断助手。根据下面的歌曲信息判断这首歌是用什么语言演唱的。

歌名：{title}
歌手：{artist}
专辑：{album}
歌词节选：
{lyrics}

只输出一个 JSON 对象：{{"language": "<语言名>", "confidence": <0-1 的小数>}}
语言名只能从这个列表里选：{languages}。
如果信息不足以判断，把 language 设为 "unknown"。不要解释。"""


def is_auto(value: str | None) -> bool:
    return (value or "").strip().lower() in AUTO_VALUES


def normalize(value: str | None) -> str | None:
    """Map a free-form language name onto the ASR's names, else ``None``."""
    raw = (value or "").strip()
    if is_auto(raw):
        return None
    lowered = raw.lower()
    aliases = {
        "zh": "Chinese",
        "中文": "Chinese",
        "汉语": "Chinese",
        "普通话": "Chinese",
        "mandarin": "Chinese",
        "zh-cn": "Chinese",
        "en": "English",
        "英文": "English",
        "英语": "English",
        "ja": "Japanese",
        "jp": "Japanese",
        "日文": "Japanese",
        "日语": "Japanese",
        "日本語": "Japanese",
        "ko": "Korean",
        "韩文": "Korean",
        "韩语": "Korean",
        "한국어": "Korean",
        "粤语": "Cantonese",
        "yue": "Cantonese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
        "it": "Italian",
        "pt": "Portuguese",
        "th": "Thai",
        "vi": "Vietnamese",
        "ar": "Arabic",
    }
    if lowered in aliases:
        return aliases[lowered]
    for name in SUPPORTED:
        if lowered == name.lower():
            return name
    return None


def guess_from_script(text: str) -> str | None:
    """Kana/hangul/Han counts; only the unambiguous cases answer."""
    kana = len(re.findall(r"[぀-ヿ]", text))
    hangul = len(re.findall(r"[가-힯ᄀ-ᇿ]", text))
    han = len(re.findall(r"[㐀-鿿豈-﫿]", text))
    if kana:
        return "Japanese"
    if hangul:
        return "Korean"
    if han >= 4 and han >= len(re.findall(r"[A-Za-z]", text)):
        return "Chinese"
    return None


def detect(
    project: dict[str, Any],
    *,
    lyrics: str = "",
    client: LLMClient | None = None,
) -> tuple[str | None, str]:
    """Return ``(language, origin)``; ``language`` is ``None`` for auto.

    ``origin`` is one of ``"manual"``, ``"llm"``, ``"script"``, ``"auto"`` and is
    shown to the user so a wrong guess is easy to spot.
    """
    configured = project.get("lyrics_language")
    # A stored language is only a *decision* when the user made it. What the
    # LLM answered on an earlier run is stored too (so the UI can show it and a
    # run without the LLM still has something), but it must be asked again:
    # the upload-time guess is made from a title alone, and for 「怪物」 that is
    # a coin toss between Chinese and Japanese. Freezing it would put a wrong
    # language on every later run, and a wrong language rewrites every word.
    if not is_auto(configured) and project.get("language_origin") != "llm":
        return normalize(configured), "manual"

    title = (project.get("title") or "").strip()
    artist = (project.get("artist") or "").strip()
    album = (project.get("album") or "").strip()
    sample = (lyrics or project.get("original_lyrics") or "")[:LYRIC_SAMPLE_CHARS]
    if not any((title, artist, sample)):
        return None, "auto"

    guess = _ask_llm(title, artist, album, sample, client)
    if guess:
        return guess, "llm"
    heuristic = guess_from_script(" ".join((title, artist, album, sample)))
    if heuristic:
        return heuristic, "script"
    return None, "auto"


def _ask_llm(
    title: str,
    artist: str,
    album: str,
    lyrics: str,
    client: LLMClient | None,
) -> str | None:
    llm = client or LLMClient()
    if not llm.available():
        return None
    prompt = PROMPT.format(
        title=title or "（未知）",
        artist=artist or "（未知）",
        album=album or "（未知）",
        lyrics=lyrics.strip() or "（无歌词）",
        languages="、".join(SUPPORTED),
    )
    try:
        payload = llm.chat_json(prompt, max_tokens=120)
    except LLMError as exc:
        logger.warning("language detection failed: %s", exc)
        return None
    if not isinstance(payload, dict):
        return None
    try:
        confidence = float(payload.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    language = normalize(payload.get("language"))
    if language is None or confidence < 0.5:
        return None
    return language


def resolve(project: dict[str, Any], *, lyrics: str = "") -> tuple[str | None, str]:
    """``detect``; an undetermined language means the ASR auto-detects."""
    return detect(project, lyrics=lyrics)
