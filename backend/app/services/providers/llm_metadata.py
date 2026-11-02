"""LLM-based song metadata extraction.

Uploads arrive with names like

    _ACAね最累的一集…_「残機 _电锯人ED2」【ずっと真夜中でいいのに。】_羅武 - 1.….mp3

The rule-based parser cannot tell which part is the song and which is the
uploader, the show or the reaction channel. The LLM can. It never invents a
field: when the answer is unusable we keep the filename heuristic instead.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config import settings
from ...schemas import SongMetadata
from ..filename_parser import parse_filename
from ..llm import LLMClient, LLMError
from .base import MetadataExtractor

logger = logging.getLogger("lyrictrace.metadata")

MAX_FIELD = 120

PROMPT = """你是音乐元数据整理助手。下面是一个音频文件名（可能来自 B 站/YouTube 下载，
含有频道名、视频标题、集数、字幕组、画质等噪声）。请判断这首歌的歌名和歌手。

文件名：
{filename}

规则：
- title 是歌曲名本身，保留原语言（日文歌用日文，中文歌用中文），不要翻译、不要加引号。
- artist 是演唱者/歌手名。如果文件名里只有企划或频道名，就把真正歌手填进去；
  确实无法判断时填空字符串。
- album 是专辑名；文件名里没有专辑就填空字符串。
- 不要输出频道名、UP 主、字幕组、节目名、集数、画质标签。
- 只输出一个 JSON 对象，不要解释。

格式：{{"title": "...", "artist": "...", "album": "..."}}"""


class LLMMetadataExtractor(MetadataExtractor):
    """Filename heuristics, refined by the LLM; falls back on any failure."""

    name = "llm"
    description = "LLM reads the messy filename and picks out the song title and artist"

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        return self._client or LLMClient()

    def available(self) -> bool:
        return self.client.available()

    def unavailable_detail(self) -> str:
        return "Set LYRICTRACE_LLM_BASE_URL / LYRICTRACE_LLM_API_KEY / LYRICTRACE_LLM_MODEL to use LLM metadata extraction."

    def extract(self, filename: str, embedded: dict[str, Any] | None = None) -> SongMetadata:
        embedded = embedded or {}
        parsed = parse_filename(filename)
        fallback = SongMetadata(
            title=embedded.get("title") or parsed.title,
            artist=embedded.get("artist") or parsed.artist,
            album=embedded.get("album") or parsed.album,
            duration=embedded.get("duration"),
            source="embedded" if embedded.get("title") else "filename",
        )
        # Tags already answer both questions; don't spend a request on them.
        if embedded.get("title") and embedded.get("artist"):
            return fallback
        if not self.available():
            return fallback

        try:
            payload = self.client.chat_json(PROMPT.format(filename=filename), max_tokens=300)
        except LLMError as exc:
            logger.warning("LLM metadata extraction failed, keeping the filename guess: %s", exc)
            return fallback

        if not isinstance(payload, dict):
            return fallback
        title = _clean(payload.get("title")) or fallback.title
        artist = _clean(payload.get("artist")) or fallback.artist
        album = _clean(payload.get("album")) or fallback.album
        if not title:
            return fallback
        logger.info("LLM metadata for %s -> %s / %s", filename, title, artist)
        return SongMetadata(
            title=title,
            artist=artist,
            album=album,
            duration=embedded.get("duration") or fallback.duration,
            source="llm",
        )


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().strip("\"'“”「」『』")
    if text.lower() in {"", "unknown", "n/a", "null", "none", "未知", "不详"}:
        return ""
    return text[:MAX_FIELD]
