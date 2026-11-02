"""Heuristic extraction of song metadata from messy audio/video filenames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".webm",
}

# Bracketed groups are almost always noise: uploader, quality, language, etc.
BRACKET_RE = re.compile(r"[\[\(【（〔][^\[\]\(\)【】（）〔〕]*[\]\)】）〕]")
BRACKET_RE_ALT = re.compile(r"[\[\(【（〔]([^\[\]\(\)【】（）〔〕]*)[\]\)】）〕]")

NOISE_WORDS = {
    "official", "official video", "official audio", "official mv", "mv", "m/v", "pv",
    "lyrics", "lyric", "lyric video", "audio", "video", "full", "hd", "hq", "4k", "8k",
    "1080p", "720p", "2160p", "hi-res", "hires", "remaster", "remastered", "explicit",
    "中字", "字幕", "完整版", "无损", "高音质", "官方", "高清", "纯享", "纯享版",
    "動態歌詞", "动态歌词", "歌詞", "歌词", "华语", "音乐", "音樂", "现场", "live",
}

PLATFORM_PREFIXES = [
    "bilibili", "哔哩哔哩", "b站", "youtube", "yt", "nicovideo", "niconico", "soundcloud",
    "spotify", "netease", "网易云", "qq音乐", "tiktok", "douyin", "抖音",
]

QUALITY_RE = re.compile(
    r"\b(\d{3,4}p|\d{3,4}x\d{3,4}|4k|8k|hdr|flac|320k|128k|aac|opus|mp3|wav|m4a|mv|pv)\b",
    re.IGNORECASE,
)
UPLOADER_RE = re.compile(r"(上传者|up主|up|uploader|by|author)\s*[:：]\s*\S+", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"\s+[-–—|｜/／~〜]+\s+|\s*[|｜]\s*")


@dataclass
class ParsedFilename:
    title: str = ""
    artist: str = ""
    album: str = ""
    cleaned: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "artist": self.artist, "album": self.album, "cleaned": self.cleaned}


def _strip_noise(text: str) -> str:
    text = BRACKET_RE.sub(" ", text)
    text = UPLOADER_RE.sub(" ", text)
    text = QUALITY_RE.sub(" ", text)
    for prefix in PLATFORM_PREFIXES:
        text = re.sub(rf"^\s*{re.escape(prefix)}\s*[:：\-]?\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -_|｜–—~〜.")


def _is_noise_segment(segment: str) -> bool:
    lowered = segment.strip().lower()
    if not lowered:
        return True
    if lowered in NOISE_WORDS:
        return True
    if any(word in lowered for word in NOISE_WORDS) and len(lowered) <= 20:
        return True
    return False


def _pick_brackets(raw: str) -> tuple[str, list[str]]:
    """Remove noise brackets, keep brackets that look like metadata (feat., 专辑)."""
    kept: list[str] = []

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if inner and not _is_noise_segment(inner) and len(inner) <= 40:
            kept.append(inner)
            return f" {inner} "
        return " "

    stripped = BRACKET_RE_ALT.sub(repl, raw)
    return stripped, kept


ALBUM_MARKER_RE = re.compile(r"(专辑|album|ost|原声|soundtrack)", re.IGNORECASE)


def parse_filename(filename: str) -> ParsedFilename:
    stem = Path(filename).stem
    stem = re.sub(r"[_　]+", " ", stem)

    # Brackets that name an album are metadata, not title noise: pull them out
    # before the generic noise stripping so they never leak into the title.
    album = ""

    def _pull_album(match: re.Match[str]) -> str:
        nonlocal album
        inner = match.group(1).strip()
        if not album and ALBUM_MARKER_RE.search(inner):
            album = re.sub(
                r"^(专辑|album|ost|原声|soundtrack)\s*[:：]?\s*", "", inner, flags=re.IGNORECASE
            ).strip()
            return " "
        return match.group(0)

    stem = BRACKET_RE_ALT.sub(_pull_album, stem)
    text, remaining_brackets = _pick_brackets(stem)
    text = _strip_noise(text)

    title = ""
    artist = ""
    parts = [p.strip() for p in SEPARATOR_RE.split(text) if p.strip()]
    if len(parts) >= 2:
        # Most platforms use "Artist - Title"; a track number prefix is stripped.
        left, right = parts[0], parts[1]
        left = re.sub(r"^\d{1,3}[.\s、]+", "", left).strip()
        right = re.sub(r"^\d{1,3}[.\s、]+", "", right).strip()
        if left and right:
            artist, title = left, right
        else:
            title = text
    elif parts:
        title = parts[0]

    if not title:
        title = text or stem

    if remaining_brackets:
        extras = " ".join(remaining_brackets)
        if not title:
            title = extras
        else:
            title = f"{title} {extras}"

    # "Title - Artist" is indistinguishable from the reverse; keep both usable
    # by preferring the longer field as title when the left side looks like a
    # platform/channel handle.
    if artist and len(artist) > 40:
        title, artist = artist, ""

    cleaned = " ".join(p for p in [title, artist] if p).strip()
    return ParsedFilename(title=title.strip(), artist=artist.strip(), album=album, cleaned=cleaned)
