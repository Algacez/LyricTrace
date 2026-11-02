"""Audio file inspection using ffprobe."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .services.filename_parser import AUDIO_EXTENSIONS

MIME_BY_EXT = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wma": "audio/x-ms-wma",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".webm": "audio/webm",
}

TAG_KEYS = {
    "title": {"title", "tit2", "TIT2"},
    "artist": {"artist", "TPE1", "©art", "author"},
    "album": {"album", "TALB", "©alb"},
}


@dataclass
class AudioInfo:
    duration: float | None = None
    mime: str = "application/octet-stream"
    tags: dict[str, str] = field(default_factory=dict)
    format_name: str = ""
    bit_rate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str = ""


def is_supported_audio(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


def guess_mime(filename: str) -> str:
    return MIME_BY_EXT.get(Path(filename).suffix.lower(), "application/octet-stream")


def _extract_tags(payload: dict) -> dict[str, str]:
    raw_tags: dict[str, str] = {}
    for container in (payload.get("format", {}).get("tags") or {},):
        for key, value in container.items():
            if isinstance(value, str) and value.strip():
                raw_tags[key.lower()] = value.strip()
    for stream in payload.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        for key, value in (stream.get("tags") or {}).items():
            if isinstance(value, str) and value.strip():
                raw_tags.setdefault(key.lower(), value.strip())

    tags: dict[str, str] = {}
    for field_name, candidates in TAG_KEYS.items():
        for key in candidates:
            value = raw_tags.get(key.lower())
            if value:
                tags[field_name] = value
                break
    return tags


def cut_audio(
    source: Path, target: Path, start: float, duration: float, *, sample_rate: int = 16000
) -> Path:
    """Write one mono 16 kHz WAV slice of ``source`` (for windowed ASR)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to slice audio for ASR.")
    command = [
        ffmpeg, "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(source),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg slice failed: {completed.stderr.strip()[:400]}")
    return target


def probe_audio(path: Path) -> AudioInfo:
    info = AudioInfo(mime=guess_mime(path.name))
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return info
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (subprocess.SubprocessError, OSError):
        return info
    if completed.returncode != 0:
        return info
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return info

    fmt = payload.get("format") or {}
    try:
        info.duration = float(fmt["duration"]) if fmt.get("duration") is not None else None
    except (TypeError, ValueError):
        info.duration = None
    info.format_name = fmt.get("format_name") or ""
    try:
        info.bit_rate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
    except (TypeError, ValueError):
        info.bit_rate = None

    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "audio":
            info.codec = stream.get("codec_name") or ""
            try:
                info.sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
            except (TypeError, ValueError):
                info.sample_rate = None
            info.channels = stream.get("channels")
            if info.duration is None and stream.get("duration"):
                try:
                    info.duration = float(stream["duration"])
                except (TypeError, ValueError):
                    pass
            break

    info.tags = _extract_tags(payload)
    return info
