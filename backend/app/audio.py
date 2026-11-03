"""Audio file inspection and repair using ffmpeg/ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .services.filename_parser import AUDIO_EXTENSIONS

logger = logging.getLogger("lyrictrace.audio")

#: How long an uploaded file may take to re-containerise. A four-minute song
#: copies in a couple of seconds; this is only a ceiling against a hang.
REMUX_TIMEOUT_SEC = 600

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


def remux_audio(source: Path, target: Path | None = None) -> Path:
    """Rewrite ``source`` into a clean container, keeping its audio bytes.

    A FLAC (or M4A, or WAV) can carry metadata the browser refuses to read: a
    PICTURE block with a malformed header is enough for Chrome to fail the whole
    file with ``MediaError`` code 4, while ffmpeg — and so this project's ASR —
    reads it without complaint. Re-muxing copies the audio frames into a fresh
    container and leaves the pictures behind, which is exactly the difference
    between the two readers.

    Cost: one copy of the file's audio (a few seconds of CPU for a typical
    song) and the loss of the embedded cover art. Tags survive, so the metadata
    this project shows is unchanged whether it is read before or after.

    On any failure the original file is left untouched and returned: a file we
    cannot repair is still a file, and the pipeline may well be able to read it.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.info("ffmpeg missing: leaving %s as uploaded", source.name)
        return source
    destination = target or source.with_name(source.stem + ".remux" + source.suffix.lower())
    command = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(source),
        # ``-map 0:a`` keeps only the audio streams, ``-vn`` drops the attached
        # pictures, and ``-c:a copy`` copies the audio frames byte for byte — so
        # what lands in the new container is the same recording, without the
        # PICTURE block the browser refuses to read. ``-map_metadata 0`` keeps
        # the title/artist/album, because the pipeline reads them off the file
        # on disk later, not off the upload.
        "-map", "0:a", "-map_metadata", "0", "-vn",
        "-c:a", "copy",
        str(destination),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=REMUX_TIMEOUT_SEC, check=False
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.info("remux of %s failed: %s", source.name, exc)
        destination.unlink(missing_ok=True)
        return source
    if completed.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        logger.info("remux of %s produced nothing: %s", source.name, completed.stderr.strip()[:200])
        destination.unlink(missing_ok=True)
        return source
    if target is None:
        # Swap in place: the caller keeps the path it already stored. A failure
        # here leaves the repaired copy under its own name, which is worse than
        # the original — so the original goes back and the copy is dropped.
        try:
            destination.replace(source)
        except OSError as exc:
            logger.info("could not replace %s with its remux: %s", source.name, exc)
            destination.unlink(missing_ok=True)
            return source
        return source
    return destination


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
