"""The recogniser's per-character table: where the per-character times come from.

Per-character times come from the ASR word stream's own tokens. The ASR service
(Qwen3-ASR, with its forced aligner switched on) transcribes the audio and aligns
its own transcript; those tokens are per character, and a character the
recogniser timed directly is a real measurement of when it was sung. Its weakness
is *absolute* placement — over a whole song it drifts — so ``char_align`` does
not trust a single token: it re-slices each line from the tokens heard over it,
fills the gaps and spreads any characters the recogniser stacked on one instant.

A second recognition pass (with the aligner on) produces the per-character table
in ``second_words``. It is used two ways:

* as the *only* source, by ``_retime_asr_only`` — every line's characters are cut
  from the tokens heard over it (the recogniser's own boundary wins, because it is
  a direct measurement);
* as a cross-check, by ``_fuse_second_opinion`` — the second pass is treated as
  a second opinion and the recogniser's boundary is kept wherever the two agree
  within ``FUSE_TOLERANCE`` (a character the ASR merely interpolated is taken
  from the second pass).

Switching the cross-check off with ``LYRICTRACE_CHAR_FUSE=false`` makes the
second pass a pure second opinion and every character keeps the recogniser's own
boundary from the line it was cut from. Either way, no external forced-aligner
service is involved — both tables are produced by the ASR service itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

from ..audio import cut_audio, probe_audio
from ..config import settings
from ..schemas import Word
from .asr_lines import repair_word_times
from .providers.base import registry

logger = logging.getLogger("lyrictrace.fuse")

#: Recognition window, the same length the word stream uses: the recogniser
#: drifts over minutes of singing, and a short window is what keeps its
#: timings usable (see ``pipeline.transcribe_audio``).
WINDOW_SEC = 20.0
#: What the cache was produced with, so a later run can tell whether to reuse it.
CACHE_NAME = "second-words.json"


def available(*, honour_switch: bool = True) -> bool:
    """Whether the recogniser's per-character table can be fetched.

    ``honour_switch`` is for the cross-check against the line-cut table: with
    ``LYRICTRACE_CHAR_FUSE=false`` there is nothing to cross-check against, and
    the line-cut table is the only source. Without that switch being honoured
    this is just "is the ASR service reachable?".
    """
    if honour_switch and not settings.char_fuse:
        return False
    return bool(settings.asr_url)


def unavailable_detail() -> str:
    if not settings.char_fuse:
        return "第二个逐字对齐来源已关闭（LYRICTRACE_CHAR_FUSE=false）"
    return "没有配置 ASR 服务（LYRICTRACE_ASR_URL），无法取得第二个逐字对齐结果"


def _windows(duration: float) -> list[tuple[float, float]]:
    """Fixed-length, non-overlapping windows covering the audio."""
    if duration <= 0:
        return [(0.0, WINDOW_SEC)]
    ranges: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        ranges.append((start, min(WINDOW_SEC, duration - start)))
        start += WINDOW_SEC
    return ranges


def _fingerprint(audio_path: Path) -> dict[str, Any]:
    """What makes two runs the same audio: the file's size and mtime."""
    try:
        stat = audio_path.stat()
    except OSError:
        return {}
    return {"size": stat.st_size, "mtime": round(stat.st_mtime, 3)}


def _cached(
    work_dir: Path, language: str | None, fingerprint: dict[str, Any]
) -> list[Word] | None:
    path = work_dir / CACHE_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (payload.get("language") or "") != (language or ""):
        # Recognised under another language: the tokens are a different reading
        # of the same audio and would fuse nothing sensible.
        return None
    if payload.get("audio") != fingerprint:
        # A different take or a different stem (separated vocals vs the mix):
        # its times say nothing about this audio.
        return None
    return [Word(**item) for item in payload.get("words", [])]


def second_words(
    audio_path: Path,
    work_dir: Path,
    language: str | None,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    honour_switch: bool = True,
) -> list[Word]:
    """The ASR service's aligned tokens for the whole song, absolute times.

    Never raises: a second opinion is not worth failing a run over, so every
    failure comes back as an empty list and the run keeps the line-cut table's
    times.
    """
    if not available(honour_switch=honour_switch):
        return []
    fingerprint = _fingerprint(audio_path)
    cached = _cached(work_dir, language, fingerprint)
    if cached is not None:
        logger.info("reusing %d cached second-opinion tokens", len(cached))
        return cached

    report = on_progress or (lambda progress, message: None)
    provider = registry.asr()
    switch = getattr(provider, "with_align", None)
    if switch is not None:
        provider = switch(True)
    if not provider.available():
        return []

    work_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_audio(audio_path).duration or 0.0
    windows = _windows(duration)
    words: list[Word] = []
    for number, (start, length) in enumerate(windows, 1):
        slice_path = work_dir / f"second-{int(start):05d}.wav"
        try:
            cut_audio(audio_path, slice_path, start, length)
            result = asyncio.run(provider.transcribe(slice_path, language))
        except Exception as exc:  # noqa: BLE001 - one window must not sink the run
            logger.warning("second-opinion ASR failed for %.1fs+%.1fs: %s", start, length, exc)
            report(number / len(windows), f"第二逐字来源第 {number}/{len(windows)} 段识别失败，已跳过")
            continue
        for word in result.words:
            if not word.text.strip() or word.start is None or word.end is None:
                continue
            words.append(
                Word(
                    text=word.text,
                    start=round(word.start + start, 3),
                    end=round(word.end + start, 3),
                    source="asr",
                )
            )
        report(number / len(windows), f"第二逐字来源识别 {number}/{len(windows)} 段")
        slice_path.unlink(missing_ok=True)

    words.sort(key=lambda word: (word.start, word.end))
    words = repair_word_times(words)
    if not words:
        return []
    (work_dir / CACHE_NAME).write_text(
        json.dumps(
            {
                "language": language or "",
                "audio": fingerprint,
                "words": [word.model_dump() for word in words],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return words
