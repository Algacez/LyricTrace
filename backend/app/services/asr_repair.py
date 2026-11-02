"""Re-recognise the sentences where the ASR's own timings do not add up.

The forced aligner occasionally drops the middle of a sung phrase. The tokens
on either side of the missing words are stamped correctly, so a *single lyric
line* ends up with a multi-second hole between two characters — a hole that no
singer leaves mid-phrase. The line's internal timings are then wrong, and no
amount of re-mapping the lyrics can fix that: the timing source itself is bad.

The unit is the **original lyric line**: a sentence is whatever the lyrics say
it is, and the ASR result is mapped onto those sentences by the kana DP. Only
a hole *inside one line's own mapped span* counts; a pause between two lines is
where the singer breathes and is normal.

Where a line's timings do not add up, the whole sentence is cut out of the
audio (padded a little for context) and recognised again — a couple of hundred
milliseconds around the hole is not enough context for the recogniser to hear
what it dropped the first time. The retry replaces that line's tokens only if
the hole is actually closed; otherwise the original tokens stay, so the outcome
is either strictly better or the original result, and every attempt is
recorded for the debug page.

A hole the sentence retry cannot close has a second cause: the line was mapped
onto a span that runs into silence, so its last characters are stamped over an
instrumental or on top of the next line. For those, ``_tight_retry`` cuts the
window at the hole instead of at the span's end and keeps the result only if
the recogniser hears the *whole* sentence inside that short stretch.
"""

from __future__ import annotations

import difflib
import inspect
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..audio import cut_audio
from ..schemas import Line, Word
from .providers.bs_roformer import to_asr_wav
from .providers.llm_aligner import align_spans

logger = logging.getLogger("lyrictrace.asr_repair")

#: A hole shorter than this is an ordinary breath between syllables.
GAP_MIN_SEC = 0.8
#: ... and one shorter than this many median tokens is ordinary too: a slow,
#: sustained note legitimately spaces syllables out.
GAP_FACTOR = 5.0
#: Audio kept on each side of the sentence, so the retry has context to align to.
WINDOW_PAD_SEC = 0.4
#: The retry must also shrink the hole to at most this share of its size.
IMPROVEMENT_RATIO = 0.5
#: A retry window is never shorter than this (a 0.2 s clip has no context).
MIN_WINDOW_SEC = 1.0
#: The re-recognised sentence must keep at least this share of the original
#: characters; a retry that hears half the line dropped the same words again.
MIN_TEXT_SHARE = 0.6
#: A *tight* retry (see ``repair_gaps``) is only trusted when it heard
#: essentially the whole line: that is the evidence the line really is sung
#: inside the short window and the original span was stretched over silence.
MIN_FULL_SHARE = 0.95
#: ... and when almost everything it heard belongs to that line. A window that
#: runs past the singing would otherwise win by adding the next line's words.
MIN_FULL_PRECISION = 0.8
#: Audio kept after the hole's start in a tight retry — enough for the
#: recogniser to finish the phrase it was in the middle of.
TIGHT_TAIL_SEC = 1.5
#: Cap per run: each retry is an ASR job, and a pathological file must not
#: turn one alignment into fifty recognitions.
MAX_RETRY_WINDOWS = 12


@dataclass
class Gap:
    """One implausible hole inside one lyric line's own timings."""

    line: int
    text: str
    start: float  # end of the token before the hole
    end: float  # start of the token after it
    span_start: float  # first token of the line, in this audio
    span_end: float  # last token of the line, in this audio

    @property
    def duration(self) -> float:
        return self.end - self.start


def token_duration(words: list[Word]) -> float:
    """Median width of one ASR token — the local pace of this song."""
    widths = [
        word.end - word.start
        for word in words
        if word.start is not None and word.end is not None and word.end > word.start
    ]
    return statistics.median(widths) if widths else 0.15


def gap_threshold(words: list[Word]) -> float:
    """The gap above which a hole inside a line cannot be a breath."""
    return max(GAP_MIN_SEC, GAP_FACTOR * token_duration(words))


def find_gaps(lines: list[Line], words: list[Word]) -> list[Gap]:
    """Every implausible hole inside one line's mapped span, in time order.

    The sentence is the original lyric line: the ASR result is mapped onto the
    lyric lines by the kana DP, and only a hole *inside* one line's own span
    means the recogniser lost part of that sentence. A pause *between* two
    lines is normal — that is where the singer breathes.
    """
    if not lines or not words:
        return []
    threshold = gap_threshold(words)
    gaps: list[Gap] = []
    for index, (line, (first, last, _confidence)) in enumerate(
        zip(lines, align_spans([line.text for line in lines], words))
    ):
        if first is None or last is None or last - first < 1:
            continue
        span = [word for word in words[first : last + 1] if word.start is not None and word.end is not None]
        if not span:
            continue
        for left, right in zip(span, span[1:]):
            hole = right.start - left.end
            if hole >= threshold:
                gaps.append(
                    Gap(
                        line=index,
                        text=line.text,
                        start=left.end,
                        end=right.start,
                        span_start=span[0].start,
                        span_end=span[-1].end,
                    )
                )
    return gaps


def _middle(word: Word) -> float:
    return ((word.start or 0.0) + (word.end or 0.0)) / 2


def _fold(text: str) -> str:
    return "".join(ch for ch in (text or "") if not ch.isspace())


def text_share(original: str, heard: str) -> float:
    """Share of the sentence's characters the retry actually heard."""
    left, right = _fold(original), _fold(heard)
    if not left:
        return 1.0
    if not right:
        return 0.0
    return _matched(left, right) / len(left)


def text_precision(original: str, heard: str) -> float:
    """Share of what the retry heard that belongs to the sentence.

    The mirror of ``text_share``: a window that runs past the singing can hear
    every character of the sentence *and* the next line's, which is not
    evidence that the sentence ends there.
    """
    left, right = _fold(original), _fold(heard)
    if not right:
        return 0.0
    return _matched(left, right) / len(right)


def _matched(left: str, right: str) -> int:
    return sum(
        block.size
        for block in difflib.SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks()
    )


def replace_span(
    words: list[Word], start: float, end: float, replacements: list[Word]
) -> tuple[list[Word], list[Word]] | None:
    """Swap every token inside ``[start, end]`` for the retry's tokens.

    "Inside" means overlapping the range: the retry re-heard the same audio, so
    its token for the last word of the stretch often runs a little past the
    original end. Judging by overlap keeps that word instead of dropping it.

    Returns the new list and the tokens that went in, or ``None`` when the
    retry heard nothing usable in that window.
    """
    def overlaps(word: Word) -> bool:
        return (
            word.start is not None
            and word.end is not None
            and word.start < end
            and word.end > start
        )

    inside = [
        word
        for word in replacements
        if overlaps(word) and word.text.strip()
    ]
    if not inside:
        return None
    kept = [word for word in words if not overlaps(word)]
    merged = kept + inside
    merged.sort(key=lambda word: (word.start or 0.0, word.end or 0.0))
    return merged, inside


def splice(
    words: list[Word],
    gap: Gap,
    replacements: list[Word],
    *,
    threshold: float,
    min_share: float = MIN_TEXT_SHARE,
    require_closed: bool = True,
    inside_range: tuple[float, float] | None = None,
) -> tuple[list[Word], float] | None:
    """Replace the whole sentence's tokens with what the retry heard.

    The retry ran over the entire sentence, so its tokens are the corrected
    word timings for that line: everything the first pass stamped inside the
    line's span is dropped and the new tokens take its place. Tokens outside
    the span are left alone — that is the next sentence, whose timings the
    first pass got right.

    The retry is kept only when it heard the sentence (≥ ``min_share`` of its
    characters) and no implausible hole is left *and* the original hole is at
    least halved. A retry that turns a 4.2 s hole into 2.6 s has not fixed
    anything a listener would hear; leaving the original timings alone is the
    honest outcome.

    ``inside_range`` says which of the retry's tokens count as the sentence,
    when that is not the line's own span — the tight retry (see
    ``repair_gaps``) listens to a window that ends at the hole, so the
    sentence it hears starts a little before the span (that is the padding)
    and its tokens must not be rejected for it. Tokens inside the span but
    outside that range are still dropped: they are the stretched tail this
    retry exists to remove.

    ``require_closed=False`` is for the tight retry: there the evidence is
    that the whole line fits inside a short window, not that a particular
    hole shrank.
    """
    lo, hi = inside_range if inside_range is not None else (gap.span_start, gap.span_end)
    inside = [
        word
        for word in replacements
        if word.start is not None
        and word.end is not None
        and word.text.strip()
        and lo <= _middle(word) <= hi
    ]
    if not inside:
        return None
    if text_share(gap.text, "".join(word.text for word in inside)) < min_share:
        return None

    kept = [word for word in words if not gap.span_start <= _middle(word) <= gap.span_end]
    merged = kept + inside
    merged.sort(key=lambda word: (word.start or 0.0, word.end or 0.0))

    # Only the silences *inside* the sentence matter: what is left between the
    # sentence and its neighbours is the singer's breath, exactly as before.
    ordered = sorted(inside, key=lambda word: (word.start or 0.0, word.end or 0.0))
    remaining = [
        right.start - left.end
        for left, right in zip(ordered, ordered[1:])
        if left.end is not None and right.start is not None
    ]
    after = max(remaining, default=0.0)
    if require_closed and not (after <= threshold and after <= gap.duration * IMPROVEMENT_RATIO):
        return None
    return merged, after


async def _tight_retry(
    gap: Gap,
    words: list[Word],
    transcribe_window: Callable[[float, float], Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Second attempt: cut the window at the hole and hear the line again.

    A hole the full-sentence retry cannot close is often not a lost word. The
    kana DP maps a line onto a run of tokens, and on a live take that run
    sometimes runs past the singing: the tokens after the hole belong to the
    next line, or are a stretched token over an instrumental. The line is then
    fine except that its last characters sit seconds too late.

    The full-sentence window cannot prove this — the recogniser hears the same
    tokens in the same places. A window that ends shortly after the hole starts
    can: if the recogniser hears the **whole** sentence inside it, the line
    really is sung in that short stretch, and the long tail was the error.

    Two guards keep a clipped or over-long window from winning by cheating:
    the retry must be essentially complete (``MIN_FULL_SHARE`` of the line's
    characters), so a window that cuts the sentence off cannot pass, and it
    must be essentially *only* this line (``MIN_FULL_PRECISION``), so a window
    that runs into the next line cannot pass either.
    """
    lo = max(0.0, gap.span_start - WINDOW_PAD_SEC)
    hi = gap.start + TIGHT_TAIL_SEC
    entry: dict[str, Any] = {"window": [round(lo, 3), round(hi, 3)]}
    if hi - lo < MIN_WINDOW_SEC:
        entry.update(accepted=False, reason="这句太短，收紧窗口没有意义")
        return entry
    try:
        heard = transcribe_window(lo, hi)
        if inspect.isawaitable(heard):
            heard = await heard
    except Exception as exc:  # noqa: BLE001 - a retry must never fail the run
        logger.warning("tight re-recognition of %.2f-%.2fs failed: %s", lo, hi, exc)
        entry.update(error=str(exc), accepted=False)
        return entry
    entry["heard"] = "".join(word.text for word in heard)
    share = text_share(gap.text, entry["heard"])
    entry["heard_share"] = round(share, 3)
    precision = text_precision(gap.text, entry["heard"])
    entry["heard_precision"] = round(precision, 3)
    if precision < MIN_FULL_PRECISION:
        entry.update(
            accepted=False,
            reason="收紧窗口后听到了这句之外的内容，说明这句并没有在窗口里唱完，保留原结果",
        )
        return entry
    spliced = splice(
        words,
        gap,
        heard,
        threshold=threshold,
        min_share=MIN_FULL_SHARE,
        require_closed=False,
        # The window starts before the line's span on purpose (that padding is
        # how the recogniser gets the first syllable right), so the first
        # token legitimately sits before ``span_start`` and still belongs to
        # this line.
        inside_range=(lo, hi),
    )
    if spliced is None:
        entry.update(
            accepted=False,
            reason="收紧窗口后没有听到完整的这一句，保留原结果",
        )
        return entry
    merged, after = spliced
    # The new tokens must actually sit inside the short window; a retry that
    # still stamps a character past the cut heard the tail after all.
    inside = [
        word
        for word in merged
        if word.start is not None and word.end is not None and lo <= _middle(word) <= hi
    ]
    if inside and inside[-1].end > hi:
        entry.update(accepted=False, reason="收紧窗口后仍有字落在窗口之外，保留原结果")
        return entry
    entry.update(accepted=True, words=merged, gap_after=after)
    return entry


async def repair_gaps(
    words: list[Word],
    lines: list[Line],
    transcribe_window: Callable[[float, float], Any],
    *,
    trace: Any = None,
    limit: int = MAX_RETRY_WINDOWS,
) -> tuple[list[Word], list[dict[str, Any]]]:
    """Re-recognise every sentence whose timings do not add up.

    ``transcribe_window(lo, hi)`` returns tokens with **absolute** times (a
    plain list or an awaitable). The window is the whole sentence as the ASR
    placed it, padded for context — re-recognising only the hole starves the
    recogniser of the words on either side, which is how it lost them.

    A window whose retry fails or does not help leaves the words untouched —
    the run continues either way, because a bad gap is a quality problem, not
    a reason to fail the alignment.
    """
    gaps = find_gaps(lines, words)
    if not gaps:
        return words, []
    threshold = gap_threshold(words)
    # One retry per sentence, not per hole: the retry re-recognises the whole
    # sentence anyway, so a second hole in the same line is covered by it.
    per_line: dict[int, list[Gap]] = {}
    for gap in gaps:
        per_line.setdefault(gap.line, []).append(gap)

    current = list(words)
    log: list[dict[str, Any]] = []
    for gap in list(per_line.values())[:limit]:
        lo = max(0.0, gap[0].span_start - WINDOW_PAD_SEC)
        hi = gap[0].span_end + WINDOW_PAD_SEC
        if hi - lo < MIN_WINDOW_SEC:
            hi = lo + MIN_WINDOW_SEC
        entry: dict[str, Any] = {
            "line": gap[0].line + 1,
            "text": gap[0].text,
            "gap": [round(gap[0].start, 3), round(gap[0].end, 3)],
            "gap_seconds": round(gap[0].duration, 3),
            "gaps": [[round(item.start, 3), round(item.end, 3)] for item in gap],
            "sentence": [round(gap[0].span_start, 3), round(gap[0].span_end, 3)],
            "window": [round(lo, 3), round(hi, 3)],
        }
        try:
            heard = transcribe_window(lo, hi)
            if inspect.isawaitable(heard):
                heard = await heard
        except Exception as exc:  # noqa: BLE001 - a retry must never fail the run
            logger.warning("re-recognising %.2f-%.2fs failed: %s", lo, hi, exc)
            entry.update(error=str(exc), accepted=False)
            log.append(entry)
            continue
        entry["heard"] = "".join(word.text for word in heard)
        share = text_share(gap[0].text, entry["heard"])
        entry["heard_share"] = round(share, 3)
        spliced = splice(current, gap[0], heard, threshold=threshold)
        if spliced is not None:
            current, after = spliced
            entry.update(accepted=True, gap_after=round(after, 3))
            logger.info(
                "line %d: re-recognised %.2f-%.2fs, gap %.2fs -> %.2fs",
                gap[0].line + 1,
                lo,
                hi,
                gap[0].duration,
                after,
            )
        else:
            reason = (
                "重识别只听到这句的一部分，保留原结果"
                if share < MIN_TEXT_SHARE
                else "重识别没有填补这句的间隔"
            )
            entry.update(accepted=False, reason=reason)
            # Second attempt, tighter: a hole that the full-sentence retry
            # cannot close is sometimes not a lost word at all — the line was
            # simply mapped to a span that runs into silence, and the tokens
            # after the hole belong to the next line (or to nothing). Cut the
            # window at the hole's start and ask again: if the recogniser now
            # hears the *whole* sentence inside that short window, the line
            # really is sung there and the long span was the error.
            tight = await _tight_retry(gap[0], current, transcribe_window, threshold=threshold)
            entry["tight"] = tight
            if tight.get("accepted"):
                current, after = tight["words"], tight["gap_after"]
                entry.update(accepted=True, gap_after=round(after, 3), mode="tight")
                logger.info(
                    "line %d: tight re-recognition %.2f-%.2fs fits the whole sentence",
                    gap[0].line + 1,
                    tight["window"][0],
                    tight["window"][1],
                )
        log.append(entry)

    accepted = sum(1 for entry in log if entry.get("accepted"))
    if trace is not None:
        trace.step(
            "asr_retry",
            {
                "threshold": round(gap_threshold(words), 3),
                "found": len(gaps),
                "attempted": len(log),
                "accepted": accepted,
                "windows": log,
            },
        )
    if accepted:
        current.sort(key=lambda word: (word.start or 0.0, word.end or 0.0))
        return current, log
    # Same list object back: the caller uses identity to tell "nothing changed".
    return words, log


def repair_note(log: list[dict[str, Any]]) -> str | None:
    """Run notes naming the lines whose timings were re-recognised.

    A line whose retry did not help is reported too: the timings there are
    still the original ones, and a listener deserves to know that the gap was
    noticed and could not be fixed by re-recognising (usually a long sustained
    note the aligner does not split, or words the recogniser drops every time).
    """
    accepted = [entry for entry in log if entry.get("accepted")]
    tight = [entry for entry in accepted if entry.get("mode") == "tight"]
    replaced = [entry for entry in accepted if entry.get("mode") != "tight"]
    kept = [entry for entry in log if not entry.get("accepted")]
    parts: list[str] = []
    if replaced:
        numbers = "、".join(str(entry["line"]) for entry in replaced)
        parts.append(f"第 {numbers} 行的识别时间戳间隔异常，已重新识别整句并替换该句的时间戳")
    if tight:
        numbers = "、".join(str(entry["line"]) for entry in tight)
        parts.append(
            f"第 {numbers} 行的时间跨度延伸到了人声之外，已用收紧的窗口重听，"
            "该句时间按实际演唱处重新标记"
        )
    if kept:
        numbers = "、".join(str(entry["line"]) for entry in kept)
        parts.append(f"第 {numbers} 行的识别时间戳间隔异常，重新识别整句后仍未填补（可能是长音或漏识别），时间保持原样")
    return "；".join(parts) if parts else None


async def resolve_extra_segments(
    suspects: list[tuple[Line, int, str]],
    words: list[Word],
    transcribe_window: Callable[[float, float], Any],
    *,
    trace: Any = None,
) -> list[Word] | None:
    """Re-recognise the stretches that look like a lyric line the ASR mangled.

    A segment the lyrics do not cover is not necessarily new content. When the
    recogniser hears a known line badly enough, the kana DP cannot place it,
    the line stays unmatched and its words look like a new sentence. The model
    spots the resemblance (text and reading, or just a fragment of a
    neighbouring line), but the fix is not to add a line — it is to hear that
    stretch again so the DP can map it onto the lyric it really is.

    Returns the new token list, or ``None`` when nothing was changed. A window
    whose retry fails changes nothing: the segment then stays an extra line,
    which is the honest fallback.
    """
    current = list(words)
    changed = False
    log: list[dict[str, Any]] = []
    for segment, line_index, line_text in suspects:
        if segment.start is None or segment.end is None:
            continue
        lo = max(0.0, segment.start - WINDOW_PAD_SEC)
        hi = segment.end + WINDOW_PAD_SEC
        entry: dict[str, Any] = {
            "line": line_index + 1,
            "line_text": line_text,
            "heard_as": segment.text,
            "window": [round(lo, 3), round(hi, 3)],
        }
        try:
            heard = transcribe_window(lo, hi)
            if inspect.isawaitable(heard):
                heard = await heard
        except Exception as exc:  # noqa: BLE001 - never fail a run over a retry
            logger.warning("re-recognising %.2f-%.2fs failed: %s", lo, hi, exc)
            entry.update(error=str(exc), accepted=False)
            log.append(entry)
            continue
        entry["heard"] = "".join(word.text for word in heard)
        replaced = replace_span(current, segment.start, segment.end, heard)
        if replaced is None:
            entry.update(accepted=False, reason="重识别没有听到这段内容")
        else:
            current, inside = replaced
            changed = True
            entry.update(accepted=True, heard_again="".join(word.text for word in inside))
            logger.info(
                "line %d: re-recognised a segment that looked like new content (%.2f-%.2fs)",
                line_index + 1,
                lo,
                hi,
            )
        log.append(entry)

    if trace is not None and log:
        trace.step(
            "extra_retry",
            {
                "attempted": len(log),
                "accepted": sum(1 for entry in log if entry.get("accepted")),
                "windows": log,
            },
        )
    if not changed:
        return None
    current.sort(key=lambda word: (word.start or 0.0, word.end or 0.0))
    return current


def prepared_audio(work_dir: Path, audio_path: Path) -> Path:
    """The 16 kHz mono vocals WAV the ASR windows were cut from.

    ``transcribe_audio`` writes it on every run, and a targeted re-recognition
    must slice the same audio so the retry hears what the first pass heard.
    """
    prepared = work_dir / "asr-input.wav"
    if prepared.exists():
        return prepared
    return to_asr_wav(audio_path, prepared)


async def retranscribe_span(
    asr_wav: Path,
    work_dir: Path,
    asr_provider,
    lo: float,
    hi: float,
    language: str | None,
) -> list[Word]:
    """Run the ASR over one slice of the prepared audio; absolute timings."""
    target = work_dir / f"retry-{int(lo * 1000):08d}.wav"
    cut_audio(asr_wav, target, lo, max(hi - lo, 0.2))
    result = await asr_provider.transcribe(target, language)
    return [
        Word(
            text=word.text,
            start=round(word.start + lo, 3) if word.start is not None else None,
            end=round(word.end + lo, 3) if word.end is not None else None,
            source="asr",
        )
        for word in result.words
        if word.text.strip() and word.start is not None and word.end is not None
    ]
