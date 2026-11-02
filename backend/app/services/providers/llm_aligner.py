"""LLM-guided lyric-to-ASR alignment.

The ASR service returns timed tokens (one per CJK character, one per latin
word). Sung lyrics never match ASR text verbatim — the recogniser mishears
words constantly — so a plain string diff is useless.

The pipeline that works in practice:

1. **Kana readings.** Both the lyric lines and the ASR tokens are converted to
   hiragana with pykakasi. Readings survive misrecognised kanji far better
   than the written text ("残機" vs "ざんき"), and they make the comparison
   pronunciation-based, which is what the singer actually produced.
2. **Global character DP.** The whole lyric kana stream is aligned against the
   whole ASR kana stream (monotonic, gap penalties), which yields one ASR word
   span per lyric line plus a per-line match confidence.
3. **LLM verification.** The LLM cannot read garbled ASR text in isolation,
   but it reliably picks the right line when shown a few candidates. For each
   lyric line we show it the kana of its own span and of its neighbours'
   spans, shuffled, and it picks the one that is this line. That flags bad
   mappings for review; it never moves a timing.

When LRCLIB supplied a synced entry, its timestamps are used as a *prior* for
lines the global DP could not place at all (a live version can reorder or repeat
a verse, which leaves the monotonic DP without room). Those lines are searched
only inside a window around the reference stamp and are flagged for review.

Whatever the ASR sang that no lyric line covers is kept as extra lines
(``source="asr-extra"``) instead of being dropped: on a live recording that is
usually real singing, and it is the recogniser's own text.

Timings always come from the ASR forced aligner: characters the ASR never
heard are interpolated between timed neighbours and marked
``source="interpolated"``, and lines with no ASR counterpart stay untimed.

The exception is a *partial cover*: when the recording sings only part of the
song (a TV-size edit, one verse and the chorus), the run of lines it never
sings is left out of the timeline instead of shown untimed — see
:mod:`app.services.partial` for the rule.
"""

from __future__ import annotations

import difflib
import inspect
import logging
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from ...schemas import Line, Timeline, Word
from ..llm import LLMClient, LLMError
from .base import AlignerProvider

logger = logging.getLogger("lyrictrace.aligner")

# Global DP scoring: a matched kana is worth 2, a mismatch costs 1.2 and a gap
# 0.9 — deliberately forgiving, because the ASR stream is noisy.
MATCH_SCORE = 2.0
MISMATCH_SCORE = -1.2
GAP_SCORE = -0.9

VERIFY_BATCH = 8
VERIFY_MAX_FAILURES = 2
MIN_SPAN_CONFIDENCE = 0.0

# Below this share of matched tokens the per-character timings are noise and
# the line is spread over its span instead (see _time_line). Measured on the
# test song: 0.4 keeps line 20 at 81.1 s (was 88.8 s) and line 36 at 140.7 s
# (was 149.7 s), and costs no other line.
MIN_MATCH_DENSITY = 0.4

# A line the DP could not place is looked for around its LRCLIB reference stamp,
# but only inside this many seconds: the live version may be slower, not a
# different song.
REFERENCE_PAD_SEC = 12.0
# Below this share of matched kana a reference-guided recovery is not trusted.
REFERENCE_MIN_CONFIDENCE = 0.5

# An uncovered ASR segment shorter than this is treated as recogniser noise
# (a stray syllable over an instrumental) rather than real singing.
EXTRA_MIN_CHARS = 4
EXTRA_MIN_SEC = 1.0
# ... but a short segment carrying several distinct characters is a whole
# phrase nevertheless: "唱给大家听" is five sung characters in 0.72 s, and the
# recogniser does not invent five different characters out of an instrument.
# Dropping it would lose the tail of the live intro.
EXTRA_MIN_SEC_SHORT = 0.4
# The model's verdict is not the last word. A segment this long and this varied
# is real speech in the recording even when the model calls it noise — the
# recogniser does not invent a sentence out of a loud instrumental. Keeping one
# line too many costs the user a click; dropping a spoken intro or an MC section
# loses content they cannot get back.
EXTRA_KEEP_CHARS = 6
EXTRA_KEEP_SEC = 1.2
# ... but the same syllable over and over is exactly what a sustained
# instrument looks like once misheard as text, so that stays a "drop".
EXTRA_KEEP_DISTINCT = 3
# A short segment this close to content that was kept is the rest of the same
# sentence: ``extra_lines`` splits at any pause longer than a breath, so a
# spoken intro arrives as several pieces, and the short ones must not be
# dropped just for being short.
EXTRA_JOIN_SEC = 3.0
# How far around a segment the model may look for the lyric line it is. A line
# from another verse is not the same sentence — the singer repeating the title
# in the intro is new content, not a misrecognition of the chorus.
CONTEXT_SEC = 25.0
CONTEXT_LINES = 4
# A stretch of ASR tokens that touches a lyric line on one side, or fills the
# gap between two lines, is part of the lyric singing: the recogniser heard the
# line badly and put the words it could not place in their own run. The lyric
# text is the authority (it is far more reliable than the transcript), so those
# tokens are folded back into that line — which then gets their time — instead
# of becoming a new "extra" line. A run this long, or this far from any line,
# is a sentence of its own: real extra content.
EXTEND_GAP_SEC = 1.2
EXTEND_MAX_WORDS = 14
# A line counts as "heard" only when the DP matched enough of its kana: a span
# built from one stray character is not evidence that the line was sung, and
# treating it as evidence splits a cut verse into pieces. Measured on 怪物
# (sung lines 0.64–1.0, a line nobody sang 0.06–0.33).
ABSENT_HEARD_CONFIDENCE = 0.5

KANA_FOLD = {chr(ord(ch) - 0x60): ch for ch in "ァィゥェォッャュョヮヵヶ"}
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
KANA_RE = re.compile(r"[^ぁ-んー]+")

try:  # pykakasi is optional: without it we fall back to the written text
    import pykakasi

    _KAKASI = pykakasi.kakasi()
except Exception:  # noqa: BLE001 - any import/config failure degrades gracefully
    _KAKASI = None
    logger.warning("pykakasi is not installed; falling back to written-text alignment")


def fold(text: str) -> str:
    """Lowercase, NFKC, katakana -> hiragana, alphanumerics only."""
    out = []
    for ch in unicodedata.normalize("NFKC", text).lower():
        ch = KANA_FOLD.get(ch, ch)
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def reading(text: str) -> str:
    """Hiragana reading of ``text`` (kanji resolved, latin dropped)."""
    if _KAKASI is None:
        return fold(text)
    try:
        hira = "".join(item["hira"] for item in _KAKASI.convert(text))
    except Exception:  # noqa: BLE001 - never let a converter glitch kill alignment
        return fold(text)
    return KANA_RE.sub("", hira)


def key_text(text: str) -> str:
    """The comparison key for one token: its reading, else its folded text."""
    return reading(text) or fold(text)


def is_cjk(ch: str) -> bool:
    return bool(CJK_RE.match(ch))


@dataclass
class LineSpan:
    index: int
    first: int
    last: int
    confidence: float = 0.0


@dataclass
class AlignReport:
    mapped: dict[int, LineSpan] = field(default_factory=dict)
    unmatched: list[int] = field(default_factory=list)
    weak: list[int] = field(default_factory=list)
    recovered: list[int] = field(default_factory=list)
    """Lines the DP missed and the LRCLIB reference located instead."""
    extra_suspects: list[tuple[Line, int, str]] = field(default_factory=list)
    """Segments the LLM thinks are really lyric line N (text included)."""
    absent: list[int] = field(default_factory=list)
    """Lines this recording does not sing at all (a partial cover; see partial)."""
    absent_detail: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    llm_failures: int = 0
    notes: list[str] = field(default_factory=list)


class LLMLyricAligner(AlignerProvider):
    name = "llm-aligner"
    description = (
        "Kana-reading DP maps lyric lines onto ASR token spans, the LLM verifies "
        "each line; timings come from the ASR forced alignment"
    )

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def available(self) -> bool:
        return self.client.available()

    def unavailable_detail(self) -> str:
        return (
            "Set LYRICTRACE_LLM_BASE_URL / LYRICTRACE_LLM_MODEL (and the ASR service URL) "
            "to enable LLM-guided alignment."
        )

    async def align(
        self,
        audio_path,  # noqa: ANN001 - the ASR already produced the timings
        lines: list[Line],
        *,
        asr_words: list[Word] | None = None,
        language: str | None = None,
        on_progress: Callable[[float, str], None] | None = None,
        reference: dict[int, Line] | None = None,
        keep_extras: bool = True,
        trace: Any = None,
        repair: Callable[[list[Line], list[Word]], list[Word]] | None = None,
        resolve_extra: Callable[[list[tuple[Line, int, str]], list[Word]], Any] | None = None,
        detect_extras: bool = False,
    ) -> Timeline:
        """Place every lyric line on the ASR word stream.

        ``keep_extras`` decides whether uncovered sung content is kept as extra
        lines at all; ``detect_extras`` decides whether the LLM is asked to
        judge those segments (noise vs. real singing) and whether a segment that
        turns out to be a mangled lyric line is re-heard. With both off the
        timeline is the original lyrics only — the recogniser is not allowed to
        add a line of its own.
        """
        if not asr_words:
            raise LLMError("Alignment needs ASR word timings; run transcription first.")
        if not self.available():
            raise LLMError(self.unavailable_detail())

        report = self.map_lines(lines, asr_words, on_progress=on_progress, trace=trace)
        if repair is not None:
            # Once the lines are mapped we can see where the ASR's own timings
            # do not add up (a multi-second hole inside one sung line). Fixing
            # those means re-recognising audio, which the caller owns; if it
            # changed anything, the mapping is redone on the better tokens.
            repaired = repair(lines, asr_words)
            if inspect.isawaitable(repaired):
                repaired = await repaired
            if repaired != asr_words:
                asr_words = repaired
                report = self.map_lines(lines, asr_words, on_progress=on_progress)
        if reference:
            self.recover_with_reference(lines, asr_words, report, reference)
        # Before anything else uses the spans: a run of tokens the DP left out
        # between two lines is that line's words misheard, not new content.
        absorb_strays(lines, asr_words, report, trace=trace)
        separate_spans(report)
        self.verify_lines(lines, asr_words, report, on_progress=on_progress, trace=trace)
        flag_collisions(lines, report)
        # A cover that sings only part of the song leaves a run of lines nothing
        # heard. Those are not failures to report: the recording simply does not
        # contain them, and they are dropped from the timeline (see partial).
        found = self.mark_absent(lines, asr_words, report, trace=trace)
        report.notes.extend(found.notes)
        coverage = coverage_note(report, asr_words)
        if coverage:
            report.notes.append(coverage)
        extras = extra_lines(asr_words, report) if keep_extras else []
        ignored = 0
        if extras:
            # ``detect_extras`` off: the recogniser may not add a line of its
            # own, but its uncovered segments are still worth asking about —
            # one of them may be a lyric line it mangled, and re-hearing that
            # audio is how the line gets placed at all. So the LLM is still
            # asked which segments are really nearby lyric lines; the kept
            # extras are dropped, only the re-hearing is used.
            extras, suspects = self.filter_extras(
                extras,
                lines,
                report,
                trace=trace,
                words=asr_words,
                lyrics_only=not detect_extras,
            )
            if suspects and resolve_extra is not None:
                # The model says these "new" segments are really lyric lines the
                # recogniser mangled. That is an ASR error, not a new sentence:
                # re-recognise their audio, then let the DP try again on the
                # better tokens. A segment that still cannot be placed stays an
                # extra — nothing is ever dropped on the model's say-so alone.
                resolved = resolve_extra(suspects, asr_words)
                if inspect.isawaitable(resolved):
                    resolved = await resolved
                if resolved is not None and resolved != asr_words:
                    asr_words = resolved
                    report = self.map_lines(lines, asr_words, on_progress=on_progress)
                    separate_spans(report)
                    extras = extra_lines(asr_words, report) if keep_extras else []
                    if extras:
                        extras, suspects = self.filter_extras(
                            extras,
                            lines,
                            report,
                            trace=trace,
                            words=asr_words,
                            lyrics_only=not detect_extras,
                        )
            if extras:
                # Original lyric lines always keep the source's own line breaks;
                # only content the original lyrics do not cover is re-split, and
                # that is a judgement call the LLM makes from what was sung.
                extras = self.split_extras(extras, asr_words, report, lines, trace=trace)
            elif not detect_extras:
                ignored = len(extra_lines(asr_words, report))
        if ignored:
            report.notes.append(
                f"未开启「识别现场加唱」：识别到 {ignored} 段原始歌词里没有的内容，"
                "未加入时间轴；如果其中有识别错误的歌词，已重新识别并按原歌词对齐"
            )
        timeline = build_timeline(lines, asr_words, report, extras=extras)
        from ..projects import compute_timeline_status

        timeline.status = compute_timeline_status(timeline)
        if trace is not None:
            trace.step(
                "timeline",
                {
                    "lines": [
                        {
                            "index": index,
                            "text": line.text,
                            "start": line.start,
                            "end": line.end,
                            "source": line.source,
                            "heard": line.heard,
                            "words": len(line.words),
                        }
                        for index, line in enumerate(timeline.lines)
                    ],
                    "status": timeline.status,
                    "notes": timeline.notes,
                },
            )
        logger.info(
            "aligned %d/%d lines (unmatched: %s, weak: %s, recovered: %s, extras: %d, "
            "llm calls: %d, failures: %d)",
            len(report.mapped),
            len(lines),
            report.unmatched,
            report.weak,
            report.recovered,
            len(extras),
            report.llm_calls,
            report.llm_failures,
        )
        return timeline

    # ------------------------------------------------------------------
    # Step 1: line -> ASR span (kana DP)
    # ------------------------------------------------------------------

    def map_lines(
        self,
        lines: list[Line],
        words: list[Word],
        *,
        on_progress: Callable[[float, str], None] | None = None,
        trace: Any = None,
    ) -> AlignReport:
        report = AlignReport()
        if on_progress:
            on_progress(0.1, "正在按读音对齐歌词与识别结果")

        spans = align_spans([line.text for line in lines], words)
        for index, (first, last, confidence) in enumerate(spans):
            if first is None or last is None or confidence <= MIN_SPAN_CONFIDENCE:
                continue
            report.mapped[index] = LineSpan(index, first, last, confidence)

        report.unmatched = [i for i in range(len(lines)) if i not in report.mapped]
        if trace is not None:
            trace.step(
                "line_spans",
                [
                    {
                        "index": index,
                        "text": line.text,
                        "reading": key_text(line.text),
                        "first": first,
                        "last": last,
                        "confidence": round(confidence, 3),
                        "span_text": "".join(w.text for w in words[first : last + 1]) if first is not None else "",
                        "span_start": words[first].start if first is not None else None,
                        "span_end": words[last].end if last is not None else None,
                        "unmatched": first is None or last is None,
                    }
                    for index, (line, (first, last, confidence)) in enumerate(zip(lines, spans))
                ],
            )
        if on_progress:
            on_progress(0.5, f"已定位 {len(report.mapped)}/{len(lines)} 行")
        return report

    # ------------------------------------------------------------------
    # Step 1a: lines the monotonic DP could not place, found near the
    #          LRCLIB reference stamp (live versions reorder/repeat verses)
    # ------------------------------------------------------------------

    def recover_with_reference(
        self,
        lines: list[Line],
        words: list[Word],
        report: AlignReport,
        reference: dict[int, Line],
    ) -> None:
        """Search a local window for lines the global DP left behind.

        The reference stamps are the studio recording's, so they are only a
        coarse hint: each unmatched line is looked for inside
        ``REFERENCE_PAD_SEC`` around its reference start, and the found span
        must still match on reading. Recovered lines are flagged for review
        because their position came from the API rather than from a clean DP.
        """
        if not report.unmatched or not reference:
            return
        covered = bytearray(len(words))
        for span in report.mapped.values():
            for index in range(max(span.first, 0), min(span.last + 1, len(words))):
                covered[index] = 1

        for index in list(report.unmatched):
            ref = reference.get(index)
            if ref is None or ref.start is None:
                continue
            lo = ref.start - REFERENCE_PAD_SEC
            hi = (ref.end if ref.end is not None else ref.start + REFERENCE_PAD_SEC) + REFERENCE_PAD_SEC
            candidates = [
                position
                for position, word in enumerate(words)
                if word.start is not None
                and word.end is not None
                and word.end > lo
                and word.start < hi
                and not covered[position]
            ]
            if len(candidates) < 2:
                continue
            window = [words[position] for position in candidates]
            first, last, confidence = align_spans([lines[index].text], window)[0]
            if first is None or last is None or confidence < REFERENCE_MIN_CONFIDENCE:
                continue
            span = LineSpan(index, candidates[first], candidates[last], confidence)
            for position in range(span.first, span.last + 1):
                if covered[position]:
                    span = None  # type: ignore[assignment]
                    break
            if span is None:
                continue
            for position in range(span.first, span.last + 1):
                covered[position] = 1
            report.mapped[index] = span
            report.recovered.append(index)
            report.unmatched.remove(index)
            lines[index].source = "asr-ref"
            if index not in report.weak:
                report.weak.append(index)
            logger.info(
                "line %d recovered near the LRCLIB reference (confidence %.2f)",
                index + 1,
                confidence,
            )
        report.weak.sort()
        report.recovered.sort()

    # ------------------------------------------------------------------
    # Step 1b: LLM picks, per line, which candidate span really is that line
    # ------------------------------------------------------------------

    def verify_lines(
        self,
        lines: list[Line],
        words: list[Word],
        report: AlignReport,
        on_progress: Callable[[float, str], None] | None = None,
        trace: Any = None,
    ) -> None:
        """Confirm each mapping by relative comparison, flagging suspicious ones.

        Garbled ASR text is unreadable in isolation, but the model reliably
        picks the right line out of a small shuffled candidate set. Verdicts
        only ever flag a line for review — they never move a timing, and the
        lyric text is never rewritten.
        """
        indices = sorted(report.mapped)
        if not indices:
            return
        heard = {index: heard_text(report.mapped[index], words) for index in indices}

        picks: dict[int, int] = {}
        asked: list[dict[str, Any]] = []
        failures = 0
        batches = (len(indices) + VERIFY_BATCH - 1) // VERIFY_BATCH
        for batch_number, start in enumerate(range(0, len(indices), VERIFY_BATCH)):
            batch = indices[start : start + VERIFY_BATCH]
            try:
                picks.update(self._ask_picks(batch, lines, heard, report.mapped))
                report.llm_calls += 1
            except LLMError as exc:
                failures += 1
                report.llm_failures += 1
                logger.warning("LLM verification failed for lines %s: %s", [i + 1 for i in batch], exc)
                if failures >= VERIFY_MAX_FAILURES:
                    logger.warning("too many LLM failures, skipping the rest of verification")
                    break
            if on_progress:
                on_progress((batch_number + 1) / batches, f"LLM 校验 {batch_number + 1}/{batches} 批")

        # Repeated lyric lines (a chorus comes back three times) make the
        # verification question ambiguous: two candidate spans read the same,
        # so the model's pick between them carries no information. Treat a pick
        # of an indistinguishable span as a confirmation instead of a flag —
        # flagging every repeated line made the review list unusable.
        repeats = Counter(fold(line.text) for line in lines)
        indistinguishable: set[int] = set()
        for index, target in picks.items():
            if target == index or index not in heard or target not in heard:
                continue
            if repeats.get(fold(lines[index].text), 0) < 2 and repeats.get(
                fold(lines[target].text), 0
            ) < 2:
                continue
            if key_text(heard[index]) == key_text(heard[target]):
                indistinguishable.update({index, target})
                logger.info(
                    "lines %d and %d read the same; the pick carries no information", index + 1, target + 1
                )

        # A pair of lines that each picked the other's span is suspicious, but
        # actually swapping the spans measured worse than leaving them alone
        # (it pushed line 20 from 81.1 s to 88.8 s against a reference of
        # 83.2 s), so mutual picks only flag the pair for review. Verdicts
        # never move a timing on their own.
        mutual: set[int] = set()
        for index, target in picks.items():
            if target == index or index in mutual or target in mutual:
                continue
            if index in indistinguishable or target in indistinguishable:
                continue
            if picks.get(target) == index and index in report.mapped and target in report.mapped:
                mutual.update({index, target})
                logger.info("lines %d and %d each picked the other's span", index + 1, target + 1)

        for index in indices:
            if index in report.recovered:
                # Already flagged: its position came from the API reference.
                continue
            target = picks.get(index)
            if index in mutual:
                lines[index].source = "asr-partial"
                report.weak.append(index)
            elif target is None:
                lines[index].source = "asr"
            elif target == index or index in indistinguishable:
                lines[index].source = "asr"
            else:
                lines[index].source = "asr-bad"
                if index not in report.weak:
                    report.weak.append(index)
        report.weak.sort()

        if trace is not None:
            trace.step(
                "llm_verify",
                {
                    "calls": report.llm_calls,
                    "failures": report.llm_failures,
                    "mutual": sorted(mutual),
                    "indistinguishable": sorted(indistinguishable),
                    "picks": [
                        {
                            "index": index,
                            "text": lines[index].text,
                            "heard": heard.get(index, ""),
                            "picked": picks.get(index),
                            "picked_heard": heard.get(picks[index]) if index in picks else None,
                            "verdict": lines[index].source,
                        }
                        for index in indices
                    ],
                },
            )

    # ------------------------------------------------------------------
    # Step 1c: ASR segments no lyric line covers — keep or drop?
    # ------------------------------------------------------------------

    def filter_extras(
        self,
        extras: list[Line],
        lines: list[Line],
        report: AlignReport,
        trace: Any = None,
        words: list[Word] | None = None,
        lyrics_only: bool = False,
    ) -> tuple[list[Line], list[tuple[Line, int, str]]]:
        """Decide which uncovered ASR segments are real content.

        A live recording adds ad-libs, an extra chorus, a spoken intro and MC
        chatter; the recogniser also emits garbage during a loud instrumental.
        The two look alike as text, so the LLM judges them against the original
        lyrics. Its verdict only binds on short fragments: a segment long and
        varied enough to be a sentence is real speech whatever the model says
        (see ``_real_speech``), and a segment the model can name as a lyric
        line is kept too — that only decides whether to re-hear it. Dropping is
        therefore reserved for short noise, never for "this line appears
        elsewhere in the lyrics". When the model is unavailable or fails, only
        clearly substantial segments (a few characters, more than a second
        long) are kept — never silence, never a single stray syllable.

        ``lyrics_only`` is the project's 「识别现场加唱」 switch being off: no
        uncovered segment is kept as a line, so the timeline holds the original
        lyrics and nothing else. The second half still runs — a segment the
        model can name as a nearby lyric line is a line the recogniser mangled,
        and the caller re-hears its audio so the DP can place it.

        Returns the kept segments plus the ones the model thinks are really a
        lyric line the DP lost (``(extra, line index)``). Those are a different
        problem — an ASR error, not a new sentence — and the caller re-recognises
        their audio before falling back to keeping them as extras.
        """
        substantial = [extra for extra in extras if is_substantial(extra)]
        if not substantial:
            return [], []
        if not self.available():
            if trace is not None:
                trace.step("llm_extras", {"skipped": "LLM 不可用，只保留够长的片段", "kept": [e.text for e in substantial]})
            return ([] if lyrics_only else substantial), []
        try:
            keep, suspects = self._ask_extras(substantial, lines, report, words)
        except LLMError as exc:
            report.llm_failures += 1
            logger.warning("LLM could not judge the extra segments: %s", exc)
            return ([] if lyrics_only else substantial), []
        report.llm_calls += 1
        # A "drop" verdict is only trusted for short fragments: a long, varied
        # segment is real speech the recogniser could not have invented, so it
        # is kept even when the model calls it noise. This is what stops a
        # spoken intro or an MC section from being lost.
        forced = [index for index, extra in enumerate(substantial) if _real_speech(extra)]
        keep = keep | set(forced)
        # A segment the model can name as a lyric line is real content by
        # definition: either the recogniser mangled that line (re-hear it) or
        # the singer really said it again — an intro mentioning the title, a
        # repeated chorus. Neither is a reason to drop it, so the "is it a
        # lyric line" answer never removes content, only routes it.
        keep = keep | set(suspects)
        kept = [extra for index, extra in enumerate(substantial) if index in keep]
        # ... and a short fragment right next to a kept one is the rest of the
        # same sentence ("行" after a long clause). The recogniser splits on
        # breaths, not on meaning, so dropping the tail of an intro would leave
        # the line truncated. Such tails are too short to reach the model at
        # all, so they are joined here rather than in the prompt.
        tails = _next_to_kept(extras, {id(extra) for extra in kept})
        if tails:
            kept_ids = {id(extra) for extra in kept} | tails
            kept = [extra for extra in extras if id(extra) in kept_ids]
        # Only a line the DP failed to place can be recovered this way: if the
        # line already has a span, moving it is a different (and riskier) fix.
        unresolved = set(report.unmatched)
        flagged = [
            (substantial[index], line_index, lines[line_index].text)
            for index, line_index in suspects.items()
            if index in keep and line_index in unresolved
        ]
        if lyrics_only:
            # 「识别现场加唱」 off: the recogniser does not get to add lines.
            # Only the mangled-lyric-line route survives, and it only reaches
            # lines the DP could not place at all — re-hearing that audio and
            # trying again is how such a line gets a time.
            if trace is not None:
                trace.step(
                    "llm_extras",
                    {
                        "skipped": "未开启「识别现场加唱」，识别出的额外内容不加入时间轴",
                        "ignored": [extra.text for extra in kept],
                        "named": [
                            {"text": extra.text, "line": line_index + 1}
                            for extra, line_index, _text in flagged
                        ],
                    },
                )
            return [], flagged
        if trace is not None:
            trace.step(
                "llm_extras",
                {
                    "candidates": [
                        {"text": extra.text, "start": extra.start, "end": extra.end}
                        for extra in substantial
                    ],
                    "keep": sorted(index + 1 for index in keep),
                    "kept": [extra.text for extra in kept],
                    "forced": sorted(index + 1 for index in forced),
                    "forced_note": (
                        "这些片段足够长、又不是同一个字反复，一定是真人说话/演唱，即使模型判为噪声也保留"
                        if forced
                        else None
                    ),
                    "joined": [extra.text for extra in extras if id(extra) in tails],
                    "joined_note": (
                        "这些短片段紧挨着被保留的内容，是同一句话被停顿切开的尾巴，一并保留"
                        if tails
                        else None
                    ),
                    "dropped": [
                        extra.text for index, extra in enumerate(substantial) if index not in keep
                    ],
                    "suspects": [
                        {
                            "text": extra.text,
                            "line": line_index + 1,
                            "line_text": lines[line_index].text,
                        }
                        for extra, line_index, _text in flagged
                    ],
                    "named": [
                        {
                            "text": substantial[index].text,
                            "line": line_index + 1,
                            "line_text": lines[line_index].text,
                            "action": "重新识别这段音频" if line_index in unresolved else "那一行已有时间，按重复演唱保留",
                        }
                        for index, line_index in sorted(suspects.items())
                    ],
                },
            )
        return kept, flagged

    def mark_absent(
        self,
        lines: list[Line],
        words: list[Word],
        report: AlignReport,
        trace: Any = None,
    ) -> Any:
        """Find the lyric lines this recording does not sing (partial covers).

        A cover may sing one verse and the chorus of a longer song. Every line
        is still in the lyrics, so the run that was never sung comes back
        ``unmatched`` — indistinguishable, row by row, from a line the
        recogniser lost. The decision is therefore made on the *audio*: a run of
        unheard lines whose neighbours leave no room for them was not sung, and
        dropping it is right. A single unheard line inside a sung verse stays
        (``partial.find_absent`` explains the rule).
        """
        from ..partial import find_absent

        found = find_absent(
            heard=[
                index in report.mapped and report.mapped[index].confidence > ABSENT_HEARD_CONFIDENCE
                for index in range(len(lines))
            ],
            spans=[
                (
                    (words[span.first].start, words[span.last].end)
                    if index in report.mapped
                    and report.mapped[index].confidence > ABSENT_HEARD_CONFIDENCE
                    and 0 <= report.mapped[index].first < len(words)
                    and 0 <= report.mapped[index].last < len(words)
                    else None
                )
                for index in range(len(lines))
                for span in [report.mapped.get(index)]
            ],
            chars=[len(key_text(line.text)) for line in lines],
            words=words,
            covered=_covered_tokens(
                words, report, [index in report.mapped for index in range(len(lines))]
            ),
        )
        if not found:
            if trace is not None:
                trace.step(
                    "partial_cover",
                    {"absent": [], "note": "没有成段的「这段录音没唱」的行，歌词全部保留"},
                )
            return found
        report.absent = list(found.indices)
        report.absent_detail = list(found.detail)
        report.unmatched = [index for index in report.unmatched if index not in set(found.indices)]
        if trace is not None:
            trace.step(
                "partial_cover",
                {
                    "absent": found.detail,
                    "lines": [index + 1 for index in found.indices],
                    "texts": [lines[index].text for index in found.indices],
                    "note": "这些歌词行这段录音没有唱（翻唱只唱了一部分），已从时间轴去掉",
                },
            )
        logger.info(
            "partial cover: %d lines are not sung in this recording (%s)",
            len(found.indices),
            found.runs,
        )
        return found

    def split_extras(
        self,
        extras: list[Line],
        words: list[Word],
        report: AlignReport,
        lines: list[Line],
        trace: Any = None,
    ) -> list[Line]:
        """Re-line the sung-but-not-in-the-lyrics content, with the LLM.

        Original lyric lines always keep the source's own line breaks; this only
        touches the extra content, whose only structure is the ASR's pauses.
        The model sees the timed tokens and groups them into lyric lines the way
        a transcriber would. The grouping is validated (a partition of the
        tokens, in order) and falls back to the pause-based grouping on any
        problem, so nothing is ever lost or invented.
        """
        if not extras:
            return extras
        tokens = [word for extra in extras for word in extra.words]
        if len(tokens) < 2 or not self.available():
            return extras
        try:
            groups = self._ask_line_breaks(tokens, lines)
        except LLMError as exc:
            report.llm_failures += 1
            logger.warning("LLM could not split the extra segments: %s", exc)
            return extras
        if groups is None:
            return extras
        report.llm_calls += 1

        split: list[Line] = []
        for group in groups:
            picked = [tokens[index] for index in group]
            text = "".join(word.text for word in picked).strip()
            if not text:
                continue
            split.append(
                Line(
                    text=text,
                    start=picked[0].start,
                    end=picked[-1].end,
                    words=[word.model_copy() for word in picked],
                    source="asr-extra",
                    note="这一行是 ASR 听到、但原始歌词里没有的内容（现场加唱、口播或即兴，由 LLM 断句）",
                )
            )
        if not split:
            return extras
        if trace is not None:
            trace.step(
                "extras",
                {
                    "before": [extra.text for extra in extras],
                    "after": [line.text for line in split],
                },
            )
        return split

    def _ask_line_breaks(self, tokens: list[Word], lines: list[Line]) -> list[list[int]] | None:
        """Ask the model how to break the extra tokens into lyric lines."""
        lyric_sample = [line.text for line in lines if line.text.strip()][:40]
        blocks = [
            f"{index + 1}. [{format_seconds(word.start)}] {word.text}"
            for index, word in enumerate(tokens)
        ]
        prompt = (
            "这是一首歌的原始歌词（节选）：\n"
            + "\n".join(lyric_sample)
            + "\n\n下面是语音识别在音频里听到、但原始歌词里没有的演唱内容，按识别 token 编号（带时间）。\n"
            "请按歌词的自然断句把这些 token 分成若干行（和上面原始歌词的行长、语气一致；"
            "不要改动、不要增删任何 token，编号必须连续且全部用到，顺序不变）。\n\n"
            + "\n".join(blocks)
            + '\n\n只输出 JSON：{"lines":[[1,2,3],[4,5]]}'
        )
        data = self.client.chat_json(prompt, max_tokens=200 + 24 * len(tokens))
        if isinstance(data, dict):
            data = data.get("lines") or data.get("groups") or []
        if not isinstance(data, list):
            raise LLMError(f"unexpected line-break payload: {data!r}")

        groups: list[list[int]] = []
        seen: list[int] = []
        for item in data:
            if not isinstance(item, list) or not item:
                return None
            group: list[int] = []
            for value in item:
                try:
                    number = int(value)
                except (TypeError, ValueError):
                    return None
                if not 1 <= number <= len(tokens):
                    return None
                group.append(number - 1)
                seen.append(number - 1)
            groups.append(group)
        # Must be a partition: every token exactly once, in order.
        if seen != list(range(len(tokens))):
            return None
        return groups


    def _ask_extras(
        self,
        extras: list[Line],
        lines: list[Line],
        report: AlignReport | None = None,
        words: list[Word] | None = None,
    ) -> tuple[set[int], dict[int, int]]:
        """Ask the model which uncovered segments are real content.

        An uncovered segment is not always new content. When the recogniser
        mangles a line badly, the DP cannot place it, the line comes out as
        "unmatched" and its words look like something the lyrics do not
        contain. The model is therefore asked, for each segment, whether it is
        a lyric line it can name — by text or, because ASR text is unreliable,
        by reading.

        Only the lyric lines that sit near the segment in *time* are offered as
        candidates: a line from another verse reading alike is a coincidence,
        and the singer repeating the title in the intro is new content rather
        than a misrecognition of the chorus. With no nearby line at all (a
        spoken intro before the first line) the question is not asked.
        """
        # Line numbers are the caller's indices (1-based), blank lines included,
        # so the model's answer maps straight back onto ``report.unmatched``.
        lyric_sample = [
            f"{index + 1}. {line.text}" for index, line in enumerate(lines) if line.text.strip()
        ][:40]
        blocks: list[str] = []
        nearby: dict[int, list[int]] = {}
        for index, extra in enumerate(extras):
            context = _nearby_lines(index, extras, lines, report, words)
            nearby[index] = context
            block = (
                f"{index + 1}. [{format_seconds(extra.start)}] {extra.text}"
                f"\n   读音：{key_text(extra.text)}"
            )
            if context:
                block += "\n   时间上附近的歌词行：" + "；".join(
                    f"{number + 1}. {lines[number].text}" for number in context
                )
            else:
                block += "\n   时间上附近没有歌词行（这一段前后几十秒都没有唱到歌词）"
            blocks.append(block)
        prompt = (
            "这是一首歌的原始歌词（节选，带行号）：\n"
            + "\n".join(lyric_sample)
            + "\n\n下面是语音识别在音频里听到、但没能对上上面任何一行歌词的片段（带时间和读音）。\n"
            "原始歌词可能是录音室版本，实际是现场版：现场可能加唱、重复、即兴，"
            "也可能只是伴奏声被人声识别器误听成了字。\n"
            "对每个片段判断：\n"
            "1. 它是不是这段录音里真实的人声内容（演唱、哼唱、口播、MC 串场、报幕都算）？"
            "只要是真的有人唱或说了这些话就保留——内容不在原始歌词里并不构成丢弃的理由，"
            "现场版本来就会有开场白、介绍、加唱；同一句歌词在别处再唱一次、"
            "或把歌名念出来，也都是真实内容，一样保留。只有确实像是伴奏/噪声被误听成的字、"
            "或同一个词的无意义反复，才丢弃。\n"
            "2. 它其实是不是**时间上附近**的某一行歌词？语音识别会把字听错，所以请同时比较读音："
            "如果这个片段和附近某一行很像，或者像是附近某一行（或相邻两行）截取出来的一段，"
            "那很可能是那一行被识别错了，而不是新加的一句。只和该片段下面列出的附近歌词行比较，"
            "不要拿别处的歌词来对；没有列出附近行时，这一问留空。\n"
            "注意：问题 2 的答案只用来判断要不要重新识别这段音频，**不是**丢弃的理由——"
            "两个问题的答案互相独立。\n\n"
            + "\n".join(blocks)
            + '\n\n只输出 JSON：{"keep":[1,3],"lyric":[{"segment":2,"line":5}]}'
            "\n（keep 里列出真实人声内容的片段编号；lyric 里列出「其实是附近某一行」的片段编号和歌词行号，"
            "没有就留空数组）"
        )
        data = self.client.chat_json(prompt, max_tokens=140 + 20 * len(extras))
        if not isinstance(data, dict):
            raise LLMError(f"unexpected extras payload: {data!r}")
        raw_keep = data.get("keep") or data.get("keep_lines") or []
        if not isinstance(raw_keep, list):
            raise LLMError(f"unexpected extras payload: {data!r}")
        keep: set[int] = set()
        for item in raw_keep:
            try:
                number = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= len(extras):
                keep.add(number - 1)

        suspects: dict[int, int] = {}
        raw_lyric = data.get("lyric") or data.get("lines") or []
        for item in raw_lyric if isinstance(raw_lyric, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                segment = int(item.get("segment"))
                line = int(item.get("line"))
            except (TypeError, ValueError):
                continue
            if not 1 <= segment <= len(extras):
                continue
            if not 1 <= line <= len(lines):
                continue
            if not lines[line - 1].text.strip():
                continue
            # A line the model was never shown nearby is a hallucinated match;
            # it would trigger a pointless re-recognition of the audio.
            if nearby.get(segment - 1) is not None and line - 1 not in nearby[segment - 1]:
                continue
            suspects[segment - 1] = line - 1
        return keep, suspects

    def _ask_picks(
        self,
        batch: list[int],
        lines: list[Line],
        heard: dict[int, str],
        mapped: dict[int, LineSpan],
    ) -> dict[int, int]:
        """For each line, ask which of (own, previous, next) span is it."""
        rng = random.Random(0x5EED)
        labels = "ABCD"
        blocks: list[str] = []
        truth: dict[int, dict[str, int]] = {}
        for number, index in enumerate(batch, 1):
            candidates = [j for j in (index - 1, index, index + 1) if j in heard]
            order = candidates[:]
            rng.shuffle(order)
            blocks.append(f"句{number} 歌词读音：{key_text(lines[index].text)}")
            truth[number] = {}
            for slot, candidate in enumerate(order):
                blocks.append(f"   {labels[slot]}) {key_text(heard[candidate])}")
                truth[number][labels[slot]] = candidate

        prompt = (
            "下面每一「句」给出歌词读音，以及若干候选读音（来自语音识别的结果，有错字漏字，"
            "但发音大体接近实际演唱）。\n为每句选出最可能是该句演唱的那个候选。\n"
            "注意：歌词里可能有完全重复的行（副歌重复出现），几个候选读音一样是正常的，"
            "选到读音相同的任何一个都不算错。\n\n"
            + "\n".join(blocks)
            + '\n\n只输出 JSON：{"picks":[{"line":1,"pick":"B"}]}'
        )
        data = self.client.chat_json(prompt, max_tokens=200 + 40 * len(batch))
        if isinstance(data, dict):
            data = data.get("picks") or data.get("lines") or []
        if not isinstance(data, list):
            raise LLMError(f"unexpected verification payload: {data!r}")

        picks: dict[int, int] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            number = item.get("line")
            pick = str(item.get("pick") or "").strip().upper()[:1]
            if not isinstance(number, int) or number not in truth:
                continue
            candidate = truth[number].get(pick)
            if candidate is None:
                continue
            picks[batch[number - 1]] = candidate
        return picks


# ----------------------------------------------------------------------
# Kana DP: lyric stream -> ASR word spans
# ----------------------------------------------------------------------


def align_spans(texts: list[str], words: list[Word]) -> list[tuple[int | None, int | None, float]]:
    """Monotonic character alignment of the lyric readings to the ASR readings.

    Returns one ``(first_word, last_word, confidence)`` per lyric line.
    ``confidence`` is the share of the line's kana the DP actually matched, so
    a line the ASR never sang comes back with a low value and no span.

    The global DP is monotonic, so a line's matched kana can land in two
    different places: the title line「我的故事未写完」 was matched onto its own
    singing *and* onto the repeated title in the spoken intro, and the span
    between them (the whole intro) then belonged to that line — the intro
    vanished from the output. So the hits are grouped by time and only the
    group with the most matches becomes the line's span; the other groups stay
    uncovered and are judged like any other sung-but-unwritten content.
    """
    lyric: list[str] = []
    ranges: list[tuple[int, int]] = []
    for text in texts:
        start = len(lyric)
        lyric.extend(key_text(text))
        ranges.append((start, len(lyric)))

    asr: list[str] = []
    word_of: list[int] = []
    for word_index, word in enumerate(words):
        for ch in key_text(word.text):
            asr.append(ch)
            word_of.append(word_index)

    if not lyric or not asr:
        return [(None, None, 0.0) for _ in texts]

    matched = _global_match(lyric, asr)

    spans: list[tuple[int | None, int | None, float]] = []
    for start, end in ranges:
        hits = [matched[i] for i in range(start, end) if i in matched]
        if not hits:
            spans.append((None, None, 0.0))
            continue
        confidence = len(hits) / max(1, end - start)
        groups = _hit_groups([word_of[h] for h in hits], words)
        # Most matches wins; ties go to the earlier group, which is where a
        # line is usually sung first.
        best = max(groups, key=lambda group: (len(group), -group[0]))
        spans.append((best[0], best[-1], confidence))
    return spans


def _hit_groups(word_indices: list[int], words: list[Word]) -> list[list[int]]:
    """Split a line's matched word indices into the places it was heard.

    Words more than ``EXTEND_GAP_SEC`` apart are different utterances, not one
    phrase: a real lyric line is sung in one go, and a match 20 s away is the
    same words somewhere else in the song (or in the intro).
    """
    unique = sorted(set(word_indices))
    groups: list[list[int]] = []
    for index in unique:
        if groups:
            previous = groups[-1][-1]
            if index - previous > 1 and _gap_sec(words[previous], words[index]) > EXTEND_GAP_SEC:
                groups.append([index])
                continue
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _gap_sec(before: Word, after: Word) -> float:
    if before.end is None or after.start is None:
        return 0.0
    return max(0.0, after.start - before.end)


def _global_match(lyric: list[str], asr: list[str]) -> dict[int, int]:
    """Needleman-Wunsch style DP; returns lyric index -> ASR index for matches."""
    n, m = len(lyric), len(asr)
    prev = [GAP_SCORE * j for j in range(m + 1)]
    back: list[bytearray] = []
    for i in range(1, n + 1):
        cur = [GAP_SCORE * i] + [0.0] * m
        row = bytearray(m + 1)
        a = lyric[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (MATCH_SCORE if a == asr[j - 1] else MISMATCH_SCORE)
            up = prev[j] + GAP_SCORE
            left = cur[j - 1] + GAP_SCORE
            best, move = diag, 0
            if up > best:
                best, move = up, 1
            if left > best:
                best, move = left, 2
            cur[j] = best
            row[j] = move
        back.append(row)
        prev = cur

    matched: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i - 1][j]
        if move == 0:
            if lyric[i - 1] == asr[j - 1]:
                matched[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return matched


def heard_text(span: LineSpan, words: list[Word]) -> str:
    return "".join(word.text for word in words[span.first : span.last + 1])


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remainder:02d}"


def coverage_note(report: AlignReport, words: list[Word]) -> str | None:
    """Describe how much of the ASR transcript the lyrics actually cover.

    A concert recording often differs from the studio lyrics: an extra verse,
    an MC break, a repeated chorus. Those ASR words have no lyric line, and the
    share of them tells the user whether the mismatch is worth a listen rather
    than a re-run. This only ever produces a note; it never moves a timing.
    """
    if not words or not report.mapped:
        return None
    covered = bytearray(len(words))
    for span in report.mapped.values():
        for index in range(max(span.first, 0), min(span.last + 1, len(words))):
            covered[index] = 1
    stray = len(words) - sum(covered)
    share = stray / len(words)
    parts: list[str] = []
    if share >= 0.2:
        parts.append(
            f"识别结果中有 {share:.0%} 的内容没有对应歌词（共 {stray}/{len(words)} 个词），"
            "可能是现场版多出的段落或间奏，逐字时间以 ASR 为准"
        )
    if report.unmatched:
        parts.append(f"{len(report.unmatched)} 行歌词没有被识别到，保持无时间并已标记")
    if report.recovered:
        parts.append(
            f"{len(report.recovered)} 行靠 LRCLIB 时间轴在附近找回位置，现场版可能与之不同，建议试听"
        )
    if report.weak:
        parts.append(f"{len(report.weak)} 行被标记为可能错位，建议人工试听校正")
    return "；".join(parts) if parts else None


EXTRA_LINE_GAP = 0.9
EXTRA_LINE_CHARS = 24


def is_substantial(extra: Line) -> bool:
    """Worth deciding on at all: a phrase, not a stray syllable.

    Length in characters is the main test — a couple of characters over an
    instrument is noise. Time matters only for a short segment: speech is dense,
    so several distinct characters inside a fraction of a second ("唱给大家听",
    5 characters in 0.72 s) are still a real phrase, while a single syllable
    stretched over a second is an instrument. Shared with the pipeline's
    no-LLM fallback so the two paths keep the same rule.
    """
    if extra.start is None or extra.end is None:
        return False
    text = fold(extra.text)
    if len(text) < EXTRA_MIN_CHARS:
        return False
    return extra.end - extra.start >= EXTRA_MIN_SEC or (
        len(set(text)) >= EXTRA_KEEP_DISTINCT
        and extra.end - extra.start >= EXTRA_MIN_SEC_SHORT
    )


def _real_speech(extra: Line) -> bool:
    """Is this segment too long and too varied to be recogniser noise?

    Text this substantial can only come from someone actually speaking or
    singing: a loud instrumental produces a stray syllable or two, never a
    whole phrase. Such a segment is kept regardless of the model's verdict,
    because dropping it loses content the user cannot recover, while keeping it
    merely adds a line they can delete. A segment that repeats one or two
    syllables is the exception — that is a sustained instrument misheard.
    """
    if extra.start is None or extra.end is None:
        return False
    text = fold(extra.text)
    return (
        len(text) >= EXTRA_KEEP_CHARS
        and extra.end - extra.start >= EXTRA_KEEP_SEC
        and len(set(text)) >= EXTRA_KEEP_DISTINCT
    )


def _next_to_kept(extras: list[Line], kept: set[int]) -> set[int]:
    """Short segments that continue a kept one, across the recogniser's pauses.

    ``extra_lines`` splits uncovered tokens wherever the ASR heard a breath, so
    one spoken sentence arrives as several segments and the tail ("行", "唱给
    大家听") is often too short to survive on its own. A short segment within
    ``EXTRA_JOIN_SEC`` of a kept one is the rest of that sentence, not noise.
    """
    joined: set[int] = set()
    while True:
        added = False
        for extra in extras:
            if id(extra) in kept or id(extra) in joined:
                continue
            if extra.start is None or extra.end is None or _real_speech(extra):
                continue
            for other in extras:
                if id(other) not in kept | joined:
                    continue
                if other.start is None or other.end is None:
                    continue
                if extra.start - other.end <= EXTRA_JOIN_SEC and other.start - extra.end <= EXTRA_JOIN_SEC:
                    joined.add(id(extra))
                    added = True
                    break
        if not added:
            return joined


def _nearby_lines(
    index: int,
    extras: list[Line],
    lines: list[Line],
    report: AlignReport | None,
    words: list[Word] | None,
) -> list[int] | None:
    """Lyric lines the segment could plausibly be a misrecognition of.

    Proximity is *time*, not text: a live intro that reads like the chorus is
    not the chorus, and a line from another verse is not the sentence being
    sung here. Only lines mapped within ``CONTEXT_SEC`` of the segment are
    offered, plus the unmatched lines next to them — those have no time of
    their own, and an unmatched neighbour is exactly the "the DP lost this
    line" case the question is meant to catch.

    ``None`` means the timings are unknown (no mapping yet); the caller then
    does not restrict the answer, because it has nothing to restrict it with.
    """
    segment = extras[index]
    if segment.start is None or report is None or not words:
        return None
    starts: dict[int, float] = {}
    for line_index, span in report.mapped.items():
        if 0 <= span.first < len(words) and words[span.first].start is not None:
            starts[line_index] = words[span.first].start
    if not starts:
        return None
    near = {line_index for line_index, start in starts.items() if abs(start - segment.start) <= CONTEXT_SEC}
    if not near:
        return []
    for line_index in list(near):
        for delta in (-1, 1):
            neighbour = line_index + delta
            while 0 <= neighbour < len(lines) and not lines[neighbour].text.strip():
                neighbour += delta
            if 0 <= neighbour < len(lines) and neighbour not in report.mapped:
                near.add(neighbour)
    # Closest in time first: the line being sung here, not one a verse away.
    # The unmatched neighbours have no time of their own — they are the lines
    # this question exists to recover — so they come last and are never cut.
    timed = sorted(
        (index for index in near if index in starts),
        key=lambda index: abs(starts[index] - segment.start),
    )
    untimed = sorted(index for index in near if index not in starts)
    return sorted(timed[:CONTEXT_LINES] + untimed)


def extra_lines(words: list[Word], report: AlignReport) -> list[Line]:
    """ASR words no lyric line covers, kept as their own lines.

    On a live recording these are the improvised lines, the extra chorus and
    the ad-libs — real singing the studio lyrics do not contain. The project
    spec says to keep them, and the text is the recogniser's own, so nothing is
    invented. They are grouped by pauses exactly like the no-lyrics fallback.
    """
    if not words or not report.mapped:
        return []
    covered = bytearray(len(words))
    for span in report.mapped.values():
        for index in range(max(span.first, 0), min(span.last + 1, len(words))):
            covered[index] = 1

    stray: list[Word] = []
    out: list[Line] = []

    def flush() -> None:
        if not stray:
            return
        text = "".join(word.text for word in stray).strip()
        if text:
            timed = [
                Word(text=word.text, start=word.start, end=word.end, source="asr")
                for word in stray
            ]
            out.append(
                Line(
                    text=text,
                    start=timed[0].start,
                    end=timed[-1].end,
                    words=timed,
                    source="asr-extra",
                    note="这一行是 ASR 听到、但原始歌词里没有的内容（现场加唱、口播或即兴）",
                )
            )
        stray.clear()

    length = 0
    previous_end: float | None = None
    for position, word in enumerate(words):
        if not covered[position]:
            gap = (word.start or 0) - (previous_end if previous_end is not None else word.start or 0)
            if stray and (gap > EXTRA_LINE_GAP or length >= EXTRA_LINE_CHARS):
                flush()
                length = 0
            stray.append(word)
            length += len(word.text.strip())
        else:
            flush()
            length = 0
        previous_end = word.end
    flush()
    return out


def flag_collisions(lines: list[Line], report: AlignReport) -> None:
    """Mark neighbouring lines that ended up on the same ASR span.

    That happens when the DP cannot find one line's words and lets it borrow a
    neighbour's span. Two lines can never be sung at the same time, so both are
    flagged for a human listen instead of silently showing duplicated timings.
    """
    order = sorted(report.mapped)
    for left, right in zip(order, order[1:]):
        a, b = report.mapped[left], report.mapped[right]
        overlap = min(a.last, b.last) - max(a.first, b.first) + 1
        if overlap <= 0:
            continue
        shorter = min(a.last - a.first + 1, b.last - b.first + 1)
        if overlap / shorter < 0.5:
            # Sharing a boundary word is normal; sharing most of the span means
            # one of the two lines lost its own words.
            continue
        lines[left].source = "asr-partial"
        lines[right].source = "asr-partial"
        for index in (left, right):
            if index not in report.weak:
                report.weak.append(index)
    report.weak.sort()


# ----------------------------------------------------------------------
# Step 2: char timing inside a span
# ----------------------------------------------------------------------


def absorb_strays(
    lines: list[Line], words: list[Word], report: AlignReport, trace: Any = None
) -> list[Line]:
    """Fold short uncovered token runs back into the lyric line they belong to.

    The kana DP is monotonic, so when it loses part of a line the words it could
    not place sit in their own run between two mapped spans: line 22 (有人悻悻
    地离开) was mapped onto 「悠然」 and 「心情的离开」 became a separate
    "extra". That is backwards — the lyric text is the far more reliable source,
    and a transcript fragment touching a line is that line's words misheard, not
    a new sentence.

    A run is absorbed when it touches a lyric line (or sits between two) and is
    short enough to be a fragment (``EXTEND_MAX_WORDS``) and close enough in
    time (``EXTEND_GAP_SEC``). Between two lines it goes to the one whose
    written text it resembles most; with no resemblance to judge by, to the
    shorter line (the one that lost words). A long or detached run stays an
    extra — that is a complete new sentence, e.g. an intro or a spoken part.

    The lyric text is never modified: only the span grows, so the line's time
    comes from the ASR tokens that were really sung there.
    """
    if not words or not report.mapped:
        return lines
    covered = bytearray(len(words))
    for span in report.mapped.values():
        for position in range(max(span.first, 0), min(span.last + 1, len(words))):
            covered[position] = 1

    runs: list[tuple[int, int]] = []
    position = 0
    while position < len(words):
        if covered[position]:
            position += 1
            continue
        end = position
        while end + 1 < len(words) and not covered[end + 1]:
            end += 1
        # Two uncovered stretches either side of a long silence are different
        # utterances, not one run: the spoken intro before the first line is
        # not part of the first line's misheard words just because no mapped
        # token sits between them.
        piece = position
        for index in range(position, end + 1):
            if index > piece and _pause(words[index - 1], words[index]) > EXTEND_GAP_SEC:
                runs.append((piece, index - 1))
                piece = index
        runs.append((piece, end))
        position = end + 1

    absorbed: list[dict[str, Any]] = []
    order = sorted(report.mapped)
    for first, last in runs:
        if last - first + 1 > EXTEND_MAX_WORDS:
            continue
        left = next((index for index in order if report.mapped[index].last == first - 1), None)
        right = next((index for index in order if report.mapped[index].first == last + 1), None)
        if left is None and right is None:
            continue
        start = words[first].start
        end_time = words[last].end
        if start is None or end_time is None:
            continue
        before = words[first - 1].end if first > 0 else None
        after = words[last + 1].start if last + 1 < len(words) else None
        touching = (before is not None and start - before <= EXTEND_GAP_SEC) or (
            after is not None and after - end_time <= EXTEND_GAP_SEC
        )
        if not touching:
            continue
        heard = fold("".join(word.text for word in words[first : last + 1]))
        target = _pick_line(heard, lines, report, left, right)
        span = report.mapped[target]
        span.first = min(span.first, first)
        span.last = max(span.last, last)
        absorbed.append(
            {
                "line": target + 1,
                "text": lines[target].text,
                "absorbed": "".join(word.text for word in words[first : last + 1]),
                "window": [round(start, 3), round(end_time, 3)],
            }
        )
        logger.info(
            "folded %r into line %d (%s): the lyric is the authority, not the transcript",
            absorbed[-1]["absorbed"],
            target + 1,
            lines[target].text,
        )

    if absorbed and trace is not None:
        trace.step(
            "stray_absorb",
            {
                "note": "这些识别片段紧贴着某一歌词行，按歌词原文并入该行，用它们的逐字时间",
                "absorbed": absorbed,
            },
        )
    return lines


def _pause(before: Word, after: Word) -> float:
    """Silence between two ASR tokens (0 when either time is missing)."""
    if before.end is None or after.start is None:
        return 0.0
    return max(0.0, after.start - before.end)


def _pick_line(
    heard: str,
    lines: list[Line],
    report: AlignReport,
    left: int | None,
    right: int | None,
) -> int:
    """Which lyric line a misheard run belongs to."""
    if left is None:
        return right  # type: ignore[return-value]
    if right is None:
        return left
    left_key = fold(lines[left].text)
    right_key = fold(lines[right].text)
    scores = (
        (difflib.SequenceMatcher(None, heard, left_key, autojunk=False).ratio(), left),
        (difflib.SequenceMatcher(None, heard, right_key, autojunk=False).ratio(), right),
    )
    best, index = max(scores, key=lambda item: item[0])
    if best > 0.2:
        return index
    # No textual clue ("心情的离开" vs 有人悻悻地离开 / 有人添一笔色彩): give it
    # to the shorter line, which is the one the DP could not fill.
    left_len = report.mapped[left].last - report.mapped[left].first
    right_len = report.mapped[right].last - report.mapped[right].first
    return left if left_len <= right_len else right


def separate_spans(report: AlignReport) -> None:
    """Stop neighbouring lines from sharing a boundary word.

    The DP is monotonic but not exclusive: a line's span often ends on the word
    the next line starts with, and both then get that word's time, which shows
    up as an overlapping line start. A single shared word is simply given to the
    longer span. A large overlap means one line lost its own words — that is
    left alone, because ``flag_collisions`` marks it for a human listen.
    """
    order = sorted(report.mapped)
    for left_index, right_index in zip(order, order[1:]):
        left, right = report.mapped[left_index], report.mapped[right_index]
        overlap = min(left.last, right.last) - max(left.first, right.first) + 1
        if overlap <= 0:
            continue
        shorter = min(left.last - left.first + 1, right.last - right.first + 1)
        if overlap / shorter >= 0.5:
            continue
        left_len = left.last - left.first + 1
        right_len = right.last - right.first + 1
        if left_len >= right_len:
            right.first += overlap
        else:
            left.last -= overlap


def _covered_tokens(words: list[Word], report: AlignReport, heard: list[bool]) -> list[bool]:
    """Which ASR tokens a *heard* lyric line already accounts for."""
    covered = [False] * len(words)
    for index, span in report.mapped.items():
        if index >= len(heard) or not heard[index]:
            continue
        for position in range(max(span.first, 0), min(span.last + 1, len(words))):
            covered[position] = True
    return covered


def build_timeline(
    lines: list[Line], words: list[Word], report: AlignReport, *, extras: list[Line] | None = None
) -> Timeline:
    out: list[Line] = []
    absent = set(report.absent)
    for index, line in enumerate(lines):
        if index in absent:
            # A partial cover: this line is simply not in this recording. It is
            # left out of the timeline rather than shown untimed, so the editor
            # holds exactly what was sung.
            continue
        span = report.mapped.get(index)
        source = line.source if line.source in {"asr", "asr-partial", "asr-bad", "asr-ref"} else "asr"
        if span is None:
            out.append(
                Line(
                    text=line.text,
                    start=None,
                    end=None,
                    words=[],
                    source="unmatched",
                    note="ASR 没有识别到这一行，未给出时间",
                )
            )
            continue
        timed = _time_line(line.text, words[span.first : span.last + 1], source=source)
        timed.note = FLAG_NOTE.get(source)
        if source != "asr":
            # On a flagged line, show what the recogniser actually heard: that
            # is the evidence a human needs to decide whether the lyric or the
            # timing is off. Clean lines stay uncluttered.
            heard = heard_text(span, words).strip()
            if heard:
                timed.heard = heard
        out.append(timed)
    if extras:
        out = merge_extras(out, extras)
    else:
        clamp_overlaps(out)
    return Timeline(status="none", lines=out, notes=list(report.notes))


def merge_extras(out: list[Line], extras: list[Line]) -> list[Line]:
    """Insert sung-but-unwritten lines where they were sung, then de-overlap.

    Insert each extra before the first timed lyric line that starts after it,
    so the editor still reads in lyric order and the extras sit where they were
    actually sung. Shared with the word-sync path, which also has to place the
    live intro, the MC section and the ad-libs that the source lyrics lack.
    """
    for extra in extras:
        position = len(out)
        if extra.start is not None:
            for index, line in enumerate(out):
                if line.start is not None and line.start > extra.start:
                    position = index
                    break
        out.insert(position, extra)
    clamp_overlaps(out)
    return out


def clamp_overlaps(lines: list[Line]) -> None:
    """Keep neighbouring lines from overlapping in time.

    Two lines cannot be sung at once. A line's tail is spread to the end of its
    mapped span, and the DP can give two neighbours spans that touch; that left
    line 3 ending at 24.96 s while line 4 started at 24.51 s. Trim the earlier
    line's trailing words so the list stays monotonic.
    """
    timed = [line for line in lines if line.start is not None and line.end is not None]
    for left, right in zip(timed, timed[1:]):
        if right.start is None or left.end is None or left.end <= right.start:
            continue
        # Safety net only: spans are de-overlapped upstream, so this trims a
        # stray interpolated tail rather than rescaling real timings. Scaling
        # (rather than clipping) keeps every character ascending and non-zero.
        limit = right.start
        start = left.start or 0.0
        if limit - start <= MIN_WORD_SEC:
            left.end = limit
            continue
        span_end = max(
            (word.end for word in left.words if word.end is not None), default=left.end
        )
        if span_end > limit:
            scale = (limit - start) / (span_end - start)
            for word in left.words:
                if word.start is not None:
                    word.start = round(start + (word.start - start) * scale, 3)
                if word.end is not None:
                    word.end = round(start + (word.end - start) * scale, 3)
        left.end = limit
        _enforce_word_times(left.words)


FLAG_NOTE = {
    "asr-bad": "LLM 认为这段识别结果更像相邻的某一行，时间可能错位",
    "asr-partial": "这一行与相邻行共用或部分共用识别结果，时间可能不准",
    "asr-ref": "这一行是靠 LRCLIB 时间轴在附近找到的，现场版可能与之不同",
}


def _time_line(text: str, words: list[Word], *, source: str) -> Line:
    """Split a lyric line into timed characters using the ASR words it covers."""
    tokens = tokenize(text)
    if not tokens or not words:
        return Line(text=text, start=None, end=None, words=[], source="unmatched")

    # Flatten the ASR words into reading characters with linear times inside
    # each word, keeping the word boundary (the timing granularity we have).
    asr_chars: list[tuple[str, float, float]] = []
    for word in words:
        chars = key_text(word.text)
        if not chars:
            continue
        start = word.start if word.start is not None else 0.0
        end = word.end if word.end is not None else start
        span = max(end - start, 0.0)
        for i, ch in enumerate(chars):
            asr_chars.append((ch, start + span * i / len(chars), start + span * (i + 1) / len(chars)))
    if not asr_chars:
        return Line(text=text, start=None, end=None, words=[], source="unmatched")

    # The lyric side is compared by reading too, so kanji the ASR wrote
    # differently still line up on pronunciation.
    lyric_text = ""
    char_to_token: list[int] = []
    for token_index, token in enumerate(tokens):
        key = key_text(token)
        lyric_text += key
        char_to_token.extend([token_index] * len(key))

    matcher = difflib.SequenceMatcher(
        None, lyric_text, "".join(ch for ch, _, _ in asr_chars), autojunk=False
    )
    # First match wins: a token takes the timing of its first matched character.
    match: list[int | None] = [None] * len(tokens)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            position = block.a + k
            if position >= len(char_to_token):
                continue
            token_index = char_to_token[position]
            if match[token_index] is None:
                match[token_index] = block.b + k

    times: list[tuple[float, float, bool]] = []
    for index in match:
        if index is not None:
            _, start, end = asr_chars[index]
            times.append((start, end, True))
        else:
            times.append((0.0, 0.0, False))

    matched = sum(1 for _, _, direct in times if direct)
    if matched < MIN_MATCH_DENSITY * len(times):
        # Too few characters matched by reading for their timings to mean
        # anything: a stray match early in a line would otherwise drag the
        # whole line onto it (line 20 measured 5.6 s late that way). Spread the
        # line over the span the ASR actually sang instead; every character is
        # marked interpolated.
        span_start = asr_chars[0][1]
        span_end = asr_chars[-1][2]
        step = (span_end - span_start) / len(times)
        times = [(span_start + step * k, span_start + step * (k + 1), False) for k in range(len(times))]
    else:
        _fill_gaps(times, asr_chars[0][1], asr_chars[-1][2])

    timed_words = [
        Word(text=token, start=round(start, 3), end=round(max(end, start), 3), source="asr" if direct else "interpolated")
        for token, (start, end, direct) in zip(tokens, times)
    ]
    _enforce_word_times(timed_words)
    return Line(
        text=text,
        start=timed_words[0].start,
        end=timed_words[-1].end,
        words=timed_words,
        source=source,
    )


MIN_WORD_SEC = 0.04


def _enforce_word_times(words: list[Word]) -> None:
    """Make a line's character timings usable: ascending, non-zero, in order.

    Splitting a word's span over its characters yields zero-width characters
    whenever the ASR word itself was zero-width, and two characters taken from
    the same ASR token can share a start. A zero-width character is invisible in
    karaoke and its LRC tag can never light up, so give it a small width and
    clamp the start order. The ASR's own starts are the anchor; this only nudges
    by fractions of a second, and never reorders characters.
    """
    previous: float | None = None
    for word in words:
        if word.start is None:
            continue
        if previous is not None and word.start < previous:
            word.start = previous
        previous = word.start
    for index, word in enumerate(words):
        if word.start is None:
            continue
        following = next(
            (later.start for later in words[index + 1 :] if later.start is not None), None
        )
        end = word.end if word.end is not None else word.start
        if end < word.start + MIN_WORD_SEC:
            end = word.start + MIN_WORD_SEC
        if following is not None and following > word.start and end > following:
            end = following
        word.end = round(end, 3)


def _fill_gaps(
    times: list[tuple[float, float, bool]], span_start: float, span_end: float
) -> None:
    """Interpolate untimed slots between timed neighbours (never past the span).

    Characters the ASR never heard at the edges are spread from the edge of the
    span the line was mapped to, so a line still starts where the singing
    started even when its first syllable was misheard.
    """
    n = len(times)
    timed = [index for index, item in enumerate(times) if item[2]]
    if not timed:
        return
    first, last = timed[0], timed[-1]
    _spread(times, 0, first, span_start, times[first][0])
    _spread(times, last + 1, n, times[last][1], span_end)
    i = first
    while i < n:
        if times[i][2]:
            i += 1
            continue
        j = i
        while j < n and not times[j][2]:
            j += 1
        if i == 0 or j >= n:
            i = j
            continue
        _spread(times, i, j, times[i - 1][1], times[j][0])
        i = j


def _spread(
    times: list[tuple[float, float, bool]], start: int, stop: int, begin: float, finish: float
) -> None:
    """Divide ``[begin, finish]`` evenly over the untimed slots ``start..stop``."""
    count = stop - start
    if count <= 0:
        return
    finish = max(finish, begin)
    step = (finish - begin) / count
    for k in range(count):
        times[start + k] = (begin + step * k, begin + step * (k + 1), False)


def tokenize(text: str) -> list[str]:
    """CJK characters individually, latin runs as words.

    Every character of ``text`` lands in exactly one token (punctuation and
    spaces attach to the preceding token) so joining the tokens reproduces the
    lyric line and the karaoke display stays faithful.
    """
    tokens: list[str] = []
    for ch in text.strip():
        if is_cjk(ch):
            tokens.append(ch)
        elif ch.isalnum() and tokens and not is_cjk(tokens[-1][-1]) and tokens[-1][-1].isalnum():
            tokens[-1] += ch
        elif not ch.isalnum() and not is_cjk(ch) and tokens:
            tokens[-1] += ch
        else:
            tokens.append(ch)
    return tokens
