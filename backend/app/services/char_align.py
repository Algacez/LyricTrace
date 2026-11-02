"""Per-character timings: the recogniser's word stream, re-checked where it cannot be trusted.

Per-character times are the ASR word stream's: the recogniser's tokens are per
character, and a character it heard directly is a real measurement of when it
was sung. Its weakness is *absolute* placement — over a whole song the
recogniser drifts, and its table alone scores 0.140 s against the official 逐字
times where a forced aligner scores 0.097 s — so it is not used alone.

A forced aligner is a service built for singing: given a short clip **and the
exact text**, it places every character by matching the audio to that text. It
cannot decide which line is sung where — that is what the ASR + DP is for —
but inside a window the ASR already located, it gives the character times
directly (upstream: 9% zero-width characters on Japanese, 1.3% on Chinese).
That is what it is used for: checking the recogniser's placement, line by line.
Where the two agree on a character (within ``FUSE_TOLERANCE``) the ASR's
boundary is kept; where they disagree, or where the ASR only interpolated the
character, the aligner's time stands.

So the two are used for what each is best at:

    ASR + kana DP   ->  which line, and roughly where (the clip window)
    aligner         ->  where each character is, to check the ASR against

Where neither can place a character, the line keeps the times the word stream
gave it — word spans split across the line's characters, not a character
alignment — and the note names the lines. The line boundaries, the text and the
extras all stay exactly as the rest of the pipeline produced them; only the end
of a line is pulled back to where the singing stops, because the aligner never
holds the final vowel and the ASR's line end can reach into the silence after
it. A line timed by the lyrics source's own verified word sync is always left
alone.

The clip handed to the aligner is cut the way upstream measured it: no padding
in front, and the edges moved out to where the vocal track is actually quiet.
A leading pad drags the neighbouring line's singing into the clip and the
aligner then puts this line's first characters on it (upstream: 0.011 s median
error at pad 0, 0.583 s at pad 0.5; measured here too: pad 0 aligned 32 of 42
lines against 29 with 0.3 s of padding, and cutting in silence 33).
"""

from __future__ import annotations

import array
import base64
import difflib
from dataclasses import dataclass, field
import logging
import math
import re
import unicodedata
import wave
from pathlib import Path
from typing import Any, Callable

import httpx

from ..audio import cut_audio
from ..config import settings
from ..schemas import Line, Timeline, Word
from .providers.llm_aligner import _time_line, is_cjk, tokenize

logger = logging.getLogger("lyrictrace.char_align")

#: Audio kept before the line. Zero: upstream measured that any leading pad
#: drags the previous line's singing into the clip, and the aligner then puts
#: this line's first characters on it (median error 0.011 s at 0, 0.583 s at
#: 0.5 s). Measured here as well: 32 of 42 lines aligned at 0 against 29 at 0.3.
PAD_SEC = 0.0
#: ... and after it, when the line does not end in silence: a clip that cuts
#: the last word off aligns that word badly.
TAIL_SEC = 0.6
#: How far to look for silence when moving a clip edge onto it.
QUIET_SEARCH = 0.6
#: Envelope frame length for the silence detection (matches upstream).
HOP_SEC = 0.02
#: A frame is "quiet" below this share of the 95th-percentile level.
QUIET_SHARE = 0.05
#: A clip shorter than this has no context for the aligner.
MIN_CLIP_SEC = 0.5
#: The clip must be cut on this granularity so millisecond offsets stay exact.
MIN_WORD_SEC = 0.04
#: Characters the aligner stacked on one boundary are spread over at most this
#: much time, so a pause after them is not filled with invented singing.
SPREAD_CAP = 0.5
#: The aligner puts the last mora on the last phoneme it heard and does not hold
#: the final vowel, so its end is systematically early (measured on
#: ``prj_4a39…``: 31 of 42 lines shorter, none longer, median −0.53 s). On a
#: belt note the held vowel is one to two seconds of singing it never accounts
#: for, and the ASR's own line end can reach past the singing into silence.
#: Neither means the phrase was lost — every character before the last one is
#: placed normally — so the last one is stretched to where the vocal track
#: actually stops, instead of throwing the whole line away.
#:
#: A line the aligner only *partly* matched is a different thing: its characters
#: are squeezed into the stretch it did hear. Measured against the part of the
#: line that has singing in it, the characters must cover this much of it.
#:
#: The threshold is deliberately loose, because the "sung" length it is measured
#: against is itself only an estimate: the ASR's line end often runs into the
#: silence after the phrase (or into an over-long token), which inflates the
#: denominator and makes a good placement read as a squeezed one. Measured on
#: the six songs in ``/root/yrc`` against their official 逐字 times: of the 31
#: lines a 0.75 gate threw away, 27 were closer to the reference than the
#: word-stream spread they fell back to. At 0.5 the per-song median character
#: error is 0.098 s against 0.109 s, and Dear Mr 「F」's p90 drops from 2.14 s
#: to 0.90 s — the tail is what the old gate was costing.
MIN_SPAN_SHARE = 0.5
#: How far back to look for the last moment the vocal track is loud: the hold on
#: a final vowel, or the silence the ASR's line end reached into.
TAIL_SEARCH = 3.0
#: A quiet stretch this long separates two phrases. When the ASR's line end lands
#: in the *next* phrase, the loud run found there is not this line's singing, and
#: the run before the silence is where this line actually stopped.
PHRASE_GAP = 1.2
#: How far back the phrase-break check may look. The run at the line end is often
#: a breath or a bleed a fraction of a second long, with the real phrase break
#: several seconds before it — further than ``TAIL_SEARCH`` reaches. Measured on
#: 残機 (55 lines): 8 s leaves the mean at 0.382 s, 30 s brings it to 0.299 s,
#: and 60 s is identical, so 30 s is past the point where it stops mattering.
PHRASE_SCAN = 30.0
#: One 20 ms frame of breath, a chair creak or cymbal bleed is not singing, so
#: the level that decides where a line's tail ends is smoothed over this long.
TAIL_SMOOTH = 0.12
#: Below this much covered audio the line has no singing to align to at all —
#: an ASR hallucination over an instrumental intro, not a line that failed.
NO_SINGING_SEC = 0.6
#: How far back a *collapsed* window may grow. The ASR sometimes gives a line
#: almost no span at all (start equals end, and both equal the next line's
#: start), which leaves a clip too short to align, so the line falls back to the
#: word stream — measured as the worst path: 0.33 s median character error
#: against 0.05 s for aligned lines. The phrase is still there, entirely before
#: that point, so the window grows back to where the loud run starts. Measured
#: on the six songs against the official 逐字 times: 3 s is best of 3/6/12 s
#: (per-song median of medians 0.097 → 0.093 s, あいつら p90 1.32 → 1.30 s,
#: 残機 p90 0.78 → 0.69 s and its line p90 1.61 → 1.20 s); 6 s and 12 s reach
#: past the previous phrase and score worse.
COLLAPSED_REACH = 3.0
#: Growing to the loud run's start is still a guess at where the phrase begins:
#: the run can be ten seconds of continuous singing, and then its start is some
#: earlier phrase. For a collapsed line the aligner itself can say which start
#: fits — it scores every candidate. Sweeping the start back in these steps and
#: keeping the best-scoring window measured 20.02 s → 14.79 s of character error
#: over the 19 collapsed lines in the six songs (6 wins, 4 losses, 9 unchanged),
#: and a 3 s reach beats 2/4/6 s. Among windows that score within ``SCAN_TIE``
#: of the best the *earliest* start wins: a phrase is sung before the ASR admits
#: it, not after, and the score is flat across the whole phrase (我的故事 L11
#: scores 1.00 from 80.32 s to 83.92 s; the official times start at 80.12 s).
COLLAPSED_SCAN_REACH = 3.0
COLLAPSED_SCAN_STEP = 0.5
#: Two windows this close in score are the same phrase placed a step apart.
SCAN_TIE = 0.05
#: A line the aligner could not place at all is usually one the ASR put several
#: seconds late, past the ±1.5 s the shifted search tries (Dear Mr 「F」 L35 sits
#: 3.6 s late and scores 0.84 with 0.68 coverage when its clip starts where the
#: phrase does). Sweep the start back further, but only accept a window that
#: both places most characters directly and covers the phrase: without that
#: pair of gates the sweep moves 秒針 L49 to a window that scores 0.61 and is
#: worse than the word stream it replaced. With them, 16.28 s → 7.31 s over the
#: five unplaced lines, no losses.
KEPT_SCAN_REACH = 6.0
KEPT_SCAN_STEP = 0.5
KEPT_SCAN_SHARE = 0.7
KEPT_SCAN_COVERAGE = 0.4
#: A line rejected as *squeezed* did place its characters, but only over the
#: head of the phrase: the clip stops before the singing does, so the answer
#: covers half of what is heard and the line falls back to the word stream.
#: The shifted search moves the window ±1.5 s and demands a higher score, which
#: a squeezed answer never gives — its coverage is capped by the clip, not by
#: where the clip starts. So move the whole window further, and take the best
#: window that passes the same gates as everything else.
SQUEEZE_SCAN_REACH = 6.0
SQUEEZE_SCAN_STEP = 0.5
#: The ASR + DP decides where a line is *first*; when its answer does not hold up
#: (the text does not match, most characters came back interpolated, the phrase
#: is squeezed into part of the clip, or there is a long hole inside it), the
#: same line is re-aligned in windows shifted by these offsets and the best
#: result wins. Neighbouring lines bound the shifts, so two lines can never land
#: on the same stretch.
SEARCH_OFFSETS = (-0.3, 0.3, -0.6, 0.6, -0.9, 0.9, -1.2, 1.2, -1.5, 1.5)
#: A shifted window must beat the ASR's own placement by this much. One
#: character of a ten-character line is worth about 0.04–0.07 of score, so a
#: smaller gain is a single character placed differently — noise, not evidence
#: that the line was on the wrong stretch. Measured on 黑猫 (43 lines): at 0.05
#: the search moved six lines whose prior placement was already right (line 1:
#: prior 29.92 s against the lyrics source's own 29.90 s, moved to 30.82 s);
#: at 0.12 it moves none of them and every line keeps the ASR's placement.
SEARCH_MARGIN = 0.12
#: Share of characters the aligner placed directly (not interpolated) below
#: which its answer for the window is treated as a guess.
SEARCH_SHARE = 0.6
#: Longest silence inside a line that still looks like a phrase (a hole means
#: the aligner matched something else).
SEARCH_GAP = 1.2
#: Lines whose characters were timed by the lyrics source (already checked
#: against this audio, see ``word_sync``) are better than the aligner's guess.
KEEP_SOURCES = ("word-sync", "word-sync-shifted", "word-sync-missing")
#: Per clip: the service is single-process and serial, and a stuck request must
#: not hold the whole alignment hostage.
REQUEST_TIMEOUT = 120.0
#: A whole-song pass sends the full vocal track and every lyric line at once
#: (a four-minute song is ~12 MB of 16 kHz WAV and takes ~20 s on CPU).
SONG_TIMEOUT = 300.0
#: Second opinion, and now the source of record: the ASR aligner's own
#: per-character table for the same song (``char_fuse``) carries each
#: character's time. A recogniser's *absolute* placement is the weak part — the
#: ASR table alone scores 0.140 s against the official 逐字 times where a forced
#: aligner scores 0.097 — but where the two agree on a character, the ASR is
#: demonstrably on the right stretch, and its own boundary is the one to keep. So:
#:
#:   * both timed the character and they agree within ``FUSE_TOLERANCE``
#:     -> the ASR's start is the character's time;
#:   * they disagree by more than that -> the aligner keeps it (the ASR is on the
#:     wrong stretch, and there is nothing to average);
#:   * the ASR only interpolated the character (it did not hear it) -> the
#:     aligner keeps it, because an interpolated time is not a measurement at all.
#:
#: Measured on the six songs in ``/root/yrc`` against the official 逐字 times:
#: 0.086 s (aligner with a 0.2 s average) -> 0.085 s median of per-song
#: medians. Adopting the ASR beyond 0.3 s loses that: 0.5 s gives 0.093 s,
#: 1 s 0.113 s, ASR everywhere 0.128 s (aligner alone 0.097 s).
FUSE_TOLERANCE = 0.3

#: Kana in the lyric text, which means the line is Japanese whatever the
#: project was labelled with. A wrong language does not make the aligner place
#: the text badly, it makes it refuse: measured on 土星 (sung in Japanese,
#: labelled Chinese from its title), 17/28 lines placed under zh and 25/28
#: under ja, with 57 characters crammed below 20 ms against 23. The label
#: itself is left alone — it also picks the ASR's language, and changing what a
#: run was labelled with is not this pass's call.
KANA_RE = re.compile(r"[぀-ヿ]")

#: Our language names -> the aligner's own language codes.
LANG_CODES = {
    "chinese": "zh",
    "cantonese": "zh",
    "japanese": "ja",
    "english": "en",
}


class CharAlignError(RuntimeError):
    """The aligner answered with an error (bad request, no output)."""


class CharAlignUnavailable(RuntimeError):
    """The aligner cannot be reached at all; stop asking for this run."""


def language_code(language: str | None) -> str | None:
    """The aligner's code for our language name, or ``None`` when unsupported."""
    return LANG_CODES.get((language or "").strip().lower())


def available() -> bool:
    return bool(settings.char_aligner_url)


def per_character_available(use_hfa: bool | None = None) -> bool:
    """Whether per-character times can be produced at all, under this setting.

    With the aligner taking part (``use_hfa``, default ``settings.char_hfa``) that
    needs the aligner service. Without it, the recogniser's own table is the
    only source, so the ASR service is what has to be reachable.
    """
    if use_hfa is None:
        use_hfa = settings.char_hfa
    if use_hfa:
        return available()
    from . import char_fuse

    return char_fuse.available(honour_switch=False)


def unavailable_detail() -> str:
    return (
        "Set LYRICTRACE_CHAR_ALIGNER_URL to the forced-aligner service base URL "
        "(e.g. http://aligner.local:8000) to get per-character timings from it."
    )


def _post(payload: dict[str, Any]) -> dict[str, Any]:
    """One align request; raises the two error kinds above."""
    return _request("/align", payload)


def _request(
    path: str, payload: dict[str, Any], *, timeout: float | None = None
) -> dict[str, Any]:
    """One request to the aligner service; raises the two error kinds above."""
    try:
        with httpx.Client(
            base_url=settings.char_aligner_url,
            timeout=httpx.Timeout(timeout or REQUEST_TIMEOUT, connect=10.0),
        ) as client:
            response = client.post(path, json=payload)
    except httpx.HTTPError as exc:
        raise CharAlignUnavailable(f"逐字对齐服务不可用：{exc}") from exc
    if response.status_code >= 500:
        raise CharAlignError(
            f"逐字对齐服务报错：{response.text.strip()[:200] or response.status_code}"
        )
    if response.status_code >= 400:
        raise CharAlignError(f"逐字对齐请求被拒绝：{response.text.strip()[:200]}")
    return response.json()


def align_text(
    audio_path: Path,
    text: str,
    language: str | None,
    *,
    offset: float = 0.0,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Align a stretch of audio against lyric text in one request.

    This is the whole-song entry point: the caller hands over the vocal track
    and the lyrics exactly as they are, and gets per-character times back. No
    ASR, no reference timeline — the aligner is the only source of timing.

    ``text`` may hold several lines (one lyric line per ``\\n``); every
    returned character carries the ``line`` index it belongs to.
    """
    code = language_code(language)
    if code is None:
        raise CharAlignError(
            f"逐字对齐服务只支持 中文 / 日文 / 英文，本次演唱语言是 {language or '未指定'}"
        )
    payload = {
        "audio_b64": base64.b64encode(Path(audio_path).read_bytes()).decode(),
        "text": text,
        "lang": code,
        "offset": round(offset, 3),
    }
    return _request("/align", payload, timeout=timeout or SONG_TIMEOUT)


def request_service(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call another endpoint of the same service (``/transcribe``), same errors."""
    return _request(path, payload)


def _brief(text: str | None, limit: int = 12) -> str:
    """A short label for a lyric line, so a note still maps to a line after edits."""
    stripped = " ".join((text or "").split())
    return stripped if len(stripped) <= limit else stripped[:limit] + "…"


def _is_spoken(char: str) -> bool:
    """Whether a character can carry a time of its own.

    Spaces and punctuation cannot: the aligner echoes them without times, or
    not at all.
    """
    return bool(char) and not char.isspace() and not unicodedata.category(char).startswith("P")


#: Share of a line's characters the aligner must echo back before its answer
#: counts as this line's answer. A forced aligner legitimately drops a character
#: it cannot fit, so demanding a perfect echo would throw away good lines; the
#: point of the check is to reject an answer about *other* text. A clip holding
#: other singing too can make it align to the wrong stretch.
MIN_MATCHED_SHARE = 0.5


def _settle(words: list[Word], span_start: float, span_end: float) -> None:
    """Make a line's character times usable: ascending, non-zero, in order.

    The aligner marks attached kana (ゃゅょ, ー) and interpolated characters
    zero-width, and a clip whose first character was clipped returns ``null``.
    A zero-width character can never light up in karaoke, so give it a minimal
    width and clamp the order — never reordering, and never past the next
    character's start.
    """
    previous: float | None = None
    for word in words:
        if word.start is None:
            word.start = previous if previous is not None else span_start
        if previous is not None and word.start < previous:
            word.start = previous
        previous = word.start
    for index, word in enumerate(words):
        following = words[index + 1].start if index + 1 < len(words) else span_end
        end = word.end if word.end is not None else word.start
        if end < word.start + MIN_WORD_SEC:
            end = word.start + MIN_WORD_SEC
        if following is not None and following > word.start and end > following:
            end = following
        word.end = round(end, 3)


def _clamp(words: list[Word], lower: float | None, upper: float | None) -> None:
    """Keep a line inside its neighbours: two lines cannot be sung at once.

    The aligner aligns the clip it was given, and the clip is padded a little,
    so its first character can land before the previous line ends. Clip that
    rather than let the timeline overlap — the neighbour's timing is finer only
    by a fraction of a second, not by a phrase.
    """
    previous: float | None = None
    for index, word in enumerate(words):
        start = word.start if word.start is not None else previous
        if start is None:
            continue
        if lower is not None and start < lower:
            start = lower
        if previous is not None and start < previous:
            start = previous
        end = word.end if word.end is not None else start
        if upper is not None and end > upper:
            end = upper
        if end < start + MIN_WORD_SEC:
            end = start + MIN_WORD_SEC
        following = words[index + 1].start if index + 1 < len(words) else None
        if following is not None and following > start and end > following:
            end = following
        if upper is not None and end > upper:
            end = upper
        word.start, word.end = round(start, 3), round(end, 3)
        previous = word.start


def _spread(words: list[Word], end: float) -> None:
    """Separate characters the aligner stacked on one instant.

    A long vowel, a melisma or a word the aligner could not split makes it put
    several characters on the same boundary, and then several of them are
    zero-width: they can never light up. Their order is right, their spacing is
    not — so spread the group evenly over the time it has, capped so a pause
    after it is not filled with invented singing.
    """
    starts = [word.start if word.start is not None else 0.0 for word in words]
    index = 0
    while index < len(starts):
        last = index
        while last + 1 < len(starts) and starts[last + 1] - starts[index] < MIN_WORD_SEC:
            last += 1
        if last > index:
            following = starts[last + 1] if last + 1 < len(starts) else end
            span = min(following - starts[index], SPREAD_CAP)
            if span > 0:
                step = span / (last - index + 1)
                for offset in range(index + 1, last + 1):
                    starts[offset] = starts[index] + (offset - index) * step
        index = last + 1
    for word, start in zip(words, starts):
        word.start = round(start, 3)


def _per_character(words: list[Word]) -> list[Word]:
    """One word per character: a CJK token never spans several characters.

    The lyric's own tokenization is per character, so an aligned line already
    comes back that way. Two other paths do not: an extra line keeps the
    recogniser's tokens as it heard them (``この`` is one token), and a kept
    line keeps the word stream's split. A token covering several characters is
    one highlight block for all of them — the karaoke lights up two characters
    at once — so its span is divided between them in order, the same way the
    official 逐字 timings divide a word. Latin runs stay one token: that is what
    the lyrics' own 逐字 data does with them too.
    """
    out: list[Word] = []
    for word in words:
        pieces = _split_token(word)
        out.extend(pieces)
    return out


def _split_token(word: Word) -> list[Word]:
    text = word.text
    if sum(1 for char in text if is_cjk(char)) < 2:
        return [word]
    chunks: list[str] = []
    for char in text:
        if is_cjk(char) or not chunks:
            chunks.append(char)
        elif is_cjk(chunks[-1][-1]) and not char.isalnum() and not char.isspace():
            # Punctuation hangs on the character before it, as in ``tokenize``.
            chunks[-1] += char
        else:
            chunks.append(char)
    if len(chunks) < 2:
        return [word]
    start = word.start if word.start is not None else 0.0
    end = word.end if word.end is not None and word.end > start else start
    step = (end - start) / len(chunks)
    return [
        Word(
            text=chunk,
            start=round(start + step * index, 3),
            end=round(start + step * (index + 1), 3),
            source=word.source,
        )
        for index, chunk in enumerate(chunks)
    ]


def _fill(words: list[Word], end: float) -> None:
    """Give every character a width of its own: to the next one, the last to ``end``.

    The aligner places *boundaries*: its ``end`` is where the last phoneme of a
    character stopped, not where the next one begins, and it does not hold the
    final vowel. Left as-is the timeline has dead air between characters (a
    karaoke highlight that blinks off mid-line) and a line that stops early.
    A character is sung until the next one starts, so that is its width.

    Two characters closer together than ``MIN_WORD_SEC`` are one highlight
    frame, not two: whichever of them comes first never lights up, and the
    second one lights up for both. The later start is therefore pushed forward
    until it has ``MIN_WORD_SEC`` of its own — a forward push only, so the
    order and the earlier characters are untouched — and the line end follows
    the last character so it can be seen too. Measured on the six songs in
    ``/root/yrc``: characters narrower than a frame fall from 76 to 0 of 3 884
    with no change in accuracy (0.085 s either way).
    """
    for index in range(1, len(words)):
        previous, word = words[index - 1], words[index]
        if word.start is None or previous.start is None:
            continue
        if word.start - previous.start < MIN_WORD_SEC:
            word.start = round(previous.start + MIN_WORD_SEC, 3)
    if words and words[-1].start is not None:
        end = max(end, round(words[-1].start + MIN_WORD_SEC, 3))
    for index, word in enumerate(words):
        following = words[index + 1].start if index + 1 < len(words) else end
        # A character ends where the next one starts, even when the gap is tiny:
        # stretching it to MIN_WORD_SEC would overlap the next character, and an
        # overlapping highlight is worse than a narrow one.
        if following is None or following <= word.start:
            following = word.start + MIN_WORD_SEC
        word.end = round(following, 3)


@dataclass
class _Attempt:
    """One window tried for a line: what the aligner answered and how good it is."""

    lo: float
    hi: float
    offset: float = 0.0
    words: list[Word] | None = None
    chars: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None
    share: float = 0.0
    coverage: float = 0.0
    score: float = 0.0
    #: This window came from walking the clip start back, not from the ASR's
    #: own placement: its start may lie before the previous line ends.
    scanned: bool = False


@dataclass
class _Placed:
    """One attempt judged against the placement gates.

    ``words`` is the answer to use; when it is ``None``, ``reason`` says why
    not and ``nothing`` marks the case where the clip holds no singing this
    text fits (an extra line, or a hallucination) rather than a squeezed match.
    """

    words: list[Word] | None = None
    reason: str | None = None
    nothing: bool = False
    #: The characters were placed, but over less than half of what the clip
    #: holds: the clip stops before the singing does.
    squeezed: bool = False
    heard: float = 0.0
    sung_end: float = 0.0
    last_end: float = 0.0


def _align_window(
    audio_path: Path,
    work_dir: Path,
    lo: float,
    hi: float,
    text: str,
    code: str,
    fallback: tuple[float, float],
    span: float,
    *,
    offset: float = 0.0,
) -> _Attempt:
    """Cut one clip, align it, and fold the answer back into lyric tokens.

    ``score`` is how many characters the aligner placed *directly* and how much
    of the line's own length it covered — the two things a wrong window destroys:
    it forces the model to interpolate characters and to squeeze the phrase.
    The length is the line's, not the clip's, so a smaller clip cannot score
    higher just by being smaller.
    """
    attempt = _Attempt(lo=lo, hi=hi, offset=offset)
    clip = work_dir / f"fa-{int(lo * 1000):08d}.wav"
    try:
        cut_audio(audio_path, clip, lo, hi - lo)
        payload = {
            "audio_b64": base64.b64encode(clip.read_bytes()).decode(),
            "text": text,
            "lang": code,
            "offset": round(lo, 3),
        }
        answer = _post(payload)
    except CharAlignUnavailable:
        # One unreachable service is the whole run's problem, not a line's.
        raise
    except (CharAlignError, OSError) as exc:
        logger.warning("char alignment failed for %s: %s", f"{lo:.2f}-{hi:.2f}s", exc)
        attempt.reason = str(exc)
        return attempt
    finally:
        clip.unlink(missing_ok=True)

    attempt.chars = answer.get("chars") or []
    attempt.words = words_from_chars(text, attempt.chars, fallback=fallback)
    timed = [char for char in attempt.chars if char.get("start") is not None]
    direct = sum(1 for char in timed if not (char.get("interpolated") or char.get("attach")))
    attempt.share = direct / len(timed) if timed else 0.0
    if attempt.words is None:
        attempt.reason = "对齐结果与这句歌词不一致，这一行只有识别词流摊开的估计时间"
        return attempt
    first = attempt.words[0].start or lo
    last = attempt.words[-1].end or first
    attempt.coverage = min(1.0, (last - first) / span) if span > 0 else 0.0
    attempt.score = round(0.65 * attempt.share + 0.35 * attempt.coverage, 3)
    return attempt


def _needs_search(attempt: _Attempt) -> bool:
    """Whether the ASR's window is doubtful enough to look around for the line."""
    words = attempt.words
    if words is None:
        return True
    if attempt.share < SEARCH_SHARE or attempt.coverage < MIN_SPAN_SHARE:
        return True
    gap = 0.0
    for current, following in zip(words, words[1:]):
        if current.end is not None and following.start is not None:
            gap = max(gap, following.start - current.end)
    return gap > SEARCH_GAP


def _judge(
    attempt: _Attempt,
    *,
    ref_start: float,
    ref_end: float,
    quiet: "QuietTrack | None",
) -> _Placed:
    """Decide whether an alignment answer is usable, and how far the line sings.

    ``ref_start``/``ref_end`` are the stretch the answer is judged against: the
    ASR's own span for the line, shifted by however far the window was moved.
    A line whose answer is rejected keeps the word stream's estimate, so the
    gate has to be sure — see ``MIN_SPAN_SHARE`` for why it is this loose.
    """
    words = attempt.words
    if words is None:
        return _Placed(
            reason=attempt.reason or "对齐结果与这句歌词不一致，这一行只有识别词流摊开的估计时间"
        )
    ref_span = ref_end - ref_start
    # ``_settle`` guarantees both ends; the aligner never holds the final
    # vowel and the ASR's line end can reach past the singing into silence,
    # so the tail is measured on the vocal track, not on the line's own end.
    last_end = words[-1].end if words[-1].end is not None else ref_end
    first_start = words[0].start if words[0].start is not None else ref_start
    sung_end = ref_end
    if quiet:
        # ``floor`` is this line's own start: a loud run that reaches back
        # past it is this line's singing, not the next line's.
        sung_end = max(last_end, min(quiet.loud_before(ref_end, floor=ref_start), ref_end))
    sung = sung_end - ref_start
    heard = last_end - first_start
    if heard < NO_SINGING_SEC and sung > NO_SINGING_SEC:
        # Nothing was placed: the clip holds no singing this text fits.
        return _Placed(
            reason=(
                f"这句几乎对不上人声（只对上 {heard:.2f}s），可能不是真的唱了这一句，"
                "逐字时间只能用识别词流摊开估计"
            ),
            nothing=True,
            heard=heard,
            sung_end=sung_end,
            last_end=last_end,
        )
    if ref_span > 0 and heard < MIN_SPAN_SHARE * sung:
        # The characters were squeezed into the stretch the aligner heard: it
        # matched the first half of the phrase and stopped there.
        return _Placed(
            reason=(
                f"对齐只覆盖到 {heard:.2f}s（这句唱了 {sung:.2f}s），"
                "疑似只对上了前半句，这一行只有识别词流摊开的估计时间"
            ),
            squeezed=True,
            heard=heard,
            sung_end=sung_end,
            last_end=last_end,
        )
    return _Placed(words=words, heard=heard, sung_end=sung_end, last_end=last_end)


def _scan_back(
    audio_path: Path,
    work_dir: Path,
    *,
    start: float,
    hi: float,
    floor: float,
    step: float,
    text: str,
    code: str,
    fallback: tuple[float, float],
    span: float,
) -> list[_Attempt]:
    """Re-align the same clip with its start walked back in ``step`` increments.

    A window that starts too late loses the phrase's head — the aligner then
    matches what it can and either squeezes the line or refuses it. The same
    clip started earlier is the same request with more context, so it is the
    cheapest evidence available about where the line really begins.
    """
    attempts: list[_Attempt] = []
    point = start
    while point >= floor - 1e-9:
        attempt = _align_window(
            audio_path, work_dir, point, hi, text, code, fallback, span, offset=point - start
        )
        attempt.scanned = True
        attempts.append(attempt)
        point -= step
    return attempts


def _scan_shift(
    audio_path: Path,
    work_dir: Path,
    *,
    lo: float,
    hi: float,
    reach: float,
    step: float,
    lower: float | None,
    upper: float | None,
    text: str,
    code: str,
    fallback: tuple[float, float],
    span: float,
) -> list[_Attempt]:
    """Re-align the line with the whole window moved, keeping its length.

    The shifted search tries the same thing at ±1.5 s; a squeezed line needs
    further out, because the clip itself is too short to hold the phrase — what
    it needs is not a different start inside the clip but a clip around the
    singing. Neighbouring lines still bound the move, so two lines cannot land
    on the same stretch.
    """
    attempts: list[_Attempt] = []
    for index in range(1, int(reach / step) + 1):
        for sign in (1, -1):
            shift = sign * index * step
            lo_try = max(lo + shift, lower if lower is not None else 0.0)
            hi_try = min(hi + shift, upper if upper is not None else float("inf"))
            # A move the neighbours cut short is a *smaller* clip, not the same
            # phrase moved: that is a different question (and one the start
            # scan answers), and a sliver ending at the next line's start can
            # match its phrase perfectly and still be the wrong line — Dear
            # Mr 「F」 L35 scores 0.99 in the 2.9 s left before L36 starts, with
            # its answer 6 s off. Allowing any shrink measured worse on the six
            # songs (1186.0 s → 1250.3 s at a 0.7 share, 1259.3 s at 0.5).
            if hi_try - lo_try < hi - lo - 1e-3:
                continue
            if abs(lo_try - lo) < 1e-3:
                continue
            attempts.append(
                _align_window(
                    audio_path, work_dir, lo_try, hi_try, text, code, fallback,
                    span, offset=lo_try - lo,
                )
            )
    return attempts


def words_from_chars(
    text: str, chars: list[dict[str, Any]], *, fallback: tuple[float, float] | None = None
) -> list[Word] | None:
    """Turn the aligner's ``chars`` into one word per lyric token.

    ``chars`` echoes our text character by character (punctuation and spaces
    included, with ``null`` times). The lyric's own tokenization is the display
    unit — CJK characters one by one, latin runs as one word — so the character
    times are folded back into those tokens: a word starts when its first
    character starts and ends when its last one ends.

    ``fallback`` is the line's span as the ASR placed it, used only for a
    leading character the aligner left untimed (the clip's first syllable is
    sometimes cut) so it still starts where the line starts.

    Returns ``None`` when the returned characters are not the line we sent: an
    alignment of *other* text says nothing about this line's timings.
    """
    # The aligner echoes only the characters it can put a sound on: spaces and
    # punctuation come back missing, or marked ``punct`` with null times. It also
    # drops the occasional character it cannot fit (measured: 8 of 9 characters
    # echoed for one line, 1 of 2 for a repeated two-character line). Comparing
    # raw lengths rejected 18 of 48 lines on a live song, and every one of them
    # came back with no times at all. So: match the echoed characters against
    # the line by sequence, and let a dropped character cost only itself.
    sent = (text or "").strip()
    tokens = tokenize(sent)
    if "".join(tokens) != sent:
        return None
    token_chars = [(index, char) for index, token in enumerate(tokens)
                   for char in token if _is_spoken(char)]
    sent_text = "".join(char for _index, char in token_chars)
    spoken = [char for char in chars if _is_spoken(char.get("c") or "")]
    spoken_text = "".join(char.get("c") or "" for char in spoken)
    if not sent_text or not spoken_text:
        return None
    matcher = difflib.SequenceMatcher(None, sent_text, spoken_text, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    if matched < MIN_MATCHED_SHARE * len(sent_text):
        return None
    mapped: dict[int, dict[str, Any]] = {}
    for a, b, size in matcher.get_matching_blocks():
        for offset in range(size):
            mapped[a + offset] = spoken[b + offset]

    words: list[Word] = []
    position = 0
    for token in tokens:
        need = sum(1 for char in token if _is_spoken(char))
        piece = [mapped[index] for index in range(position, position + need) if index in mapped]
        position += need
        timed = [char for char in piece if char.get("start") is not None]
        if not timed:
            # Pure punctuation at the line's edge: it has no sound of its own,
            # so it hangs on the neighbouring character.
            anchor = words[-1].end if words else None
            words.append(Word(text=token, start=anchor, end=anchor, source="interpolated"))
            continue
        start = timed[0]["start"]
        end = timed[-1]["end"]
        direct = any(not (char.get("interpolated") or char.get("attach")) for char in timed)
        words.append(
            Word(
                text=token,
                start=round(float(start), 3),
                end=round(float(end if end is not None else start), 3),
                # ``aligner``: the forced aligner placed this character itself. Not
                # ``asr`` — that is the word stream's estimate, a different and
                # much coarser thing, and the two must stay distinguishable.
                source="aligner" if direct else "interpolated",
            )
        )
    if not words:
        return None
    # A leading character the aligner left untimed hangs on the first timed one.
    # Falling back to 0.0 here instead used to drag the whole line to the start
    # of the song (measured: nine lines of a live song came back at 0.00 s).
    first = next((word.start for word in words if word.start is not None), None)
    last = next((word.end for word in reversed(words) if word.end is not None), None)
    span_start = fallback[0] if fallback else (first if first is not None else 0.0)
    span_end = fallback[1] if fallback else (last if last is not None else span_start + MIN_WORD_SEC)
    _settle(words, span_start, span_end)
    return words


class QuietTrack:
    """Where the vocal track is silent, frame by frame.

    Upstream's lesson: a clip edge that falls in the middle of singing makes
    the aligner pull the clip's first characters onto whatever it hears first
    (up to 6.8 s early), so edges belong in silence. A 20 ms RMS envelope of the
    vocal track answers that in one pass — 0.27 s for a four-minute song in
    plain Python, which is cheaper than the request it saves.
    """

    def __init__(self, path: Path) -> None:
        self.hop = HOP_SEC
        self.levels: list[float] = []
        try:
            with wave.open(str(path)) as handle:
                rate = handle.getframerate()
                width = handle.getsampwidth()
                channels = handle.getnchannels()
                frames = array.array("h")
                frames.frombytes(handle.readframes(handle.getnframes()))
        except (OSError, EOFError, wave.Error) as exc:
            logger.info("no envelope for %s (%s); clips keep their line bounds", path, exc)
            return
        if width != 2 or not frames or rate <= 0:
            # Only the 16-bit WAVs this app writes are measured; anything else
            # simply leaves the windows where the line bounds put them.
            return
        step = max(1, int(HOP_SEC * rate)) * channels
        self.hop = step / channels / rate
        mono = frames[0::channels] if channels > 1 else frames
        total = len(mono) // step
        squares = [float(value) * value for value in mono]
        levels = [
            math.sqrt(sum(squares[i * step : (i + 1) * step]) / step)
            for i in range(total)
        ]
        self.levels = levels
        self.threshold = max(_percentile(levels, 0.95) * QUIET_SHARE, 1e-4)
        # A short blip must not pass for singing: the tail is measured on a
        # moving average, which a 20 ms breath cannot lift over the threshold.
        width = max(1, int(TAIL_SMOOTH / self.hop))
        running = 0.0
        self.smoothed: list[float] = []
        for index, level in enumerate(levels):
            running += level
            if index >= width:
                running -= levels[index - width]
            self.smoothed.append(running / min(index + 1, width))

    def __bool__(self) -> bool:
        return bool(self.levels)

    def is_quiet(self, moment: float) -> bool:
        index = int(moment / self.hop)
        if not self.levels:
            return False
        return self.levels[min(max(index, 0), len(self.levels) - 1)] <= self.threshold

    def before(self, moment: float, limit: float = QUIET_SEARCH) -> float:
        """The latest quiet frame at or before ``moment`` (within ``limit``)."""
        if not self.levels:
            return moment
        index = min(int(moment / self.hop), len(self.levels) - 1)
        stop = max(0, index - int(limit / self.hop))
        for k in range(index, stop - 1, -1):
            if self.levels[k] <= self.threshold:
                return k * self.hop
        return moment

    def after(self, moment: float, limit: float = QUIET_SEARCH) -> float:
        """The earliest quiet frame at or after ``moment`` (within ``limit``)."""
        if not self.levels:
            return moment
        index = max(int(moment / self.hop), 0)
        stop = min(len(self.levels) - 1, index + int(limit / self.hop))
        for k in range(index, stop + 1):
            if self.levels[k] <= self.threshold:
                return k * self.hop
        return moment

    def loud_end(self, moment: float, limit: float) -> float:
        """Where the loud run that starts at or after ``moment`` stops.

        Used to cut a clip that holds one phrase and then a long silence: the
        ASR sometimes gives a single token fifteen seconds, the line is sung at
        the head of it, and the aligner puts the line's first characters there
        and its last ones at the far edge — the line is then accepted, with its
        characters spread over fifteen seconds of which two are singing.

        Walks to the first loud frame at or after ``moment``, then to the end of
        that run. Returns ``limit`` when there is no loud run before it.
        """
        if not self.smoothed:
            return limit
        index = min(max(int(moment / self.hop), 0), len(self.smoothed) - 1)
        stop = min(len(self.smoothed) - 1, int(limit / self.hop))
        while index <= stop and self.smoothed[index] <= self.threshold:
            index += 1
        if index > stop:
            return limit
        while index + 1 < len(self.smoothed) and self.smoothed[index + 1] > self.threshold:
            index += 1
        return min(limit, (index + 1) * self.hop)

    def silence_after(self, moment: float, limit: float) -> float:
        """How long the vocal stays quiet from ``moment`` on (capped at ``limit``)."""
        if not self.smoothed:
            return 0.0
        index = min(max(int(moment / self.hop), 0), len(self.smoothed) - 1)
        stop = min(len(self.smoothed) - 1, int((moment + limit) / self.hop))
        start = index
        while index <= stop and self.smoothed[index] <= self.threshold:
            index += 1
        return min(limit, (index - start) * self.hop)

    def run_start(self, moment: float, reach: float) -> float:
        """Where the loud run containing ``moment`` begins (at most ``reach`` back).

        ``moment`` is expected to be inside singing; a quiet frame returns
        ``moment`` itself. Used to grow a window the ASR collapsed: the line is
        all *before* the point the ASR left it at, and this is where the phrase
        it belongs to starts.
        """
        if not self.smoothed:
            return moment
        index = min(max(int(moment / self.hop), 0), len(self.smoothed) - 1)
        if self.smoothed[index] <= self.threshold:
            return moment
        stop = max(0, index - int(reach / self.hop))
        while index > stop and self.smoothed[index - 1] > self.threshold:
            index -= 1
        return index * self.hop

    def loud_before(
        self, moment: float, limit: float = TAIL_SEARCH, floor: float | None = None
    ) -> float:
        """The last moment at or before ``moment`` where the vocal is *not* quiet.

        Where the singing actually stops. The aligner's end is where the last
        phoneme stopped and never holds the final vowel (a belt note can be two
        seconds of held singing), while the ASR's line end can sit past the
        singing in silence. Both are answered by this: stretch the last
        character to here, and no further.

        The frame found this way can belong to the *next* phrase: the ASR often
        ends a line just before the following one starts, and the first loud
        frame before that end is the next phrase's onset. When a quiet stretch of
        at least ``PHRASE_GAP`` separates the two, and the run before that silence
        still ends inside this line (after ``floor``, the line's own start), the
        earlier run's end is where this line stopped. A run that begins before
        ``floor`` is this line's own singing, so it is never cut back.
        """
        if not self.smoothed:
            return moment
        index = min(int(moment / self.hop), len(self.smoothed) - 1)
        stop = max(0, index - int(limit / self.hop))
        for k in range(index, stop - 1, -1):
            if self.smoothed[k] > self.threshold:
                break
        else:
            return moment
        # Walk to the start of the loud run ``k`` belongs to, then look for a
        # phrase break before it. That look-back reaches further than the tail
        # search: the run at the line end is often a breath, with the real break
        # several seconds earlier.
        floor_index = 0 if floor is None else min(int(floor / self.hop), index)
        scan = max(floor_index, index - int(PHRASE_SCAN / self.hop))
        start = k
        while start > scan and self.smoothed[start - 1] > self.threshold:
            start -= 1
        gap = start - 1
        while gap >= scan and self.smoothed[gap] <= self.threshold:
            gap -= 1
        if (
            gap > floor_index
            and (start - 1 - gap) * self.hop >= PHRASE_GAP
            and gap >= scan
        ):
            # ``k`` sits in the next phrase; the run before the silence is ours.
            k = gap
        # The frame is loud up to its own end, not just its start.
        return min(moment, (k + 1) * self.hop)


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * share))]


def _window(
    lines: list[Line], index: int, line: Line, quiet: "QuietTrack | None" = None
) -> tuple[float, float]:
    """The clip window for one line: its own span, never overlapping a neighbour.

    The start is the line's own start — no padding, see ``PAD_SEC``. When the
    vocal track has a quiet frame just outside the line, the edge moves out to
    it instead, so the clip begins and ends where the singing does.

    A line the ASR collapsed to a point leaves no clip at all. That is not a
    line with no singing: the phrase is there, entirely before the point the ASR
    kept (its start equals its end, and both equal the next line's start). So
    when the window comes out too short, it grows back to the start of the loud
    run the line sits in — but no further than the previous line's start, so the
    two can never be placed on the same singing.
    """
    start = line.start or 0.0
    end = line.end if line.end is not None else start
    lo = max(0.0, start - PAD_SEC)
    hi = end + TAIL_SEC
    if quiet:
        # Only outward, and only as far as the search allows: a clip that
        # reaches into the next phrase is the problem this is solving.
        lo = max(0.0, quiet.before(lo))
        hi = max(hi, quiet.after(hi))
    hi = max(hi, lo + MIN_CLIP_SEC)
    previous = lines[index - 1] if index > 0 else None
    following = lines[index + 1] if index + 1 < len(lines) else None
    if previous is not None and previous.end is not None and previous.end > lo:
        lo = previous.end
    if following is not None and following.start is not None and following.start < hi:
        hi = following.start
    if quiet and hi - lo >= MIN_CLIP_SEC:
        # A clip that holds one phrase and then a long silence is the ASR's
        # fault, not the singer's: one token can be given fifteen seconds while
        # the line is sung in the first two. The aligner then places the line's
        # first characters at the head and its last ones at the far edge, and
        # the line passes every gate with its characters spread over the whole
        # silence. Cutting the clip where the first phrase stops removes the
        # room to spread into.
        sung_end = quiet.loud_end(lo, hi)
        if sung_end - lo >= MIN_CLIP_SEC:
            silence = quiet.silence_after(sung_end, hi - sung_end)
            # A phrase break, and something sung again after it: the clip holds
            # two phrases. A line that merely stops being sung — the ASR's end
            # reaching into the outro — is quiet all the way to the clip's end,
            # and its tail is what ``loud_before`` already handles.
            if silence >= PHRASE_GAP and silence < hi - sung_end:
                # ... but only when the head could hold the line at all. A line
                # the ASR started early, before the phrase it is sung in, has
                # its characters in the *second* phrase: 土星 L6 is 24
                # characters and the phrase before the break is 0.46 s of
                # singing — cutting the clip there leaves the aligner a window
                # it cannot place the text in (score 0.67, coverage 0.06),
                # while the untruncated clip places it at 0.95. Each character
                # needs at least ``MIN_WORD_SEC``, so a head shorter than that
                # for the line's own length is not this line's phrase.
                spoken = sum(1 for char in (line.text or "") if _is_spoken(char))
                if sung_end - lo >= MIN_WORD_SEC * max(spoken, 1):
                    hi = sung_end
    if quiet and hi - lo < MIN_CLIP_SEC:
        floor = 0.0
        if previous is not None and previous.start is not None:
            floor = previous.start
        grown = max(floor, quiet.run_start(hi - quiet.hop, COLLAPSED_REACH))
        if hi - grown >= MIN_CLIP_SEC:
            return grown, hi
    return lo, hi


def _retime_asr_only(
    timeline: Timeline,
    *,
    audio_path: Path,
    work_dir: Path,
    language: str | None,
    trace: Any,
    on_progress: Callable[[float, str], None] | None,
) -> list[str]:
    """Per-character times from the recogniser alone: no forced-aligner request.

    The recogniser's tokens are per character, and a character it heard is a
    real measurement of when it was sung. What it does not give is which
    character is sung where when the token covers several of them, or how a
    line is split when it only heard part of it — so each line is re-sliced
    from the tokens heard over it (``_time_line``: reading-matched, gaps
    filled, untimed characters spread over the span it did hear).

    This is the default (``char_hfa`` off). Measured against the official 逐字
    times in ``/root/yrc`` it is 0.140 s per song against 0.085 s with a forced
    aligner fused in — the recogniser's absolute placement drifts over a whole
    song — which is why the aligner path is one switch away rather than gone. A
    line it cannot
    re-slice keeps the word stream's split spans, and every note says which of
    the two it was.
    """
    from . import char_fuse

    if not char_fuse.available(honour_switch=False):
        return [
            "逐字时间来源只启用了识别器（未开启强制对齐），"
            f"但识别服务不可用（{char_fuse.unavailable_detail()}）；"
            "逐字时间只能由已有的识别词流按行摊开估计"
        ]
    code = language_code(language)
    if code is None:
        return [
            f"识别器只支持 中文 / 日文 / 英文，本次演唱语言是 {language or '未指定'}，"
            "逐字时间只能由已有的识别词流按行摊开估计"
        ]
    report = on_progress or (lambda progress, message: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    tokens = char_fuse.second_words(
        audio_path,
        work_dir,
        language,
        on_progress=lambda fraction, message: report(0.9 * fraction, message),
        honour_switch=False,
    )
    if not tokens:
        return [
            "逐字时间来源只启用了识别器（未开启强制对齐），而识别服务没有给出逐字结果；"
            "逐字时间只能由已有的识别词流按行摊开估计"
        ]

    sliced = 0
    kept: list[tuple[int, str]] = []
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(timeline.lines):
        if line.start is None or not (line.text or "").strip():
            continue
        if line.source in KEEP_SOURCES:
            # The lyrics source's own 逐字 times survived the gate in
            # ``word_sync``; a second measurement cannot improve them.
            continue
        lo = line.start - 0.2
        hi = (line.end if line.end is not None else line.start + 0.4) + 0.4
        inside = [word for word in tokens if word.start is not None and lo <= word.start < hi]
        alt = _time_line(line.text or "", inside, source="asr") if inside else None
        entry: dict[str, Any] = {"line": index + 1, "tokens": len(inside)}
        if alt is None or not alt.words:
            entry["reason"] = "这一行的时段里没有识别到可以按读音对上的字"
            entries.append(entry)
            kept.append((index + 1, _brief(line.text)))
            continue
        words = _per_character(alt.words)
        line_end = alt.words[-1].end or (line.end or hi)
        _spread(words, line_end)
        _fill(words, line_end)
        line.words = words
        line.start, line.end = words[0].start, words[-1].end
        entry.update(
            accepted=True,
            before=[round(lo + 0.2, 3), round(hi - 0.4, 3)],
            after=[line.start, line.end],
            interpolated=sum(1 for word in words if word.source == "interpolated"),
        )
        entries.append(entry)
        sliced += 1
        report(sliced / max(len(timeline.lines), 1), f"逐字时间 {sliced} 句")

    # Lines this pass did not re-slice keep the word stream's split spans, and
    # those can overlap (several characters taken from one recognised word
    # share a start). Same treatment as everywhere else: spread in order, and
    # each character lasts until the next one starts.
    for index, _label in kept:
        line = timeline.lines[index - 1]
        words = _per_character(line.words or [])
        if not words:
            continue
        line.words = words
        line_end = line.end if line.end is not None else (words[-1].end or 0.0)
        _spread(words, line_end)
        _fill(words, line_end)
    for line in timeline.lines:
        if line.source in KEEP_SOURCES or not line.words:
            continue
        line.words = _per_character(line.words)
        if line.words and line.start is not None:
            line.start = line.words[0].start
            line.end = line.words[-1].end

    if trace is not None:
        trace.step(
            "char_align",
            {
                "source": "asr",
                "service": settings.asr_url,
                "language": code,
                "lines": len(entries),
                "aligned": sliced,
                "kept": len(kept),
                "fuse": {"available": True, "mode": "asr-only", "words": len(tokens)},
                "windows": entries,
            },
        )
    if not sliced:
        return [
            "逐字时间来源只启用了识别器（未开启强制对齐），而识别结果没能对上任何一句；"
            "逐字时间只能由已有的识别词流按行摊开估计"
        ]
    note = (
        f"逐字时间来自识别器（{sliced}/{len(entries)} 句；本次没有启用强制对齐），"
        "每句按识别词流的逐字 token 重新切开"
    )
    if kept:
        names = "、".join(f"第 {number} 句（{label}）" for number, label in kept)
        note += f"；{names} 识别没有对上，只有词流按行摊开的估计时间"
    note += "。需要更准时可以开启「强制对齐逐字」：两个后端一致时取识别器，分歧时用对齐器"
    return [note]


def _fuse_second_opinion(
    *,
    audio_path: Path,
    work_dir: Path,
    language: str | None,
    accepted: list[Line],
    report: Callable[[float, str], None],
) -> dict[str, Any] | None:
    """Hand the characters the ASR aligner timed itself, where the aligner agrees.

    Per-character times come from the recogniser's word stream: where the two
    backends agree within ``FUSE_TOLERANCE`` the ASR's boundary is the one
    kept (agreement is what proves the recogniser is on the right stretch).
    Where they disagree, one of them is on the wrong stretch and the ASR is
    the one that drifts over a whole song, so the aligner keeps the character;
    a character the ASR only interpolated keeps the aligner's time as well.

    Only lines this pass placed itself take part: a kept line is already on the
    word stream's spread, and a word-synced line is the lyrics source's own
    verified timing — neither is a measurement the ASR aligner can refine.
    """
    if not accepted:
        return None
    from . import char_fuse

    if not char_fuse.available():
        return {"available": False, "reason": char_fuse.unavailable_detail()}
    tokens = char_fuse.second_words(
        audio_path, work_dir, language,
        on_progress=lambda fraction, message: report(0.9 + 0.1 * fraction, message),
    )
    if not tokens:
        return {"available": True, "words": 0, "reason": "第二个逐字对齐来源没有给出可用结果"}

    fused_lines = 0
    fused_chars = 0
    details: list[dict[str, Any]] = []
    for line in accepted:
        if not line.words or line.start is None or line.end is None:
            continue
        lo = line.start - 0.2
        hi = line.end + 0.4
        inside = [word for word in tokens if word.start is not None and lo <= word.start < hi]
        if not inside:
            continue
        alt = _time_line(line.text or "", inside, source="asr-align")
        moved = _fuse_line(line, alt, FUSE_TOLERANCE)
        if not moved:
            continue
        fused_lines += 1
        fused_chars += moved
        details.append(
            {
                "text": _brief(line.text),
                "start": round(line.start, 3),
                "end": round(line.end, 3),
                "blended": moved,
            }
        )
    logger.info("fused %d characters on %d lines with the ASR aligner", fused_chars, fused_lines)
    return {
        "available": True,
        "words": len(tokens),
        "tolerance": FUSE_TOLERANCE,
        "lines": fused_lines,
        "blended": fused_chars,
        "detail": details,
    }


def _fuse_line(line: Line, alt: Line, tolerance: float) -> int:
    """Give each character the ASR's time where the two backends agree; returns how many moved.

    The ASR's own table is the source of record, so where it and the aligner agree
    within ``tolerance`` the ASR's start is kept outright — the agreement is
    what proves the recogniser is on the right stretch, and its boundary is the
    one the requirement asks for. Where they disagree the ASR is somewhere else
    in the song and the aligner keeps the character; a character the ASR only
    interpolated was never measured, so it keeps the aligner's time too.

    Taking one of the two boundaries (rather than their average) can still put
    two neighbours out of order — their gap was smaller than the difference
    between the tables — so the order the lyric has is restored afterwards, and
    every character again lasts until the next one starts.
    """
    if len(line.words) != len(alt.words):
        return 0
    starts: list[float | None] = []
    adopted: list[bool] = []
    moved = 0
    for mine, other in zip(line.words, alt.words):
        if mine.start is None or other.start is None or other.source == "interpolated":
            starts.append(mine.start)
            adopted.append(False)
            continue
        if abs(mine.start - other.start) <= tolerance:
            starts.append(other.start)
            adopted.append(True)
            moved += 1
        else:
            starts.append(mine.start)
            adopted.append(False)
    if not moved:
        return 0
    previous: float | None = None
    for index, word in enumerate(line.words):
        start = starts[index]
        if start is None:
            continue
        if previous is not None and start < previous:
            start = previous
        word.start = round(start, 3)
        if adopted[index]:
            # Say where the time came from: this character's boundary is the
            # recogniser's own, not the forced aligner's.
            word.source = "asr"
        previous = start
    # Taking one table's boundary where the other's was milliseconds away can
    # still pull two neighbours together, so the same spreading an aligned line
    # gets has to run here too, or the blend leaves characters too narrow to
    # ever light up (土星: 24 of 411 characters below 20 ms, most of them on
    # fused lines).
    _spread(line.words, line.end or line.words[-1].start)
    _fill(line.words, line.end or line.words[-1].start)
    line.start = line.words[0].start
    line.end = line.words[-1].end
    return moved


def retime(
    timeline: Timeline,
    *,
    audio_path: Path,
    work_dir: Path,
    language: str | None,
    trace: Any = None,
    on_progress: Callable[[float, str], None] | None = None,
    use_hfa: bool | None = None,
) -> list[str]:
    """Give every line its per-character times.

    Runs over the finished timeline: the lines, their text and their order are
    already decided by the ASR + DP, and this pass only settles the times of
    the characters inside each line.

    Two sources, and ``use_hfa`` (default ``settings.char_hfa``, off) decides
    which take part. With the forced aligner on, it places every character line
    by line and the recogniser's own table is fused in where the two agree
    (``FUSE_TOLERANCE``). With it off, the recogniser's table is the only
    source: each line is re-sliced from the tokens heard over it, and lines the
    recogniser did not hear keep the word stream's split spans.

    Returns run notes describing what happened; an empty note list means the
    step was skipped (no usable service, or a language it cannot handle).
    """
    if use_hfa is None:
        use_hfa = settings.char_hfa
    if not use_hfa:
        return _retime_asr_only(
            timeline,
            audio_path=audio_path,
            work_dir=work_dir,
            language=language,
            trace=trace,
            on_progress=on_progress,
        )
    if not available():
        return []
    code = language_code(language)
    if code is None:
        return [
            f"逐字对齐服务只支持 中文 / 日文 / 英文，本次演唱语言是 {language or '未指定'}，"
            "逐字时间只能由识别词流按行摊开估计"
        ]

    report = on_progress or (lambda progress, message: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    quiet = QuietTrack(audio_path)
    targets = [
        (index, line)
        for index, line in enumerate(timeline.lines)
        if line.start is not None
        and (line.text or "").strip()
        and line.source not in KEEP_SOURCES
    ]
    if not targets:
        return []

    aligned = 0
    kept: list[tuple[int, str]] = []
    #: Extras the aligner proved are not sung at all: an "asr-extra" line is
    #: speculative by definition, so a line with no singing behind it is a
    #: hallucination over an instrumental intro, not a line to keep.
    dropped: list[tuple[int, str]] = []
    #: Lines the aligner moved: the ASR put them on the wrong stretch of audio.
    moved: list[tuple[int, str, float]] = []
    #: Lines whose clip start was found by scanning back, not taken from the ASR.
    rescanned: list[tuple[int, str, float]] = []
    accepted_lines: list[Line] = []
    entries: list[dict[str, Any]] = []
    #: Lines whose text was Japanese while the project's language was not.
    switched = 0
    for number, (index, line) in enumerate(targets, 1):
        lo, hi = _window(timeline.lines, index, line, quiet)
        entry: dict[str, Any] = {"line": index + 1, "window": [round(lo, 3), round(hi, 3)]}
        text = (line.text or "").strip()
        # Kana only appear in Japanese, so a line that has any of them is
        # Japanese whatever the project was labelled with — and the label is
        # often a title guess (土星 is sung in Japanese and its title is the
        # Chinese reading of it). Asking the aligner to align Japanese as
        # Chinese does not place the text badly, it refuses it: on 土星 the
        # same windows gave 17/28 lines under zh against 25/28 under ja.
        line_code = code
        if code == "zh" and KANA_RE.search(text):
            line_code = "ja"
            switched += 1
            entry["lang"] = "ja"
        lower = timeline.lines[index - 1].end if index > 0 else None
        following = timeline.lines[index + 1] if index + 1 < len(timeline.lines) else None
        upper = following.start if following is not None else None
        fallback = (line.start or lo, line.end or hi)
        line_span = (line.end or hi) - (line.start or lo)
        # Both scans are bounded by where the previous line *starts*, not where
        # it ends: its end is the least reliable number in the timeline (the
        # last character is stretched to wherever the singing stopped), so a
        # floor taken from it can sit past the phrase being looked for — Dear
        # Mr 「F」 L35 ends at 187.12 s while its phrase really stops at 183.29 s.
        previous_start = timeline.lines[index - 1].start if index > 0 else None
        short = hi - lo < MIN_CLIP_SEC
        scanned: list[_Attempt] = []
        first: _Attempt | None = None
        attempts: list[_Attempt] = []
        best: _Attempt | None = None
        placed: _Placed | None = None
        if short:
            # The ASR left no clip at all: its start and end are the same point,
            # which is also where the next line starts. Growing back to the loud
            # run's start (``_window``) can still come up empty when a breath
            # right before the point is the only loud frame there (我的故事 L11:
            # the run is 85.54-85.76 s, 0.22 s). The phrase is not gone, though,
            # and the aligner scores a start directly, so look for it before
            # handing the line to the word stream.
            if settings.char_align_search:
                scanned = _scan_back(
                    audio_path,
                    work_dir,
                    start=min(lo, hi - MIN_CLIP_SEC),
                    hi=hi,
                    floor=max(previous_start or 0.0, lo - KEPT_SCAN_REACH),
                    step=KEPT_SCAN_STEP,
                    text=text,
                    code=line_code,
                    fallback=fallback,
                    span=line_span,
                )
                rescue = [
                    attempt
                    for attempt in scanned
                    if attempt.words is not None
                    and attempt.share >= KEPT_SCAN_SHARE
                    and attempt.coverage >= KEPT_SCAN_COVERAGE
                ]
                if rescue:
                    top = max(attempt.score for attempt in rescue)
                    pick = min(
                        (attempt for attempt in rescue if attempt.score >= top - SCAN_TIE),
                        key=lambda attempt: attempt.lo,
                    )
                    chosen = _judge(pick, ref_start=pick.lo, ref_end=pick.hi, quiet=quiet)
                    if chosen.words is not None:
                        best, placed = pick, chosen
            if best is None:
                entry["reason"] = "这句太短，没有可以对齐的片段"
                entries.append(entry)
                kept.append((index + 1, _brief(line.text)))
                continue
        else:
            first = _align_window(
                audio_path, work_dir, lo, hi, text, line_code, fallback, line_span
            )
            attempts = [first]
            if settings.char_align_search and _needs_search(first):
                # The ASR's window did not hold up. Look around it: the line is
                # still there, just not where the word stream said. Neighbours
                # bound the shifts so two lines cannot end up on the same
                # stretch.
                for offset in SEARCH_OFFSETS:
                    lo_try = max(lo + offset, lower if lower is not None else 0.0)
                    hi_try = min(hi + offset, upper if upper is not None else float("inf"))
                    if hi_try - lo_try < MIN_CLIP_SEC or abs(lo_try - lo) < 1e-3:
                        continue
                    attempts.append(
                        _align_window(
                            audio_path, work_dir, lo_try, hi_try, text, line_code,
                            fallback, line_span, offset=lo_try - lo,
                        )
                    )
            best = first
            for attempt in attempts[1:]:
                if attempt.score > best.score + SEARCH_MARGIN:
                    best = attempt

        def judge(attempt: _Attempt) -> _Placed:
            # A window found by scanning back was found *because* the ASR's own
            # placement did not hold up, so judging it against that placement
            # would be circular: the clip it was aligned in is the only span
            # left that means anything.
            if attempt.scanned:
                return _judge(
                    attempt, ref_start=attempt.lo, ref_end=attempt.hi, quiet=quiet
                )
            return _judge(
                attempt,
                ref_start=(line.start or lo) + attempt.offset,
                ref_end=(line.end or hi) + attempt.offset,
                quiet=quiet,
            )

        # A line that came from a scanned start is already chosen and
        # judged; the searches below belong to a line the ASR did place.
        if first is not None:
            placed = judge(best)
            # The ASR collapsed this line to (almost) a point, so where the clip
            # starts is a guess made from the loud run. The aligner scores a start
            # directly: sweep it back and keep the earliest start among the windows
            # that tie the best score — a phrase is sung before the ASR admits it.
            collapsed = (line.end or lo) - (line.start or lo) < MIN_CLIP_SEC
            if settings.char_align_search and collapsed:
                point = line.start or lo
                scanned = _scan_back(
                    audio_path,
                    work_dir,
                    start=min(point, hi - MIN_CLIP_SEC),
                    hi=hi,
                    floor=max(previous_start or 0.0, point - COLLAPSED_SCAN_REACH),
                    step=COLLAPSED_SCAN_STEP,
                    text=text,
                    code=line_code,
                    fallback=fallback,
                    span=line_span,
                )
                pool = attempts + scanned
                top = max(attempt.score for attempt in pool)
                # Same margin as the shifted search: a window that only ties the
                # ASR's own placement is not evidence the phrase is elsewhere, and
                # moving on a tie costs more than it gains (measured: 我的故事 L33
                # scores 1.00 both at the ASR's 218.24 s and two phrases back at
                # 215.74 s, and moving it there costs 0.08 s → 2.53 s). The score is
                # flat across a phrase, so the earliest of the tied windows is the
                # one the phrase starts in — but only if the gates accept it: a
                # window that scores well and still cannot place the characters is
                # not a start, and taking it would leave the line kept while the
                # note claims it was found (あいつら L51: 149.53 s scores 1.00 and
                # the answer covers 0.03 s, so the line is kept — correctly).
                for pick in sorted(
                    (attempt for attempt in pool if attempt.score >= top - SCAN_TIE),
                    key=lambda attempt: attempt.lo,
                ):
                    if pick is best or pick.score <= best.score + SEARCH_MARGIN:
                        continue
                    chosen = judge(pick)
                    if chosen.words is not None:
                        best, placed = pick, chosen
                        break
            elif settings.char_align_search and placed.words is None and not placed.nothing:
                # The aligner could not place this line — usually because the ASR
                # put it several seconds late, past what the shifted search
                # reaches. Two ways out, tried in this order:
                #
                # * a *squeezed* answer did place the characters, just over the
                #   head of the phrase: the clip ends before the singing does, so
                #   moving the start inside it cannot help (coverage is capped by
                #   the clip's own length) — the whole window moves instead, and
                #   the best window that passes the gates wins;
                # * a line with no answer at all needs its start walked back, and
                #   only a window that both places most characters directly and
                #   covers the phrase is taken: one that merely scores a little
                #   higher than the failed one is another guess, and the word
                #   stream is the better answer then.
                if placed.squeezed:
                    for attempt in _scan_shift(
                        audio_path,
                        work_dir,
                        lo=lo,
                        hi=hi,
                        reach=SQUEEZE_SCAN_REACH,
                        step=SQUEEZE_SCAN_STEP,
                        lower=lower,
                        upper=upper,
                        text=text,
                        code=line_code,
                        fallback=fallback,
                        span=line_span,
                    ):
                        if attempt.words is None:
                            continue
                        chosen = judge(attempt)
                        if chosen.words is None:
                            continue
                        # The rejected placement is no baseline to beat: any
                        # window the gates accept is better than the word
                        # stream, so among those the highest score wins.
                        if best is first or attempt.score > best.score:
                            best, placed = attempt, chosen
                if placed.words is None:
                    scanned = _scan_back(
                        audio_path,
                        work_dir,
                        start=lo,
                        hi=hi,
                        floor=max(previous_start or 0.0, lo - KEPT_SCAN_REACH),
                        step=KEPT_SCAN_STEP,
                        text=text,
                        code=line_code,
                        fallback=fallback,
                        span=line_span,
                    )
                    rescue = [
                        attempt
                        for attempt in scanned
                        if attempt.words is not None
                        and attempt.share >= KEPT_SCAN_SHARE
                        and attempt.coverage >= KEPT_SCAN_COVERAGE
                    ]
                    if rescue:
                        pick = max(rescue, key=lambda attempt: (attempt.score, -attempt.lo))
                        rescued = judge(pick)
                        if rescued.words is not None:
                            best, placed = pick, rescued
            if len(attempts) > 1:
                entry["search"] = [
                    {"offset": round(attempt.offset, 2), "score": attempt.score}
                    for attempt in attempts
                ]
            if best is not first:
                entry["moved"] = round(best.offset, 2)
        if best.scanned:
            entry["scanned"] = round(best.lo, 2)
            entry["scan"] = [
                {"start": round(attempt.lo, 2), "score": attempt.score}
                for attempt in scanned
            ]
            rescanned.append((index + 1, _brief(line.text), best.lo))
        words = placed.words
        if placed.nothing:
            # Nothing was placed: the clip holds no singing this text fits.
            if line.source == "asr-extra" and len(timeline.lines) > 1:
                # An extra is speculative by definition, so a line with no
                # singing behind it is an ASR hallucination over an
                # instrumental intro — keeping it would only leave a line with
                # invented timings in the timeline.
                entry["reason"] = (
                    f"这句在伴奏里没有唱到人声（只对上 {placed.heard:.2f}s），"
                    "是识别多出来的衬词，已删掉"
                )
                entries.append(entry)
                dropped.append((index, _brief(line.text)))
                continue
            entry["reason"] = placed.reason
            entries.append(entry)
            kept.append((index + 1, _brief(line.text)))
            continue
        if words is None:
            entry["reason"] = placed.reason
            entries.append(entry)
            kept.append((index + 1, _brief(line.text)))
            continue
        sung_end = placed.sung_end
        last_end = placed.last_end
        # A window found by scanning back starts where the phrase does, which can
        # be before the previous line's end — and that end is itself the least
        # reliable number in the timeline (its last character is stretched to
        # where the singing stopped). Bound the line by the previous line's
        # *start* instead, so two lines cannot both claim the same phrase.
        clamp_lower = lower
        if best.scanned and index > 0:
            clamp_lower = timeline.lines[index - 1].start
        _clamp(words, clamp_lower, upper)

        # The last character carries the final held vowel, which the aligner
        # never times: stretch it to where the singing stops, not into the
        # silence the ASR's line end may have reached into.
        line_end = max(last_end, min(sung_end, upper or float("inf")))
        _spread(words, line_end)
        _fill(words, line_end)
        entry.update(
            accepted=True,
            before=[round(line.start or 0.0, 3), round(line.end or 0.0, 3)],
            after=[words[0].start, words[-1].end],
        )
        entries.append(entry)
        if best is not first and not best.scanned:
            moved.append((index + 1, _brief(line.text), best.offset))
        line.words = words
        line.start = words[0].start
        line.end = words[-1].end
        accepted_lines.append(line)
        aligned += 1
        report(number / len(targets), f"逐字对齐 {number}/{len(targets)} 句")
        logger.info(
            "line %d re-timed by the aligner: %.2f-%.2fs -> %.2f-%.2fs",
            index + 1,
            entry["before"][0],
            entry["before"][1],
            entry["after"][0],
            entry["after"][1],
        )

    fused = _fuse_second_opinion(
        audio_path=audio_path,
        work_dir=work_dir,
        # The second opinion is a recognition, so it needs the language the
        # text is in, not the one the project was labelled with: kana in the
        # lyrics means the singing is Japanese, and recognising it as Chinese
        # would produce tokens that fuse nothing.
        language="Japanese" if switched else language,
        accepted=accepted_lines,
        report=report,
    )

    if trace is not None:
        trace.step(
            "char_align",
            {
                "service": settings.char_aligner_url,
                "language": code,
                "japanese_lines": switched,
                "lines": len(targets),
                "aligned": aligned,
                "kept": len(kept),
                "dropped": len(dropped),
                "fuse": fused,
                "windows": entries,
            },
        )
    # The lines this pass left alone keep the word stream's split spans, and
    # those can overlap (several characters taken from one recognised word share
    # a start). Same treatment as an aligned line: characters stacked on one
    # instant are spread in order, and each lasts until the next one starts.
    # The word stream's own starts are never moved otherwise.
    for index, _label in kept:
        line = timeline.lines[index - 1]
        words = _per_character(line.words or [])
        if not words:
            continue
        line.words = words
        line_end = line.end if line.end is not None else (words[-1].end or 0.0)
        _spread(words, line_end)
        _fill(words, line_end)
    # Every line in the timeline hands out one time per character, whichever
    # path timed it: a recogniser token covering two characters would be one
    # highlight block for both, and the goal is a per-character timeline.
    for line in timeline.lines:
        if line.source in KEEP_SOURCES or not line.words:
            continue
        line.words = _per_character(line.words)
        if line.words and line.start is not None:
            line.start = line.words[0].start
            line.end = line.words[-1].end
    if dropped:
        gone = {index for index, _ in dropped}
        timeline.lines = [
            line for index, line in enumerate(timeline.lines) if index not in gone
        ]
        logger.info("dropped %d unsung extra line(s) after alignment", len(dropped))

    skipped = sum(
        1
        for line in timeline.lines
        if line.source in KEEP_SOURCES and (line.text or "").strip()
    )
    if aligned == 0:
        note = "逐字对齐服务没有为任何一句给出结果，逐字时间仍由识别词流给出"
        if skipped:
            note = (
                f"逐字对齐服务没有为任何一句给出结果；这 {skipped} 行用的是歌词源的逐字时间，"
                "其余只有识别词流给出的估计时间"
            )
        if dropped:
            names = "、".join(f"第 {index + 1} 句（{label}）" for index, label in dropped)
            note += f"；{names} 在伴奏里没有人声（识别多出来的衬词），已删掉"
        if switched:
            note += f"；歌词里有假名，其中 {switched} 句是按日文对齐的"
        return [note]
    note = (
        f"逐字时间以识别词流为准（{aligned}/{len(targets) - len(dropped)} 句），"
        "强制对齐兜住识别没听到、或两个后端相差过大的字"
    )
    if switched:
        note += (
            f"；歌词里有假名，其中 {switched} 句按日文对齐"
            f"（项目语言标的是 {language or '未指定'}）"
        )
    if dropped:
        names = "、".join(f"第 {index + 1} 句（{label}）" for index, label in dropped)
        note += f"；{names} 在伴奏里没有人声（识别多出来的衬词），已删掉"
    if kept:
        names = "、".join(f"第 {number} 句（{label}）" for number, label in kept)
        note += (
            f"；{names} 强制对齐未成功，这些行只有识别词流按行摊开的估计时间，"
            "不是逐字对齐结果"
        )
    if moved:
        places = "、".join(
            f"第 {number} 句（{label}）{offset:+.1f}s" for number, label, offset in moved
        )
        note += f"；{places} 识别放偏了，已按对齐结果挪回"
    if rescanned:
        places = "、".join(
            f"第 {number} 句（{label}）从 {start:.2f}s 起" for number, label, start in rescanned
        )
        note += f"；{places} 识别给的位置对不上，按对齐分数把片段起点往前找过"
    if skipped:
        note += f"；另有 {skipped} 行用的是歌词源的逐字时间，未参与强制对齐"
    if fused and fused.get("blended"):
        note += (
            f"；其中 {fused['blended']} 个字与 ASR 识别词流给出的时间相差在 "
            f"{FUSE_TOLERANCE:g} 秒以内，逐字时间按识别词流给出"
            "（识别没听到、或两个后端相差更大的字仍按强制对齐结果）"
        )
    return [note]
