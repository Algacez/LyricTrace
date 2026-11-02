"""Block-wise alignment: short audio blocks, each fed only the lines sung there.

Why blocks
----------
A forced aligner has to spread the whole text across the whole audio it is
given. Hand it a five-minute song and eight hundred characters and it compresses
the tail: measured on a 257.6 s song, the last character landed at 186.8 s and
line starts were off by 9.5 s median — every character still timed, all of them
monotonic, nothing visibly broken. Hand it twelve seconds and two lines and its
per-character output is back in the range it was trained on: measured median
0.049 s, p90 0.94 s, against a reference from the same recording.

Why an anchor
-------------
Which lines belong to which block **cannot** be read out of the aligner's own
output. Given one line and a whole block it stretches that line over the block;
given eight lines and the same block it squeezes them in. Both look plausible
(char duration 0.36 s vs 0.67 s, no gap, no error), so every "count the lines
from the aligner's response" heuristic ends up 5–35 s off. The plan therefore
needs an anchor from outside: the source lyric timeline when the provider
supplied one. Anchor times are used **only** to decide which block a line
belongs to — no anchor timestamp ever reaches the output, every time in the
result comes from the aligner.

One signal from the aligner *is* usable, as a guard rather than a decision: when
a block is given far more text than it holds, the surplus characters come back
at the service's minimum width (0.020 s, back to back). That is reported, never
silently repaired.
"""

from __future__ import annotations

import array
import math
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FRAME = 0.02
"""Envelope resolution in seconds."""

MERGE_GAP = 0.30
"""Silences shorter than this do not split a sung run."""

MIN_RUN = 0.25
"""A run of loud frames shorter than this is noise, not singing."""

GROUP_GAP = 1.20
"""A pause at least this long ends a block."""

MAX_BLOCK = 12.0
"""Longest block handed to the aligner. Longer blocks start to compress."""

MIN_BLOCK = 1.2
"""The service refuses anything shorter than this."""

EDGE_PAD = 0.10
"""Kept around a block so a syllable at the boundary is not cut in half."""

COLLAPSE = 0.035
"""Characters at or below this width mean the block got more text than it holds.

The service squeezes surplus text to exactly 0.020 s; real fast singing stays
above 0.05 s, so the line between the two is sharp.
"""

WINDOW_LEAD = 0.20
"""Audio kept before a line's anchor when each line gets its own window."""

WINDOW_TAIL = 0.15
"""Audio kept after the next line's anchor, so the tail of a held note survives."""

MAX_WINDOW = 24.0
"""Longest window a line may be given when the block boundary cuts through it."""

MIN_OVERLAP = 0.20
"""Least overlap a line must have with a block to count as sung in it.

A two-letter line whose anchor sat in a silent stretch still overlapped the next
block by 0.02 s, which was enough to hand it that block's audio and time it
5.0 s from where it was actually sung.
"""

MAX_LINE_SPAN = 6.0
"""Cap on how long a lyric line is assumed to last when picking its block.

Providers set a line's end to the next line's start, so a line followed by a
long instrumental is described as lasting ten or twenty seconds — enough to
steal the next block (measured: one repeated two-character line landed a block
late and dragged its neighbour 5.7 s out of place)."""

@dataclass
class Plan:
    """Which lyric lines are sung inside which block."""

    blocks: list[tuple[float, float]]
    assignment: list[list[int]] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)
    unplaced: list[int] = field(default_factory=list)
    interpolated: list[int] = field(default_factory=list)
    unplaceable: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class BlockResult:
    """Per-character times keyed by lyric line, plus what happened."""

    chars_by_line: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    plan: Plan | None = None
    requests: int = 0
    notes: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


AlignFn = Callable[[Path, str, float], dict[str, Any]]
"""``align_fn(slice_path, text, offset) -> {"chars": [...]}`` with absolute times."""


# --------------------------------------------------------------------------- #
# vocal envelope -> blocks
# --------------------------------------------------------------------------- #
def envelope(wav_path: Path) -> tuple[list[float], float]:
    """20 ms RMS of the vocal track, plus the level above which a frame counts as sung."""
    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        raw = handle.readframes(frames)
    if width != 2:
        raise ValueError(f"需要 16-bit PCM，实际是 {8 * width} bit")
    samples = array.array("h")
    samples.frombytes(raw)
    if channels > 1:
        samples = array.array("h", samples[::channels])
    step = max(1, int(rate * FRAME))
    rms = [
        math.sqrt(sum(value * value for value in samples[i : i + step]) / step) / 32768.0
        for i in range(0, len(samples) - step + 1, step)
    ]
    ordered = sorted(rms)
    # Loud enough to be singing in this recording, not merely louder than the
    # quietest tenth — a separated live stem keeps audience noise well above zero.
    threshold = max(
        ordered[int(0.10 * (len(ordered) - 1))] * 3.0,
        ordered[int(0.95 * (len(ordered) - 1))] * 0.08,
        0.0025,
    )
    return rms, threshold


def blocks_from(rms: list[float], threshold: float) -> list[tuple[float, float]]:
    """Sung stretches, merged into blocks of at most ``MAX_BLOCK`` seconds."""
    runs: list[list[int]] = []
    for index, value in enumerate(rms):
        if value <= threshold:
            continue
        if runs and index - runs[-1][1] <= MERGE_GAP / FRAME:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    runs = [run for run in runs if (run[1] - run[0] + 1) * FRAME >= MIN_RUN]

    groups: list[list[list[int]]] = []
    for run in runs:
        if groups:
            last = groups[-1][-1]
            if (
                (run[0] - last[1]) * FRAME <= GROUP_GAP
                and (run[1] - groups[-1][0][0]) * FRAME <= MAX_BLOCK
            ):
                groups[-1].append(run)
                continue
        groups.append([run])

    out: list[tuple[float, float]] = []
    for group in groups:
        out.extend(_split(group, rms))
    return out


def _split(group: list[list[int]], rms: list[float]) -> list[tuple[float, float]]:
    """Cut an over-long group at its longest pause; without a pause, at its quietest frame.

    A live vocal stem can stay above the threshold for a minute (applause, bleed,
    reverb tails), so waiting for a real silence is not enough.
    """
    span = (group[-1][1] - group[0][0]) * FRAME
    if span <= MAX_BLOCK:
        lo = max(0.0, group[0][0] * FRAME - EDGE_PAD)
        hi = (group[-1][1] + 1) * FRAME + EDGE_PAD
        return [(lo, hi if hi - lo >= MIN_BLOCK else lo + MIN_BLOCK)]
    if len(group) == 1:
        start, stop = group[0][0], group[0][1]
        low = start + int(0.35 * (stop - start))
        high = start + int(0.65 * (stop - start))
        cut_at = min(range(low, max(low + 1, high)), key=lambda i: rms[i])
        return _split([[start, cut_at]], rms) + _split([[cut_at + 1, stop]], rms)
    gaps = [((group[i + 1][0] - group[i][1]) * FRAME, i) for i in range(len(group) - 1)]
    _gap, cut = max(gaps)
    return _split(group[: cut + 1], rms) + _split(group[cut + 1 :], rms)


# --------------------------------------------------------------------------- #
# anchors -> plan
# --------------------------------------------------------------------------- #
def fill_anchors(anchors: list[float | None]) -> tuple[list[float], list[int]]:
    """Fill missing anchors from their neighbours; returns the filled list and which were invented.

    Used for the block plan only. A line whose anchor was invented is listed so
    the caller can say so — it never changes how that line is *timed*, the
    aligner still does that.
    """
    resolved: list[float] = []
    invented: list[int] = []
    for index, value in enumerate(anchors):
        if value is not None:
            resolved.append(value)
            continue
        before = next((anchors[i] for i in range(index - 1, -1, -1) if anchors[i] is not None), None)
        after = next(
            (anchors[i] for i in range(index + 1, len(anchors)) if anchors[i] is not None), None
        )
        if before is not None and after is not None:
            resolved.append((before + after) / 2)
        elif before is not None:
            resolved.append(before)
        elif after is not None:
            resolved.append(after)
        else:
            resolved.append(0.0)
        invented.append(index)
    return resolved, invented


def _spans(starts: list[float], ends: list[float | None] | None) -> list[tuple[float, float]]:
    """Line spans for the block plan; a missing end runs up to the next line's start."""
    spans: list[tuple[float, float]] = []
    for index, start in enumerate(starts):
        end = ends[index] if ends and index < len(ends) else None
        if end is None:
            following = next((s for s in starts[index + 1 :] if s > start + 0.05), None)
            end = following if following is not None else start + 3.0
        if end <= start:
            end = start + 0.5
        spans.append((start, min(end, start + MAX_LINE_SPAN)))
    return spans


def plan(
    blocks: list[tuple[float, float]],
    anchors: list[float | None],
    ends: list[float | None] | None = None,
) -> Plan:
    """Assign each lyric line to the block that holds most of it.

    Assigning by "the line starts inside this block" is not enough: a block ends
    at a pause, and a line can start a fraction of a second before that pause and
    run on into the next block (measured: one line starting 0.13 s before the cut
    pulled the whole following block 5.7 s out of place). Overlap decides.
    """
    result = Plan(blocks=blocks, assignment=[[] for _ in blocks])
    if not blocks:
        result.unplaced = list(range(len(anchors)))
        result.notes.append("人声里没找到可以切开的演唱段，无法分块对齐")
        return result
    filled, invented = fill_anchors(anchors)
    result.starts = filled
    result.interpolated = invented
    spans = _spans(filled, ends)
    cursor = 0
    for index, (start, end) in enumerate(spans):
        best: tuple[float, int] | None = None
        for block_index in range(cursor, len(blocks)):
            lo, hi = blocks[block_index]
            overlap = min(end, hi) - max(start, lo)
            if overlap < MIN_OVERLAP:
                continue
            if best is None or overlap > best[0] + 1e-6:
                best = (overlap, block_index)
        if best is None:
            # No overlap with any block: the envelope never heard singing here.
            # Snapping the line to the nearest block was measured and is worse —
            # it puts the line inside a stretch of audio that belongs to a
            # different line (one two-letter line landed 5.0 s off). Such a line
            # gets no window at all and is reported instead.
            result.unplaceable.append(index)
            continue
        target = best[1]
        cursor = target
        result.assignment[target].append(index)
    empty = [i for i, lines in enumerate(result.assignment) if not lines]
    if empty:
        result.notes.append(
            f"有 {len(empty)} 个演唱段没有分到歌词（第 "
            + "、".join(str(i + 1) for i in empty[:6])
            + " 段），这些段里的人声不会被输出"
        )
    if invented:
        result.notes.append(
            f"{len(invented)} 行没有源自带时间，按相邻行推断它落在哪一段（只影响分块，不影响时间）"
        )
    if result.unplaceable:
        result.notes.append(
            f"{len(result.unplaceable)} 行的时间点落在人声静音处（分离后的人声里听不到这段），"
            "这些行不给逐字时间、按相邻行插值并标记（第 "
            + "、".join(str(i + 1) for i in result.unplaceable[:6])
            + " 行）"
        )
    return result


# --------------------------------------------------------------------------- #
# audio slicing
# --------------------------------------------------------------------------- #
def slice_wav(source: Path, lo: float, hi: float, target: Path) -> Path:
    """Write ``[lo, hi)`` of a 16-bit PCM wav to ``target``."""
    with wave.open(str(source), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        start = max(0, int(lo * rate))
        count = max(1, int(round((hi - lo) * rate)))
        handle.setpos(min(start, handle.getnframes()))
        frames = handle.readframes(count)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return target


def chars_from_tokens(text: str, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild per-character times from the aligner's token stream.

    The service normally folds its token times onto characters itself, but not
    always: a two-character held note here (「别怕」sung over 11.7 s) came back
    with both characters untimed while its two tokens carried 218.249-218.580
    and 218.580-229.160. Rebuilding from the tokens recovers that line.

    Only used when the character mapping came back empty, and only when the
    number of timed tokens matches the number of characters that are actually
    sung — punctuation and spaces take no token, so a mismatch means the tokens
    are not this line's.
    """
    spoken = [
        index
        for index, char in enumerate(text)
        if not (char.isspace() or unicodedata.category(char).startswith("P"))
    ]
    timed = [
        token
        for token in tokens
        if token.get("start") is not None and token.get("end") is not None
    ]
    if not spoken or len(timed) != len(spoken):
        return []
    return [
        {"line": 0, "c": text[index], "start": token["start"], "end": token["end"]}
        for index, token in zip(spoken, timed)
    ]


# --------------------------------------------------------------------------- #
# the block loop
# --------------------------------------------------------------------------- #
def _jobs(
    blocks: list[tuple[float, float]],
    assignment: list[list[int]],
    interpolated: list[int],
    starts: list[float],
    duration: float,
    ends: list[float | None] | None = None,
) -> list[tuple[float, float, list[int]]]:
    """Audio windows to align, each with the lines sung inside it.

    A block holding two lines forces the aligner to share the stretch between
    them, and it shares by proportion: measured 0.19 s median per-character
    error with two or three lines in a block, against 0.02 s for the line starts
    themselves. Cutting each block down to one line at a time removes that
    division, using the same anchors the block plan already came from. Lines
    whose anchor was invented keep their whole block — there is nothing to cut
    them apart with.

    Windows stay inside their block **unless the boundary cuts through the
    line**: a held note fading under the envelope threshold is still singing, and
    one line here runs 16 s past the boundary. Clipping its window put the last
    character 15 s off. Only such lines get a window that reaches past the block
    (up to ``MAX_WINDOW``); letting *every* window run past its block was
    measured and is worse — the extra silence gets distributed into the
    characters (per-character median 0.079 s -> 0.280 s, within 100 ms
    51% -> 36%).
    """
    jobs: list[tuple[float, float, list[int]]] = []
    for block_index, (lo, hi) in enumerate(blocks):
        inside = assignment[block_index]
        if not inside:
            continue
        if any(index in interpolated for index in inside):
            jobs.append((lo, hi, list(inside)))
            continue
        if len(inside) == 1:
            # One line in the block: the block already is its window, head and
            # tail included — trimming it to the anchor was measured and costs
            # accuracy (per-character median 0.079 s -> 0.130 s). Only reach past
            # the block end when the line itself is longer than the block.
            line_index = inside[0]
            limit = ends[line_index] if ends and line_index < len(ends) else None
            start = starts[line_index] if line_index < len(starts) else None
            if limit is not None and start is not None and limit + WINDOW_TAIL > hi + 1e-6:
                reach = hi
                for later in range(block_index + 1, len(blocks)):
                    later_lo, later_hi = blocks[later]
                    if later_lo - reach > MERGE_GAP or later_hi - start > MAX_WINDOW:
                        break
                    reach = later_hi
                jobs.append((lo, min(duration, min(limit + WINDOW_TAIL, reach + WINDOW_TAIL)), [line_index]))
            else:
                jobs.append((lo, hi, [line_index]))
            continue
        for position, line_index in enumerate(inside):
            following = inside[position + 1] if position + 1 < len(inside) else None
            start = starts[line_index] if line_index < len(starts) else None
            if start is None:
                jobs.append((lo, hi, list(inside)))
                break
            limit = starts[following] if following is not None and following < len(starts) else None
            if limit is None and ends and line_index < len(ends):
                limit = ends[line_index]
            window_lo = max(lo, start - WINDOW_LEAD)
            natural_hi = (limit + WINDOW_TAIL) if limit is not None else hi
            window_hi = min(hi, natural_hi)
            if natural_hi > hi + 1e-6:
                # The line runs past its block: reach over the following blocks
                # while the pause between them is short enough to be a breath.
                reach = hi
                for later in range(block_index + 1, len(blocks)):
                    later_lo, later_hi = blocks[later]
                    if later_lo - reach > MERGE_GAP or later_hi - start > MAX_WINDOW:
                        break
                    reach = later_hi
                window_hi = min(duration, max(hi, min(natural_hi, reach + WINDOW_TAIL)))
            if window_hi - window_lo < MIN_BLOCK:
                window_lo, window_hi = lo, hi
            jobs.append((window_lo, window_hi, [line_index]))
    return jobs
def align_blocks(
    *,
    texts: list[str],
    anchors: list[float | None],
    vocal_wav: Path,
    align_fn: AlignFn,
    work_dir: Path,
    language: str | None = None,
    ends: list[float | None] | None = None,
) -> BlockResult:
    """Align the whole song block by block; every time in the result comes from ``align_fn``."""
    result = BlockResult()
    rms, threshold = envelope(vocal_wav)
    blocks = blocks_from(rms, threshold)
    result.plan = plan(blocks, anchors, ends)
    result.notes.extend(result.plan.notes)
    if not blocks:
        return result

    collapsed: list[int] = []
    failed: list[tuple[int, str, str]] = []
    with wave.open(str(vocal_wav), "rb") as handle:
        duration = handle.getnframes() / float(handle.getframerate())
    jobs = _jobs(
        blocks,
        result.plan.assignment,
        result.plan.interpolated,
        result.plan.starts,
        duration,
        ends,
    )
    for job_index, (lo, hi, inside) in enumerate(jobs):
        text = "\n".join(texts[i].strip() for i in inside)
        if not text:
            continue
        slice_path = slice_wav(vocal_wav, lo, hi, work_dir / f"block-{job_index:03d}.wav")
        try:
            answer = align_fn(slice_path, text, lo)
        except Exception as error:  # noqa: BLE001 - one bad window must not sink the song
            # Measured: a line with no characters the aligner can pronounce for
            # this language (a Latin annotation in a Chinese song) makes the
            # service answer 500. Losing that one window is recoverable; losing
            # the whole song is not.
            result.requests += 1
            failed.append((job_index, text[:20], str(error)[:120]))
            try:
                slice_path.unlink()
            except OSError:
                pass
            continue
        result.requests += 1
        by_line: dict[int, list[dict[str, Any]]] = {}
        for char in answer.get("chars") or []:
            by_line.setdefault(int(char.get("line") or 0), []).append(char)
        for offset, source in enumerate(inside):
            piece = [
                char
                for char in by_line.get(offset, [])
                if char.get("start") is not None and char.get("end") is not None
            ]
            if not piece:
                piece = chars_from_tokens(texts[source], answer.get("tokens") or [])
            if piece:
                result.chars_by_line[source] = piece
                if any(char["end"] - char["start"] <= COLLAPSE for char in piece
                       if not char.get("punct")):
                    collapsed.append(job_index)
        try:
            slice_path.unlink()
        except OSError:
            pass

    result.stats.update(
        {
            "blocks": len(blocks),
            "blocks_with_lyrics": sum(1 for lines in result.plan.assignment if lines),
            "lines": len(texts),
            "placed": len(result.chars_by_line),
            "requests": result.requests,
            "block_s": [round(hi - lo, 2) for lo, hi in blocks],
        }
    )
    missing = [i for i in range(len(texts)) if i not in result.chars_by_line]
    if missing:
        result.notes.append(
            f"有 {len(missing)} 行没有拿到逐字时间（第 "
            + "、".join(str(i + 1) for i in missing[:6])
            + " 行），这些行的时间是插值出来的，已标 interpolated"
        )
    if failed:
        result.notes.append(
            "有 "
            f"{len(failed)} 段对齐请求失败（"
            + "；".join(f"第 {index + 1} 段「{text}」：{reason}" for index, text, reason in failed[:3])
            + "），这些段的歌词没有逐字时间，已标 interpolated"
        )
    if collapsed:
        result.notes.append(
            "这些段落可能被塞进了多于它容纳量的歌词（对齐器把字挤到了最小宽度）：第 "
            + "、".join(str(i + 1) for i in sorted(set(collapsed))[:6])
            + " 段，建议核对"
        )
    return result
