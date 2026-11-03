import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useAutosave } from "../hooks/useAutosave";
import { useHistory } from "../hooks/useHistory";
import type { LyricLine, Project, ReferenceLine, Timeline, Word } from "../types";
import {
  STATUS_LABEL,
  computeStatus,
  formatTime,
  parseTimeInput,
  round,
  splitWords,
} from "../utils";
import { Waveform, type WaveformHandle } from "./Waveform";

interface Props {
  project: Project;
  onProjectChange: (project: Project) => void;
  /**
   * Where an edit is saved. The project page writes the timeline through
   * ``PUT /api/projects/{id}/timeline``; the API cache page writes a cached
   * job's result through ``PUT /api/align/{job_id}/result``. Same editor, same
   * per-character times, two records — so the store is injected rather than
   * branched on inside the editor.
   */
  save?: (timeline: Timeline) => Promise<unknown>;
  /** Where the audio comes from. Defaults to this project's own file; ``null``
   * when the record has no audio left to play. */
  audioUrl?: string | null;
}

type Focus = { line: number; word: number } | null;

/** An inclusive run of characters inside one line, the slider's working set. */
type WordRange = { line: number; from: number; to: number };

const EMPTY: Timeline = { status: "none", lines: [] };

/** Shortest span a character may be squeezed to by the slider, in seconds. */
const MIN_WORD_SECONDS = 0.05;

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

/**
 * A time the user moved is no longer the number the ASR reported: mark it
 * ``manual``. A character that was interpolated stops being one — it is not
 * the even split it claimed to be any more.
 */
function markManual(word: Word) {
  if (word.start !== null) word.source = "manual";
}

interface BarInfo {
  /** The line's own window: what the slider is drawn across. */
  bar: [number, number];
  /** Where each character sits on that bar, in order. */
  spans: [number, number][];
}

/**
 * Lay the line out on its own progress bar.
 *
 * Characters with real timings keep them; the rest are given an even slot by
 * length so a run that has not been stamped yet can still be dragged into
 * place. Nothing here is written back — it only decides where the handles sit
 * before the user moves them.
 */
function barSpans(line: LyricLine): BarInfo | null {
  const words = line.words;
  if (!words.length) return null;
  const starts = words.map((word) => word.start).filter((value): value is number => value !== null);
  const ends = words.map((word) => word.end).filter((value): value is number => value !== null);
  const barStart = line.start ?? (starts.length ? Math.min(...starts) : null);
  if (barStart === null) return null;
  // A line with starts but no ends yet still gets a bar: the right handle is
  // how that end gets set. The provisional extent is only drawn, never saved.
  const barEnd = line.end ?? (ends.length ? Math.max(...ends) : Math.max(...starts) + 1);
  if (barEnd <= barStart) return null;
  const weights = words.map((word) => Math.max(word.text.replace(/\s/g, "").length, 1));
  const total = weights.reduce((sum, value) => sum + value, 0);
  const spans: [number, number][] = [];
  let cursor = barStart;
  words.forEach((word, index) => {
    const slot = ((barEnd - barStart) * weights[index]) / total;
    if (word.start !== null && word.end !== null) spans.push([word.start, word.end]);
    else if (word.start !== null) spans.push([word.start, Math.max(word.start + MIN_WORD_SECONDS, cursor + slot)]);
    else spans.push([cursor, cursor + slot]);
    cursor += slot;
  });
  return { bar: [barStart, barEnd], spans };
}

/**
 * How far the selection's two ends may travel.
 *
 * The line window is the outer limit. Word *starts* must stay ascending (the
 * backend rejects anything else), so neither the run's first start nor its
 * last start may cross a neighbour. The run's end may run past the next
 * character's start — overlapping syllables are the user's call to make, and
 * only starts are ordered.
 */
function rangeLimits(line: LyricLine, from: number, to: number) {
  const info = barSpans(line);
  if (!info) return null;
  const words = line.words;
  const previous = from > 0 ? words[from - 1] : null;
  const next = to < words.length - 1 ? words[to + 1] : null;
  return {
    currentStart: info.spans[from][0],
    currentEnd: info.spans[to][1],
    minStart: Math.max(info.bar[0], previous?.start ?? -Infinity),
    maxStart: Math.min(next?.start ?? Infinity, info.bar[1]),
    minShift: Math.max(info.bar[0], previous?.start ?? -Infinity) - info.spans[from][0],
    maxShift: Math.min(info.bar[1] - info.spans[to][1], (next?.start ?? Infinity) - info.spans[to][0]),
  };
}

export function TimelineEditor({ project, onProjectChange, save, audioUrl }: Props) {
  const history = useHistory<Timeline>(project.timeline ?? EMPTY);
  const timeline = history.value;
  const [focus, setFocus] = useState<Focus>(null);
  // The characters the slider acts on. Kept apart from ``focus``: the keyboard
  // stamps one character at a time, the slider moves a whole run of them.
  const [range, setRange] = useState<WordRange | null>(null);
  // Follow the audio: while it plays, the list scrolls to the line being sung.
  // Off-switch is there because scrolling the list while editing a line far
  // from the playhead is exactly what a user does not want.
  const [follow, setFollow] = useState(true);
  const followed = useRef(-1);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const waveRef = useRef<WaveformHandle>(null);

  const persist = useCallback(
    async (value: Timeline) => {
      const payload = { ...value, status: computeStatus(value) };
      if (save) {
        await save(payload);
        return;
      }
      const updated = await api.saveTimeline(project.id, payload);
      onProjectChange(updated);
    },
    [project.id, onProjectChange, save],
  );

  const autosave = useAutosave(timeline, persist, {
    delay: 1000,
    enabled: timeline.lines.length > 0,
    signature: JSON.stringify(timeline.lines),
  });

  /**
   * A drag session: remember the timeline as it was before the first move, so
   * the whole drag undoes in one step (and saves once, not per pixel).
   */
  const dragBefore = useRef<Timeline | null>(null);
  const beginDrag = () => {
    dragBefore.current = history.value;
  };
  const endDrag = () => {
    const before = dragBefore.current;
    dragBefore.current = null;
    if (before) history.commit(before);
  };

  const mutate = (updater: (draft: Timeline) => void) => {
    const apply = (current: Timeline) => {
      const draft: Timeline = {
        ...current,
        lines: current.lines.map((line) => ({ ...line, words: line.words.map((word) => ({ ...word })) })),
      };
      updater(draft);
      draft.status = computeStatus(draft);
      return draft;
    };
    // While a slider handle is being dragged the value changes on every
    // pointer move; those moves are folded into one undo step by ``endDrag``.
    // The updater form matters: pointer moves can arrive faster than React
    // re-renders, and each one must build on the previous move's result.
    if (dragBefore.current) history.replace(apply);
    else history.set(apply);
  };

  const setLineTime = (index: number, key: "start" | "end", value: number | null) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line) return;
      line[key] = value === null ? null : round(value);
    });
  };

  const stampLine = (index: number, key: "start" | "end") => {
    setLineTime(index, key, round(waveRef.current?.getTime() ?? playhead));
  };

  const setWordTime = (lineIndex: number, wordIndex: number, key: "start" | "end", value: number | null) => {
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      const word = line?.words[wordIndex];
      if (!word) return;
      word[key] = value === null ? null : round(value);
      if (key === "start" && value !== null && line.start === null) line.start = round(value);
    });
  };

  /** Stamp the selected word's start at the playhead and close the previous word. */
  const stampWord = (lineIndex: number, wordIndex: number) => {
    const time = round(waveRef.current?.getTime() ?? playhead);
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      if (!line) return;
      if (line.words.length === 0) {
        line.words = splitWords(line.text).map((text) => ({ text, start: null, end: null }));
      }
      const word = line.words[wordIndex];
      if (!word) return;
      word.start = time;
      word.source = "manual";
      const previous = line.words[wordIndex - 1];
      if (previous && previous.start !== null && previous.end === null) previous.end = time;
      if (line.start === null) line.start = time;
    });
    setFocus({ line: lineIndex, word: wordIndex });
    setRange({ line: lineIndex, from: wordIndex, to: wordIndex });
  };

  const closeWord = (lineIndex: number, wordIndex: number) => {
    const time = round(waveRef.current?.getTime() ?? playhead);
    const word = timeline.lines[lineIndex]?.words[wordIndex];
    if (!word) return;
    if (word.start !== null) setWordTime(lineIndex, wordIndex, "end", Math.max(time, word.start));
    else setWordTime(lineIndex, wordIndex, "start", time);
  };

  const splitLineWords = (index: number) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line) return;
      if (line.words.some((word) => word.start !== null) && !window.confirm("该行已有逐字时间，重新拆分将清除这些时间。继续？")) {
        return;
      }
      line.words = splitWords(line.text).map((text) => ({ text, start: null, end: null }));
    });
  };

  const clearLineWords = (index: number) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line) return;
      line.words = [];
    });
  };

  /**
   * Split every line into display words so the per-character stamps have slots.
   * Existing timings are kept unless the user agrees to drop them.
   */
  const splitAllWords = () => {
    const timed = timeline.lines.filter((line) => line.words.some((word) => word.start !== null)).length;
    if (timed && !window.confirm(`${timed} 行已有逐字时间，重新拆分会清除这些时间。继续？`)) return;
    mutate((draft) => {
      draft.lines.forEach((line) => {
        line.words = splitWords(line.text).map((text) => ({ text, start: null, end: null }));
      });
    });
  };

  /** Fill in the gaps: every untimed line is spread evenly across its own window. */
  const distributeAll = () => {
    const ready = timeline.lines.filter((line) => line.start !== null && line.end !== null).length;
    if (!ready) return;
    if (!window.confirm(`按行时长均分会把 ${ready} 行的逐字时间全部重算（标记为插值）。继续？`)) return;
    mutate((draft) => {
      draft.lines.forEach((line) => {
        if (line.start === null || line.end === null) return;
        const words: Word[] = line.words.length
          ? line.words
          : splitWords(line.text).map((text) => ({ text, start: null, end: null }));
        if (!words.length) return;
        const span = Math.max(0, line.end - line.start);
        const weights = words.map((word) => Math.max(word.text.replace(/\s/g, "").length, 1));
        const total = weights.reduce((sum, value) => sum + value, 0);
        let cursor = line.start;
        words.forEach((word, i) => {
          const length = (weights[i] / total) * span;
          word.start = round(cursor);
          word.end = round(cursor + length);
          word.source = "interpolated";
          cursor += length;
        });
        line.words = words;
      });
    });
  };

  const distribute = (index: number) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line || line.start === null || line.end === null) return;
      const words: Word[] = line.words.length
        ? line.words
        : splitWords(line.text).map((text) => ({ text, start: null, end: null }));
      if (!words.length) return;
      const span = Math.max(0, line.end - line.start);
      const weights = words.map((word) => Math.max(word.text.replace(/\s/g, "").length, 1));
      const total = weights.reduce((sum, value) => sum + value, 0);
      let cursor = line.start;
      words.forEach((word, i) => {
        const length = (weights[i] / total) * span;
        word.start = round(cursor);
        word.end = round(cursor + length);
        word.source = "manual";
        cursor += length;
      });
      line.words = words;
    });
  };

  const shiftAll = (delta: number) => {
    mutate((draft) => {
      draft.lines.forEach((line) => {
        if (line.start !== null) line.start = round(Math.max(0, line.start + delta));
        if (line.end !== null) line.end = round(Math.max(0, line.end + delta));
        line.words.forEach((word) => {
          if (word.start !== null) word.start = round(Math.max(0, word.start + delta));
          if (word.end !== null) word.end = round(Math.max(0, word.end + delta));
        });
      });
    });
  };

  /**
   * Move one line — its start, its end and every character — by ``delta``.
   * The shift is cut short at 0 rather than clamping each value on its own:
   * clamping them separately would squash the line against the start of the
   * song and destroy the spacing that was just listened for.
   */
  const shiftLine = (index: number, delta: number) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line) return;
      const times = [line.start, line.end, ...line.words.flatMap((word) => [word.start, word.end])].filter(
        (value): value is number => value !== null && value !== undefined,
      );
      if (!times.length) return;
      const applied = Math.max(delta, -Math.min(...times));
      if (Math.abs(applied) < 1e-9) return;
      const move = (value: number | null): number | null => (value === null ? null : round(value + applied));
      line.start = move(line.start);
      line.end = move(line.end);
      line.words.forEach((word) => {
        word.start = move(word.start);
        word.end = move(word.end);
      });
    });
  };

  /**
   * Spread the selected run evenly across the window it already occupies.
   * Same idea as the whole-line 均分, but only the characters the user picked:
   * a line whose rhythm is right except for three mangled characters can be
   * repaired without re-timing the rest.
   */
  const distributeRange = (lineIndex: number, from: number, to: number) => {
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      if (!line) return;
      const info = barSpans(line);
      if (!info) return;
      const selected = line.words.slice(from, to + 1);
      if (selected.length < 2) return;
      const start = info.spans[from][0];
      const end = info.spans[to][1];
      const span = end - start;
      if (span <= 0) return;
      const weights = selected.map((word) => Math.max(word.text.replace(/\s/g, "").length, 1));
      const total = weights.reduce((sum, value) => sum + value, 0);
      let cursor = start;
      selected.forEach((word, offset) => {
        const length = (weights[offset] / total) * span;
        word.start = round(cursor);
        word.end = round(Math.max(cursor + MIN_WORD_SECONDS, cursor + length));
        // Evenly split by the user, not heard from the recogniser.
        word.source = "manual";
        cursor += length;
      });
    });
  };

  /**
   * Drag one end of the selected run. The characters inside are mapped
   * linearly onto the new window, which keeps the rhythm that was heard and
   * only stretches or squeezes it; they are marked ``manual`` because the
   * numbers are now the user's, not the recogniser's.
   */
  const setRangeTime = (lineIndex: number, from: number, to: number, key: "start" | "end", value: number) => {
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      if (!line) return;
      const info = barSpans(line);
      const limits = rangeLimits(line, from, to);
      if (!info || !limits) return;
      const selected = line.words.slice(from, to + 1);
      if (!selected.length) return;
      const { currentStart, currentEnd } = limits;
      if (currentEnd - currentStart < MIN_WORD_SECONDS) return;
      // The two handles cannot cross: each drag is limited by where the other
      // one currently sits, not by a minimum span invented after the fact.
      const nextStart =
        key === "start"
          ? clamp(value, limits.minStart, Math.min(currentEnd - MIN_WORD_SECONDS, limits.maxStart))
          : currentStart;
      const nextEnd = key === "end" ? clamp(value, nextStart + MIN_WORD_SECONDS, info.bar[1]) : currentEnd;
      const ratio = (nextEnd - nextStart) / (currentEnd - currentStart);
      selected.forEach((word, offset) => {
        const [wordStart, wordEnd] = info.spans[from + offset];
        word.start = round(nextStart + (wordStart - currentStart) * ratio);
        word.end = round(Math.max(word.start + MIN_WORD_SECONDS, nextStart + (wordEnd - currentStart) * ratio));
        markManual(word);
      });
      if (line.start === null) line.start = round(nextStart);
      if (line.end === null) line.end = round(nextEnd);
    });
  };

  /**
   * Slide the selected run without changing its shape. Bounded by the line
   * window and by the neighbouring characters, so a drag stops at them
   * instead of reordering the line.
   */
  const shiftRange = (lineIndex: number, from: number, to: number, delta: number) => {
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      if (!line) return;
      const info = barSpans(line);
      const limits = rangeLimits(line, from, to);
      if (!info || !limits) return;
      const applied = clamp(delta, limits.minShift, limits.maxShift);
      if (Math.abs(applied) < 1e-9) return;
      line.words.slice(from, to + 1).forEach((word, offset) => {
        const [wordStart, wordEnd] = info.spans[from + offset];
        word.start = round(wordStart + applied);
        word.end = round(wordEnd + applied);
        markManual(word);
      });
    });
  };

  const removeLine = (index: number) => {
    mutate((draft) => {
      draft.lines.splice(index, 1);
    });
    setFocus(null);
    setRange(null);
  };

  /**
   * Add a lyric line the source did not have. The row is inserted after the
   * selected line (or appended) with no timing — a line with invented times
   * would be indistinguishable from one the ASR actually heard.
   */
  const insertLine = (after: number | null) => {
    const at = after === null ? timeline.lines.length : after + 1;
    mutate((draft) => {
      draft.lines.splice(at, 0, {
        text: "",
        start: null,
        end: null,
        words: [],
        source: null,
        note: null,
      });
    });
    setFocus({ line: at, word: 0 });
    revealLine(at);
  };

  /**
   * Editing the text of a line invalidates its per-character words: they no
   * longer spell what is displayed. Re-split them, and if the line already has
   * a start/end, spread them inside that window (marked ``interpolated`` so the
   * UI keeps saying the timing is derived, not heard).
   */
  const setLineText = (index: number, text: string) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (!line) return;
      const unchanged = line.words.map((word) => word.text).join("") === text;
      line.text = text;
      if (unchanged) return;
      const tokens: Word[] = splitWords(text).map((token) => ({
        text: token,
        start: null,
        end: null,
        source: "interpolated",
      }));
      if (line.start !== null && line.end !== null && tokens.length) {
        const span = Math.max(0, line.end - line.start);
        const weights = tokens.map((token) => Math.max(token.text.replace(/\s/g, "").length, 1));
        const total = weights.reduce((sum, value) => sum + value, 0);
        let cursor = line.start;
        tokens.forEach((token, i) => {
          const length = (weights[i] / total) * span;
          token.start = round(cursor);
          token.end = round(cursor + length);
          cursor += length;
        });
      }
      line.words = tokens;
      // The text now differs from the lyric the ASR was matched against, so
      // any "this line is suspicious" verdict is stale.
      line.note = null;
      if (line.source && LINE_SOURCE_LABEL[line.source]) line.source = null;
    });
  };

  /**
   * Rename a single character. Its timing slot stays put — the singer still
   * sang *something* there, the text was just misrecognised or mis-split — and
   * the line text follows so the two never disagree.
   */
  const setWordText = (lineIndex: number, wordIndex: number, text: string) => {
    mutate((draft) => {
      const line = draft.lines[lineIndex];
      const word = line?.words[wordIndex];
      if (!line || !word || word.text === text) return;
      const words = line.words;
      const joined = words.map((item) => item.text).join("");
      word.text = text;
      // Only rewrite the line when the words were exactly its text; a hand-made
      // word list may carry different punctuation and must not be overwritten.
      if (joined === line.text) line.text = words.map((item) => item.text).join("");
      line.note = null;
      if (line.source && LINE_SOURCE_LABEL[line.source]) line.source = null;
    });
  };

  const toggleReviewed = (index: number) => {
    mutate((draft) => {
      const line = draft.lines[index];
      if (line) line.reviewed = !line.reviewed;
    });
  };

  const seekTo = (index: number) => {
    const line = timeline.lines[index];
    if (line?.start !== null && line?.start !== undefined) waveRef.current?.seek(line.start);
  };

  /**
   * Scroll a row into view. The row lives inside a scrollable column, so
   * ``scrollIntoView`` is asked for the nearest scroll container only.
   */
  const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
  const revealLine = useCallback((index: number) => {
    const node = rowRefs.current[index];
    if (node) node.scrollIntoView({ block: "center", behavior: "smooth" });
  }, []);

  const jumpToLine = (index: number) => {
    if (index < 0 || index >= timeline.lines.length) return;
    setFocus({ line: index, word: 0 });
    setRange(null);
    seekTo(index);
    revealLine(index);
  };

  /**
   * Pick a character. Shift-click extends the run from its anchor, which is
   * what makes a group of characters draggable as one on the slider.
   */
  const selectWord = (lineIndex: number, wordIndex: number, extend: boolean) => {
    const anchor =
      extend && range && range.line === lineIndex ? range.from : extend && focus?.line === lineIndex ? focus.word : wordIndex;
    setRange({ line: lineIndex, from: anchor, to: wordIndex });
    setFocus({ line: lineIndex, word: wordIndex });
  };

  /** The characters the slider works on: the multi-selection, else the focus. */
  const selectionFor = (index: number): { from: number; to: number } | null => {
    const count = timeline.lines[index]?.words.length ?? 0;
    if (range && range.line === index) {
      return { from: Math.min(range.from, range.to), to: Math.max(range.from, range.to) };
    }
    if (focus?.line === index && focus.word < count) return { from: focus.word, to: focus.word };
    return null;
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const target = event.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return;
    if (!focus) return;
    const { line, word } = focus;
    if (event.key === "Enter" || event.key === "s") {
      event.preventDefault();
      stampWord(line, word);
      if (timeline.lines[line]?.words[word + 1]) selectWord(line, word + 1, false);
    } else if (event.key === "e") {
      event.preventDefault();
      closeWord(line, word);
    } else if (event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault();
      setWordTime(line, word, "start", null);
      setWordTime(line, word, "end", null);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      jumpToLine(Math.min(line + 1, timeline.lines.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      jumpToLine(Math.max(line - 1, 0));
    } else if (event.key === "[") {
      event.preventDefault();
      stampLine(line, "start");
    } else if (event.key === "]") {
      event.preventDefault();
      stampLine(line, "end");
    }
  };

  // The lyrics source's own (studio) timeline, line by line. Display only.
  const referenceFor = (() => {
    const items = project.reference_lines ?? [];
    const byText = new Map<string, ReferenceLine>();
    const byIndex = new Map<number, ReferenceLine>();
    for (const item of items) {
      byIndex.set(item.index, item);
      const key = referenceKey(item.text);
      if (key && !byText.has(key)) byText.set(key, item);
    }
    // Text first: deleting or reordering rows keeps the hint attached to the
    // right lyric. The index is only a fallback for edited lines.
    return (line: LyricLine, index: number): ReferenceLine | null =>
      byText.get(referenceKey(line.text)) ?? byIndex.get(index) ?? null;
  })();

  const status = computeStatus(timeline);
  const flagged = timeline.lines
    .map((line, index) => (needsReview(line) ? index : -1))
    .filter((index) => index >= 0);

  // The line being sung right now, so the list follows the audio. It is the
  // line of the last word that has started — so the line stays current through
  // the gap after its last character, until the next line's first word starts.
  const activeLine = (() => {
    let index = -1;
    let latest = -1;
    timeline.lines.forEach((line, i) => {
      const words = line.words.length ? line.words : [{ text: line.text, start: line.start, end: line.end }];
      words.forEach((word) => {
        if (word.start !== null && word.start !== undefined && word.start <= playhead + 0.001 && word.start >= latest) {
          latest = word.start;
          index = i;
        }
      });
    });
    return index;
  })();

  /**
   * Keep the line being sung in view, the way the karaoke strip above does.
   * Only while playing: a paused playhead that happens to sit on a line must
   * not yank the list around while the user is editing elsewhere. ``followed``
   * remembers the last line scrolled to, so this fires once per line rather
   * than on every timeupdate.
   */
  useEffect(() => {
    if (!playing) {
      followed.current = -1;
      return;
    }
    if (!follow || activeLine < 0 || followed.current === activeLine) return;
    followed.current = activeLine;
    revealLine(activeLine);
  }, [follow, playing, activeLine]);

  const jumpToNextFlagged = () => {
    if (!flagged.length) return;
    const current = focus?.line ?? -1;
    // Skip the line we are already sitting on, so repeated clicks advance
    // even when that line stays flagged (the user has not confirmed it yet).
    const next = flagged.find((index) => index > current) ?? flagged[0];
    jumpToLine(next);
  };

  const jumpToPrevFlagged = () => {
    if (!flagged.length) return;
    const current = focus?.line ?? Number.MAX_SAFE_INTEGER;
    const previous = [...flagged].reverse().find((index) => index < current) ?? flagged[flagged.length - 1];
    jumpToLine(previous);
  };

  if (timeline.lines.length === 0) {
    return (
      <div className="card">
        <h2>时间轴</h2>
        <p className="muted">先在右侧粘贴原始歌词，然后点击「开始逐字打轴」。</p>
      </div>
    );
  }

  return (
    <div className="card" onKeyDown={onKeyDown} tabIndex={0}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>
          时间轴 <span className="badge info">{STATUS_LABEL[status]}</span>
          {flagged.length > 0 && <span className="badge flag">{flagged.length} 行待确认</span>}
        </h2>
        <div className="row small">
          <span className={autosave.status === "error" ? "error-text" : "muted"}>
            {autosave.status === "saving" && "自动保存中…"}
            {autosave.status === "saved" && "已自动保存"}
            {autosave.status === "error" && `保存失败：${autosave.error}`}
            {autosave.status === "idle" && "编辑后自动保存"}
          </span>
          <button onClick={history.undo} disabled={!history.canUndo} title="Ctrl+Z">
            ↶ 撤销
          </button>
          <button onClick={history.redo} disabled={!history.canRedo} title="Ctrl+Shift+Z">
            ↷ 重做
          </button>
        </div>
      </div>

      {timeline.notes && timeline.notes.length > 0 && (
        <ul className="run-notes">
          {timeline.notes.map((note, index) => (
            <li key={index}>{note}</li>
          ))}
        </ul>
      )}

      <Waveform
        ref={waveRef}
        audioUrl={audioUrl ?? api.audioUrl(project.id)}
        timeline={timeline}
        selectedLine={focus?.line ?? null}
        onSeekLine={(index) => setFocus({ line: index, word: 0 })}
        onTimeUpdate={setPlayhead}
        onPlayingChange={setPlaying}
      />

      <div className="toolbar" style={{ marginTop: 12 }}>
        <button onClick={() => shiftAll(-0.1)}>整体 -0.1s</button>
        <button onClick={() => shiftAll(0.1)}>整体 +0.1s</button>
        <span className="sep" />
        <button
          className={follow ? "active-toggle" : ""}
          aria-pressed={follow}
          onClick={() => setFollow((value) => !value)}
          title="播放时自动把正在唱的那一行滚到视野里；关掉后列表不再跟着播放器走"
        >
          {follow ? "✓ 跟随播放" : "跟随播放"}
        </button>
        {/* 逐字打轴: split every line into characters, then fill the line windows. */}
        <button onClick={splitAllWords} title="把每一行歌词拆成逐字槽位（不会改行时间）">
          全部拆成逐字
        </button>
        <button
          onClick={distributeAll}
          disabled={!timeline.lines.some((line) => line.start !== null && line.end !== null)}
          title="按每行的起止时间均分到每个字，标记为插值；之后可用播放器逐字微调"
        >
          逐字打轴（按行均分）
        </button>
        <span className="sep" />
        <button
          onClick={() => insertLine(focus?.line ?? null)}
          title={
            focus
              ? "在选中行下面插入一行歌词（没有时间，需要自己打点）"
              : "在末尾插入一行歌词（没有时间，需要自己打点）"
          }
        >
          + 插入歌词行
        </button>
        {flagged.length > 0 && (
          <>
            <span className="sep" />
            <button className="flag-action" onClick={jumpToPrevFlagged} title="跳到上一个待确认的行">
              ↑ 上一处待确认
            </button>
            <button className="flag-action" onClick={jumpToNextFlagged} title="跳到下一个待确认的行">
              ↓ 下一处待确认（{flagged.length}）
            </button>
          </>
        )}
        <span className="sep" />
        <span className="muted small">
          <span className="kbd">Enter</span> 标记当前字 ·
          <span className="kbd">E</span> 结束当前字 ·
          <span className="kbd">[</span> 行起点 ·
          <span className="kbd">]</span> 行终点 ·
          <span className="kbd">Del</span> 清除 ·
          <span className="kbd">Shift</span>+点击 多选字
        </span>
      </div>

      <div className="timeline-editor" style={{ marginTop: 12 }}>
        {timeline.lines.map((line, index) => (
          <LineRow
            key={index}
            line={line}
            index={index}
            selected={focus?.line === index}
            active={activeLine === index}
            focusWord={focus?.line === index ? focus.word : null}
            selection={selectionFor(index)}
            rowRef={(node) => {
              rowRefs.current[index] = node;
            }}
            onSelect={() => {
              setFocus({ line: index, word: 0 });
              setRange(null);
              seekTo(index);
            }}
            onSeek={() => seekTo(index)}
            onStampLine={(key) => stampLine(index, key)}
            onTimeChange={(key, value) => setLineTime(index, key, value)}
            onShiftLine={(delta) => shiftLine(index, delta)}
            onWordClick={(wordIndex, extend) => {
              selectWord(index, wordIndex, extend);
              // Extending a selection is not a request to listen to that one
              // character, so only a plain click moves the playhead.
              if (extend) return;
              const word = line.words[wordIndex];
              if (word?.start !== null && word?.start !== undefined) waveRef.current?.seek(word.start);
            }}
            onWordStamp={(wordIndex) => stampWord(index, wordIndex)}
            onWordTimeChange={(wordIndex, key, value) => setWordTime(index, wordIndex, key, value)}
            onWordTextChange={(wordIndex, text) => setWordText(index, wordIndex, text)}
            onRangeTimeChange={(from, to, key, value) => setRangeTime(index, from, to, key, value)}
            onRangeShift={(from, to, delta) => shiftRange(index, from, to, delta)}
            onRangeDistribute={(from, to) => distributeRange(index, from, to)}
            onDragStart={beginDrag}
            onDragEnd={endDrag}
            onSplit={() => splitLineWords(index)}
            onClearWords={() => clearLineWords(index)}
            onDistribute={() => distribute(index)}
            onRemove={() => removeLine(index)}
            onInsertBelow={() => insertLine(index)}
            onToggleReviewed={() => toggleReviewed(index)}
            onTextChange={(text) => setLineText(index, text)}
            reference={referenceFor(line, index)}
            onSeekTime={(time) => waveRef.current?.seek(time)}
            onApplyReference={() => {
              const reference = referenceFor(line, index);
              if (!reference || reference.start === null) return;
              mutate((draft) => {
                const target = draft.lines[index];
                if (!target) return;
                target.start = round(reference.start as number);
                target.end = reference.end === null ? null : round(reference.end);
                // The source timing is a hint, not something the ASR heard.
                target.source = "asr-ref";
                target.reviewed = false;
              });
            }}
          />
        ))}
      </div>
    </div>
  );
}

interface LineRowProps {
  line: LyricLine;
  index: number;
  selected: boolean;
  active: boolean;
  focusWord: number | null;
  /** The run of characters the slider acts on, in this line. */
  selection: { from: number; to: number } | null;
  rowRef: (node: HTMLDivElement | null) => void;
  onSelect: () => void;
  onSeek: () => void;
  onStampLine: (key: "start" | "end") => void;
  onTimeChange: (key: "start" | "end", value: number | null) => void;
  onShiftLine: (delta: number) => void;
  onWordClick: (wordIndex: number, extend: boolean) => void;
  onWordStamp: (wordIndex: number) => void;
  onWordTimeChange: (wordIndex: number, key: "start" | "end", value: number | null) => void;
  onWordTextChange: (wordIndex: number, text: string) => void;
  onRangeTimeChange: (from: number, to: number, key: "start" | "end", value: number) => void;
  onRangeShift: (from: number, to: number, delta: number) => void;
  onRangeDistribute: (from: number, to: number) => void;
  onDragStart: () => void;
  onDragEnd: () => void;
  onSplit: () => void;
  onClearWords: () => void;
  onDistribute: () => void;
  onRemove: () => void;
  onInsertBelow: () => void;
  onToggleReviewed: () => void;
  onTextChange: (text: string) => void;
  /** The lyrics source's own timing for this line, shown as a hint only. */
  reference: ReferenceLine | null;
  onSeekTime: (time: number) => void;
  onApplyReference: () => void;
}

const LINE_SOURCE_LABEL: Record<string, { label: string; className: string }> = {
  "asr-partial": { label: "部分吻合", className: "badge flag" },
  "asr-bad": { label: "可能错行", className: "badge err" },
  "asr-ref": { label: "参考定位", className: "badge flag" },
  "word-sync-shifted": { label: "逐字平移", className: "badge flag" },
  "word-sync-missing": { label: "逐字缺行", className: "badge err" },
  unmatched: { label: "未识别到", className: "badge err" },
};

/** Kept, but not flagged: ASR heard singing the original lyrics do not cover. */
const EXTRA_SOURCE_LABEL: Record<string, { label: string; className: string }> = {
  "asr-extra": { label: "现场加唱", className: "badge info" },
  "word-sync": { label: "歌词源逐字", className: "badge info" },
};

/** Lines the aligner or the LLM was unsure about, still waiting for a listen. */
function needsReview(line: LyricLine): boolean {
  return Boolean(line.source && LINE_SOURCE_LABEL[line.source]) && !line.reviewed;
}

/** Fold a lyric line the way the backend matches it to the source timeline. */
function referenceKey(text: string): string {
  return text.replace(/[\s　。、，！？!?,.…~〜ー「」『』（）()\[\]【】・\-—–/]+/g, "").toLowerCase();
}

function TimeField({
  value,
  onChange,
  placeholder,
}: {
  value: number | null;
  onChange: (value: number | null) => void;
  placeholder: string;
}) {
  const [draft, setDraft] = useState(value === null ? "" : formatTime(value));
  useEffect(() => {
    setDraft(value === null ? "" : formatTime(value));
  }, [value]);
  return (
    <input
      className="time-input"
      value={draft}
      placeholder={placeholder}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        const parsed = parseTimeInput(draft);
        if (draft.trim() && parsed === null) {
          setDraft(value === null ? "" : formatTime(value));
          return;
        }
        onChange(parsed);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") (event.target as HTMLInputElement).blur();
      }}
      onClick={(event) => event.stopPropagation()}
    />
  );
}

function WordChip({
  word,
  selected,
  focus,
  onSelect,
  onStamp,
  onTextChange,
}: {
  word: Word;
  selected: boolean;
  focus: boolean;
  onSelect: (extend: boolean) => void;
  onStamp: () => void;
  onTextChange: (text: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(word.text);
  useEffect(() => {
    if (!editing) setDraft(word.text);
  }, [word.text, editing]);

  const startEditing = () => {
    setDraft(word.text);
    setEditing(true);
  };
  const commit = () => {
    setEditing(false);
    if (draft !== word.text) onTextChange(draft);
  };

  const className = `word-chip ${word.start !== null ? "timed" : ""} ${
    word.source === "interpolated" ? "interpolated" : ""
  } ${selected ? "selected" : ""} ${focus ? "focus" : ""}`;
  const title = `${formatTime(word.start)} → ${formatTime(word.end)}${
    word.source === "interpolated"
      ? "（插值，对齐器没有直接给出这个字的时间）"
      : word.source === "asr"
        ? "（识别词流直接给出的时间）"
        : word.source === "aligner"
          ? "（强制对齐给出的时间）"
          : word.source === "merged"
            ? "（LLM 按语意合并出的词，时间是它覆盖的字里第一个的起点到最后一个的终点）"
            : ""
  }`;

  if (editing) {
    return (
      <input
        className="word-chip-input"
        autoFocus
        value={draft}
        size={Math.max(1, Math.min(8, draft.length || 1))}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            commit();
          } else if (event.key === "Escape") {
            event.preventDefault();
            setDraft(word.text);
            setEditing(false);
          }
        }}
      />
    );
  }

  return (
    <span
      className={className}
      title={`${title}（点击选中，Shift+点击选一段，✎ 改字，双击打点）`}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(event.shiftKey);
      }}
      onDoubleClick={(event) => {
        event.stopPropagation();
        onStamp();
      }}
    >
      {word.text || "␣"}
      {focus && (
        <button
          className="ghost word-edit"
          title="修改这个字（保留它的时间）"
          onClick={(event) => {
            event.stopPropagation();
            startEditing();
          }}
        >
          ✎
        </button>
      )}
    </span>
  );
}

/**
 * The line as a progress bar with two draggable ends.
 *
 * The bar spans the line's own window; the filled part is the selected run of
 * characters. Dragging an end stretches or squeezes that run inside the line,
 * dragging the fill moves it. Everything is in seconds, so the handles line up
 * with the times shown in the row above rather than with pixel positions.
 */
function WordSlider({
  line,
  from,
  to,
  onChange,
  onShift,
  onDragStart,
  onDragEnd,
}: {
  line: LyricLine;
  from: number;
  to: number;
  onChange: (key: "start" | "end", value: number) => void;
  onShift: (delta: number) => void;
  onDragStart: () => void;
  onDragEnd: () => void;
}) {
  const barRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{
    kind: "start" | "end" | "move";
    x: number;
    barStart: number;
    barEnd: number;
    start: number;
    end: number;
  } | null>(null);
  const [dragging, setDragging] = useState<"start" | "end" | "move" | null>(null);
  // The callbacks are rebuilt on every render; the drag listeners only need
  // the latest ones, not to be re-attached mid-drag.
  const latest = useRef({ onChange, onShift, onDragEnd });
  latest.current = { onChange, onShift, onDragEnd };

  useEffect(() => {
    if (!dragging) return;
    const onMove = (event: PointerEvent) => {
      const state = drag.current;
      const node = barRef.current;
      if (!state || !node) return;
      const rect = node.getBoundingClientRect();
      if (rect.width <= 0) return;
      const delta = ((event.clientX - state.x) / rect.width) * (state.barEnd - state.barStart);
      if (state.kind === "start") latest.current.onChange("start", state.start + delta);
      else if (state.kind === "end") latest.current.onChange("end", state.end + delta);
      else {
        // Moving the whole run is applied step by step: the clamp at the
        // neighbours would otherwise be re-applied to a cumulative offset.
        latest.current.onShift(delta);
        state.x = event.clientX;
      }
    };
    const onUp = () => {
      drag.current = null;
      setDragging(null);
      latest.current.onDragEnd();
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
  }, [dragging]);

  const info = barSpans(line);
  if (!info) return null;
  const [barStart, barEnd] = info.bar;
  const start = info.spans[from][0];
  const end = info.spans[to][1];
  const count = to - from + 1;
  const label = line.words
    .slice(from, to + 1)
    .map((word) => word.text)
    .join("");
  const pct = (time: number) => clamp(((time - barStart) / (barEnd - barStart)) * 100, 0, 100);

  const begin = (kind: "start" | "end" | "move") => (event: React.PointerEvent) => {
    event.stopPropagation();
    event.preventDefault();
    drag.current = { kind, x: event.clientX, barStart, barEnd, start, end };
    onDragStart();
    setDragging(kind);
  };

  return (
    <div className="word-slider" onClick={(event) => event.stopPropagation()}>
      <div className="word-slider-head small muted">
        <span className="word-slider-label" title={label}>
          {count > 1 ? `已选 ${count} 字「${label}」` : `当前字「${label}」`}
        </span>
        <span className="word-slider-times">
          {formatTime(start)} → {formatTime(end)}
          <span className="muted"> （{(end - start).toFixed(2)}s）</span>
        </span>
      </div>
      <div className={`word-slider-bar ${dragging ? "dragging" : ""}`} ref={barRef}>
        <div className="word-slider-track" />
        <div
          className="word-slider-fill"
          style={{ left: `${pct(start)}%`, width: `${Math.max(0, pct(end) - pct(start))}%` }}
          onPointerDown={begin("move")}
          title="拖动整段，保持每个字的相对位置"
        />
        <button
          type="button"
          className="word-slider-handle start"
          style={{ left: `${pct(start)}%` }}
          onPointerDown={begin("start")}
          title="拖动设置这一段开始的时间"
          aria-label="这一段开始"
        />
        <button
          type="button"
          className="word-slider-handle end"
          style={{ left: `${pct(end)}%` }}
          onPointerDown={begin("end")}
          title="拖动设置这一段结束的时间"
          aria-label="这一段结束"
        />
      </div>
    </div>
  );
}

function LineRow(props: LineRowProps) {
  const { line, index, selected, active, focusWord } = props;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(line.text);
  // Whole-line shift: the preset buttons cover the usual nudges, this covers
  // "this line is 0.43 s late" without twelve clicks.
  const [offset, setOffset] = useState("0.1");
  const offsetValue = (() => {
    const parsed = Number(offset.trim());
    return Number.isFinite(parsed) && parsed !== 0 ? parsed : null;
  })();
  const flag = line.source ? LINE_SOURCE_LABEL[line.source] : undefined;
  const extra = line.source ? EXTRA_SOURCE_LABEL[line.source] : undefined;

  const startEditing = () => {
    setDraft(line.text);
    setEditing(true);
  };
  const commit = () => {
    setEditing(false);
    if (draft !== line.text) props.onTextChange(draft);
  };

  return (
    <div
      ref={props.rowRef}
      className={`line-row ${selected ? "selected" : ""} ${active ? "active" : ""} ${
        needsReview(line) ? "flagged" : ""
      }`}
      onClick={(event) => {
        // A click on a control is not "select this row": the row handler
        // resets the character selection, which would silently retarget the
        // selection buttons (选中均分 / 选中 ±0.05s) at the row's first
        // character instead of the run the user picked.
        const target = event.target as HTMLElement;
        if (target.closest("button, input, .word-slider, .word-chip")) return;
        props.onSelect();
      }}
    >
      <div className="line-head">
        <span className="index">{index + 1}</span>
        {editing ? (
          <input
            type="text"
            className="line-text-input"
            autoFocus
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={commit}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                commit();
              } else if (event.key === "Escape") {
                event.preventDefault();
                setEditing(false);
              }
            }}
            onClick={(event) => event.stopPropagation()}
          />
        ) : (
          <span
            className="text"
            onClick={(event) => {
              // Clicking the selected row's text edits it right away; a click
              // on another row first selects that row.
              if (selected) {
                event.stopPropagation();
                startEditing();
              }
            }}
            onDoubleClick={(event) => {
              event.stopPropagation();
              startEditing();
            }}
            title="点击或双击编辑歌词"
          >
            {line.text || <span className="muted">（空行，点击输入）</span>}
          </span>
        )}
        {!editing && (
          <button
            className="ghost edit-text"
            title="编辑这一行的歌词文本"
            onClick={(event) => {
              event.stopPropagation();
              startEditing();
            }}
          >
            ✎
          </button>
        )}
        {flag && <span className={flag.className}>{flag.label}</span>}
        {extra && <span className={extra.className}>{extra.label}</span>}
        {line.reviewed && <span className="badge ok">已确认</span>}
        {line.note && (
          <span className="line-note small" title={line.note}>
            {line.note}
          </span>
        )}
        {props.reference && props.reference.start !== null && (
          <button
            className="ghost reference-time small"
            title={`歌词源自带时间轴：${formatTime(props.reference.start)} → ${formatTime(
              props.reference.end,
            )}（仅作参考，点击跳转试听）`}
            onClick={(event) => {
              event.stopPropagation();
              if (props.reference?.start !== null && props.reference?.start !== undefined) {
                props.onSeekTime(props.reference.start);
              }
            }}
          >
            参考 {formatTime(props.reference.start, false)}
          </button>
        )}
        <TimeField value={line.start} onChange={(value) => props.onTimeChange("start", value)} placeholder="开始" />
        <TimeField value={line.end} onChange={(value) => props.onTimeChange("end", value)} placeholder="结束" />
        <button
          className="ghost"
          title="在当前播放位置标记行起点"
          onClick={(event) => {
            event.stopPropagation();
            props.onStampLine("start");
          }}
        >
          [
        </button>
        <button
          className="ghost"
          title="在当前播放位置标记行终点"
          onClick={(event) => {
            event.stopPropagation();
            props.onStampLine("end");
          }}
        >
          ]
        </button>
      </div>

      {line.heard && (
        <div className="heard-line small muted" title="ASR 实际听到的内容，可与上面的歌词对照">
          ASR 听到：{line.heard}
        </div>
      )}

      {line.words.length > 0 && (
        <div className="word-strip">
          {line.words.map((word, wordIndex) => (
            <WordChip
              key={wordIndex}
              word={word}
              selected={
                selected &&
                props.selection !== null &&
                wordIndex >= props.selection.from &&
                wordIndex <= props.selection.to
              }
              focus={selected && focusWord === wordIndex}
              onSelect={(extend) => props.onWordClick(wordIndex, extend)}
              onStamp={() => props.onWordStamp(wordIndex)}
              onTextChange={(text) => props.onWordTextChange(wordIndex, text)}
            />
          ))}
        </div>
      )}

      {selected && props.selection && line.words.length > 0 && (
        <>
          <WordSlider
            line={line}
            from={props.selection.from}
            to={props.selection.to}
            onChange={(key, value) => props.onRangeTimeChange(props.selection!.from, props.selection!.to, key, value)}
            onShift={(delta) => props.onRangeShift(props.selection!.from, props.selection!.to, delta)}
            onDragStart={props.onDragStart}
            onDragEnd={props.onDragEnd}
          />
          {/* Acting on the selection without dragging: nudge it, or spread it
              evenly inside the window it already covers. */}
          <div className="word-slider-actions small">
            <button
              onClick={() => props.onRangeShift(props.selection!.from, props.selection!.to, -0.05)}
              title="把选中的这几个字一起往前挪 0.05 秒"
            >
              选中 -0.05s
            </button>
            <button
              onClick={() => props.onRangeShift(props.selection!.from, props.selection!.to, 0.05)}
              title="把选中的这几个字一起往后挪 0.05 秒"
            >
              选中 +0.05s
            </button>
            <button
              onClick={() => props.onRangeDistribute(props.selection!.from, props.selection!.to)}
              disabled={props.selection.from === props.selection.to}
              title="把选中的这几个字在它们当前的起止时间内按字数均分（标记为人工）"
            >
              选中均分
            </button>
          </div>
        </>
      )}

      {selected && (
        <div className="toolbar" style={{ marginTop: 8 }}>
          <button onClick={props.onSeek}>跳到本行</button>
          <span className="sep" />
          {/* Whole-line shift: the line window and every character move together,
              so the spacing that was listened for survives the nudge. */}
          <span className="muted small">整行平移</span>
          <button
            onClick={() => props.onShiftLine(-0.1)}
            disabled={line.start === null && line.end === null && !line.words.some((word) => word.start !== null)}
            title="把本行的起止时间和所有逐字时间一起往前挪 0.1 秒"
          >
            -0.1s
          </button>
          <button
            onClick={() => props.onShiftLine(0.1)}
            title="把本行的起止时间和所有逐字时间一起往后挪 0.1 秒"
          >
            +0.1s
          </button>
          <button onClick={() => props.onShiftLine(-0.01)} title="往前挪 0.01 秒">
            -0.01s
          </button>
          <button onClick={() => props.onShiftLine(0.01)} title="往后挪 0.01 秒">
            +0.01s
          </button>
          <input
            className="time-input shift-input"
            value={offset}
            onChange={(event) => setOffset(event.target.value)}
            placeholder="秒"
            title="自定义平移量（秒，可为负），按回车应用"
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              event.stopPropagation();
              if (event.key !== "Enter" || offsetValue === null) return;
              event.preventDefault();
              props.onShiftLine(offsetValue);
            }}
          />
          <span className="sep" />
          <button onClick={props.onSplit}>按歌词拆分逐字</button>
          <button onClick={props.onDistribute} disabled={line.start === null || line.end === null}>
            按行时长均分
          </button>
          {props.reference && props.reference.start !== null && (
            <button
              onClick={props.onApplyReference}
              title="把歌词源自带的时间写进本行（标记为「参考定位」，之后仍可按实际演唱微调）"
            >
              用参考时间
            </button>
          )}
          {line.words.length > 0 && (
            <button className="ghost" onClick={props.onClearWords}>
              清除逐字
            </button>
          )}
          {focusWord !== null && line.words[focusWord] && (
            <>
              <span className="sep" />
              <span className="muted small">当前字：</span>
              <TimeField
                value={line.words[focusWord].start}
                onChange={(value) => props.onWordTimeChange(focusWord, "start", value)}
                placeholder="字开始"
              />
              <TimeField
                value={line.words[focusWord].end}
                onChange={(value) => props.onWordTimeChange(focusWord, "end", value)}
                placeholder="字结束"
              />
            </>
          )}
          <span className="sep" />
          <button onClick={props.onToggleReviewed}>
            {line.reviewed ? "标记为待确认" : "已人工确认"}
          </button>
          <span className="sep" />
          <button onClick={props.onInsertBelow} title="在这一行下面插入一行歌词（没有时间）">
            + 在下方插入
          </button>
          <button className="danger ghost" onClick={props.onRemove}>
            删除行
          </button>
        </div>
      )}
    </div>
  );
}
