import type { Timeline, TimelineStatus } from "./types";

export function formatTime(seconds: number | null | undefined, withMs = true): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return "--:--";
  const clamped = Math.max(0, seconds);
  const minutes = Math.floor(clamped / 60);
  const secs = Math.floor(clamped % 60);
  const ms = Math.round((clamped - Math.floor(clamped)) * 1000);
  const base = `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  if (!withMs) return base;
  return `${base}.${String(ms).padStart(3, "0")}`;
}

export function parseTimeInput(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Number(trimmed);
  const match = trimmed.match(/^(\d+):(\d+(?:\.\d+)?)$/);
  if (match) return Number(match[1]) * 60 + Number(match[2]);
  return null;
}

export function round(value: number, digits = 3): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/** Split a lyric line into display words: CJK per character, latin per word. */
const TOKEN_RE = /[\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]|[^\s\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]+|\s+/g;

export function splitWords(text: string): string[] {
  const parts = text.match(TOKEN_RE);
  if (!parts) return [];
  const words: string[] = [];
  for (const part of parts) {
    if (/^\s+$/.test(part)) {
      // Trailing whitespace belongs to the preceding token so re-joining is lossless.
      if (words.length) words[words.length - 1] += part;
      else words.push(part);
    } else {
      words.push(part);
    }
  }
  return words.filter((word) => word.length > 0);
}

export function computeStatus(timeline: Timeline | null): TimelineStatus {
  if (!timeline || timeline.lines.length === 0) return "none";
  const timed = timeline.lines.filter((line) => line.start !== null);
  if (timed.length === 0) return "untimed";
  const worded = timed.filter((line) => line.words.some((word) => word.start !== null));
  if (worded.length === 0) return "line_only";
  const complete = timed.every(
    (line) => line.words.length > 0 && line.words.every((word) => word.start !== null && word.end !== null),
  );
  return complete && worded.length === timed.length ? "complete" : "partial";
}

export const STATUS_LABEL: Record<TimelineStatus, string> = {
  none: "无时间轴",
  untimed: "未打轴",
  line_only: "仅行级时间",
  partial: "部分逐字",
  complete: "逐字完整",
};
