import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { cssVar, onThemeChange } from "../theme";
import type { Timeline, Word } from "../types";
import { formatTime } from "../utils";

/** WaveSurfer renders to canvas, so it cannot read CSS variables itself. */
function waveColors() {
  return {
    waveColor: cssVar("--wave", "#a8b5c9"),
    progressColor: cssVar("--wave-progress", "#0284c7"),
    cursorColor: cssVar("--wave-cursor", "#db2777"),
  };
}

export interface WaveformHandle {
  seek: (time: number) => void;
  playPause: () => void;
  getTime: () => number;
  getDuration: () => number;
}

export interface WaveformProps {
  /** The file to draw. ``null`` when the record's upload has been swept. */
  audioUrl: string | null;
  timeline: Timeline | null;
  selectedLine: number | null;
  onSeekLine?: (index: number) => void;
  onTimeUpdate?: (time: number) => void;
  onReady?: (duration: number) => void;
  /** Play/pause state, so the editor can follow the line being sung. */
  onPlayingChange?: (playing: boolean) => void;
}

/**
 * WaveSurfer wrapper: waveform, line markers and a karaoke highlight layer.
 * Imperative handle lets the timeline editor stamp the current playhead.
 */
export const Waveform = forwardRef<WaveformHandle, WaveformProps>(function Waveform(
  { audioUrl, timeline, selectedLine, onSeekLine, onTimeUpdate, onReady, onPlayingChange },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const waveRef = useRef<WaveSurfer | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [error, setError] = useState<string | null>(null);

  // No audio (a job whose upload the sweep already dropped) still shows the
  // timeline, the markers and the editor: a character can be stamped from the
  // keyboard and the times read off the line list. Only playback is gone.
  if (!audioUrl) {
    return (
      <div className="waveform-panel">
        <p className="muted small" style={{ margin: 0 }}>
          这条记录的音频已经不在了，逐字结果仍可查看和修正。
        </p>
      </div>
    );
  }

  useImperativeHandle(
    ref,
    () => ({
      seek: (time: number) => {
        const wave = waveRef.current;
        if (!wave) return;
        wave.setTime(Math.max(0, Math.min(time, wave.getDuration() || time)));
        setCurrentTime(time);
      },
      playPause: () => waveRef.current?.playPause(),
      getTime: () => waveRef.current?.getCurrentTime() ?? 0,
      getDuration: () => waveRef.current?.getDuration() ?? 0,
    }),
    [],
  );

  useEffect(() => {
    if (!containerRef.current) return;
    const wave = WaveSurfer.create({
      container: containerRef.current,
      ...waveColors(),
      cursorWidth: 2,
      height: 96,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      normalize: true,
      url: audioUrl,
    });
    waveRef.current = wave;

    wave.on("ready", () => {
      setDuration(wave.getDuration());
      onReady?.(wave.getDuration());
    });
    wave.on("timeupdate", (time) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });
    wave.on("play", () => {
      setPlaying(true);
      onPlayingChange?.(true);
    });
    wave.on("pause", () => {
      setPlaying(false);
      onPlayingChange?.(false);
    });
    wave.on("finish", () => {
      setPlaying(false);
      onPlayingChange?.(false);
    });
    wave.on("error", (err) => setError(err instanceof Error ? err.message : String(err)));

    return () => {
      wave.destroy();
      waveRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  const seek = (time: number) => {
    const wave = waveRef.current;
    if (!wave) return;
    wave.setTime(Math.max(0, Math.min(time, wave.getDuration() || time)));
    setCurrentTime(time);
    onTimeUpdate?.(time);
  };

  useEffect(() => {
    waveRef.current?.setPlaybackRate(rate);
  }, [rate]);

  // Canvas colours are captured at creation time, so re-apply them on theme switch.
  useEffect(
    () =>
      onThemeChange(() => {
        waveRef.current?.setOptions(waveColors());
      }),
    [],
  );

  // The word being sung right now. The highlight follows the word's own span:
  // it starts at ``start`` and is cleared at ``end``, even when the next
  // character has not begun yet — a breath between characters is silence, and
  // leaving the previous character lit through it says the singer is still on
  // it. The *line* stays on the last word that started, so the strip does not
  // flicker between characters.
  const lastStarted = (() => {
    if (!timeline) return null;
    let found: { line: number; word: number; start: number } | null = null;
    timeline.lines.forEach((_line, lineIndex) => {
      lineWords(timeline, lineIndex).forEach((word, wordIndex) => {
        if (word.start === null || word.start > currentTime + 0.001) return;
        if (found === null || word.start >= found.start) {
          found = { line: lineIndex, word: wordIndex, start: word.start };
        }
      });
    });
    return found as { line: number; word: number; start: number } | null;
  })();

  const activeLineIndex = lastStarted?.line ?? -1;

  return (
    <div className="waveform-panel">
      <div className="waveform" ref={containerRef} />
      {error && <p className="error-text">音频加载失败：{error}</p>}

      <div className="transport">
        <button className="primary" onClick={() => waveRef.current?.playPause()}>
          {playing ? "⏸ 暂停" : "▶ 播放"}
        </button>
        <button onClick={() => seek(0)}>⏮ 开头</button>
        <span className="mono time-readout">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <label className="rate">
          速度
          <select value={rate} onChange={(event) => setRate(Number(event.target.value))}>
            {[0.5, 0.75, 1, 1.25, 1.5].map((value) => (
              <option key={value} value={value}>
                {value}×
              </option>
            ))}
          </select>
        </label>
      </div>

      {timeline && timeline.lines.some((line) => line.start !== null) && duration > 0 && (
        <div className="markers" aria-label="行标记">
          {timeline.lines.map((line, index) =>
            line.start === null ? null : (
              <button
                key={index}
                className={`marker ${index === selectedLine ? "selected" : ""} ${
                  index === activeLineIndex ? "active" : ""
                }`}
                style={{ left: `${(line.start / duration) * 100}%` }}
                title={`${formatTime(line.start)} ${line.text}`}
                onClick={() => {
                  seek(line.start ?? 0);
                  onSeekLine?.(index);
                }}
              />
            ),
          )}
          <div className="marker-playhead" style={{ left: `${(currentTime / duration) * 100}%` }} />
        </div>
      )}

      {timeline && activeLineIndex >= 0 && (
        <div className="karaoke-line">
          {lineWords(timeline, activeLineIndex).map((word, wordIndex) => {
            // Lit only while the character is actually being sung: its own
            // ``end`` clears it, even if the next character starts later, so a
            // breath between characters leaves nothing lit. A character whose
            // span is already behind the playhead stays marked as sung.
            const started = word.start !== null && word.start !== undefined && word.start <= currentTime + 0.001;
            const active = started && (word.end === null || word.end === undefined || currentTime < word.end);
            const sung = started && !active;
            return (
              <span
                key={wordIndex}
                className={`karaoke-word ${active ? "active" : ""} ${sung ? "sung" : ""}`}
              >
                {word.text}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
});

export function lineWords(timeline: Timeline, index: number): Word[] {
  const line = timeline.lines[index];
  if (!line) return [];
  if (line.words.length) return line.words;
  return [{ text: line.text, start: line.start, end: line.end }];
}
