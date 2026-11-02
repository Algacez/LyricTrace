import { useEffect, useRef, useState } from "react";

/**
 * Debounced autosave. ``save`` receives the latest value; returns a status
 * string the UI can show ("saved" | "saving" | "error" | "idle").
 */
export function useAutosave<T>(
  value: T,
  save: (value: T) => Promise<void>,
  options: { delay?: number; enabled?: boolean; signature?: unknown } = {},
) {
  const { delay = 1200, enabled = true, signature } = options;
  const [status, setStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const first = useRef(true);
  const timer = useRef<number | null>(null);
  const latest = useRef(value);
  const saveRef = useRef(save);
  latest.current = value;
  saveRef.current = save;

  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    if (!enabled) return;
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setStatus("saving");
      saveRef
        .current(latest.current)
        .then(() => {
          setStatus("saved");
          setError(null);
        })
        .catch((err: unknown) => {
          setStatus("error");
          setError(err instanceof Error ? err.message : String(err));
        });
    }, delay);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature ?? value, delay, enabled]);

  return { status, error };
}
