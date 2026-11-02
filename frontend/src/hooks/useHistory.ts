import { useCallback, useEffect, useRef, useState } from "react";

export interface HistoryState<T> {
  present: T;
  past: T[];
  future: T[];
}

/**
 * Undo/redo store. ``set`` records the previous value; ``replace`` updates
 * without recording (used when the server echoes back the saved value).
 */
export function useHistory<T>(initial: T) {
  const [state, setState] = useState<HistoryState<T>>({ present: initial, past: [], future: [] });
  const lastExternal = useRef<T>(initial);

  const reset = useCallback((value: T) => {
    lastExternal.current = value;
    setState({ present: value, past: [], future: [] });
  }, []);

  const set = useCallback((updater: T | ((prev: T) => T)) => {
    setState((current) => {
      const next = typeof updater === "function" ? (updater as (prev: T) => T)(current.present) : updater;
      if (Object.is(next, current.present)) return current;
      return {
        present: next,
        past: [...current.past.slice(-99), current.present],
        future: [],
      };
    });
  }, []);

  const replace = useCallback((value: T | ((prev: T) => T)) => {
    setState((current) => {
      const next = typeof value === "function" ? (value as (prev: T) => T)(current.present) : value;
      if (!Object.is(next, current.present)) lastExternal.current = next;
      return { ...current, present: next };
    });
  }, []);

  /**
   * Record ``before`` as the state to undo to, keeping the current value.
   * A drag edits on every pointer move; this is how those moves collapse into
   * one undo step: live edits ``replace``, then the value from before the drag
   * is committed once at the end.
   */
  const commit = useCallback((before: T) => {
    setState((current) => {
      if (Object.is(before, current.present)) return current;
      return { present: current.present, past: [...current.past.slice(-99), before], future: [] };
    });
  }, []);

  const undo = useCallback(() => {
    setState((current) => {
      if (current.past.length === 0) return current;
      const previous = current.past[current.past.length - 1];
      return {
        present: previous,
        past: current.past.slice(0, -1),
        future: [current.present, ...current.future],
      };
    });
  }, []);

  const redo = useCallback(() => {
    setState((current) => {
      if (current.future.length === 0) return current;
      const [next, ...rest] = current.future;
      return { present: next, past: [...current.past, current.present], future: rest };
    });
  }, []);

  const canUndo = state.past.length > 0;
  const canRedo = state.future.length > 0;

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || target?.isContentEditable) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) redo();
        else undo();
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        redo();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo, redo]);

  return { value: state.present, set, replace, commit, reset, undo, redo, canUndo, canRedo };
}
