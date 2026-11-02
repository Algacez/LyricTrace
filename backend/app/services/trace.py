"""Intermediate results of an alignment run, kept so the user can inspect them.

The pipeline makes several opaque decisions: which lyrics source answered, what
the ASR heard per window, how the kana DP mapped each line, what the LLM was
asked and what it answered. When a line ends up at the wrong time, none of that
is visible from the final timeline.

``Trace`` is a small recorder the pipeline writes into; each ``step`` is one
JSON-able payload stored in ``debug_steps`` and shown by the debug page. It is
best-effort by design: tracing must never fail a run, so every write is guarded.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import db

logger = logging.getLogger("lyrictrace.trace")

#: Steps in the order they happen. The debug page renders them in this order.
STEP_ORDER = (
    "metadata",
    "lyrics_search",
    "lyrics_selected",
    "language",
    "align_flow",
    "asr_windows",
    "asr_words",
    "asr_retry",
    "reference",
    "word_sync_search",
    "word_sync",
    "line_spans",
    "stray_absorb",
    "llm_verify",
    "llm_extras",
    "extra_retry",
    "extras",
    "char_align",
    "llm_merge",
    "timeline",
)


class Trace:
    """Recorder for one alignment run. ``enabled`` False makes every call a no-op."""

    def __init__(self, project_id: str | None, run_id: str | None = None, *, enabled: bool = True) -> None:
        self.project_id = project_id
        self.run_id = run_id or db.new_id("run_")
        self.enabled = enabled and bool(project_id)
        self._seen: set[str] = set()

    def step(self, name: str, payload: Any) -> None:
        """Record one intermediate result; failures are logged, never raised."""
        if not self.enabled or self.project_id is None:
            return
        self._seen.add(name)
        try:
            with db.connect() as conn:
                db.add_debug_step(conn, self.project_id, self.run_id, name, payload)
        except Exception as exc:  # noqa: BLE001 - debugging must not break a run
            logger.warning("could not record debug step %s: %s", name, exc)

    def finish(self) -> None:
        if not self.enabled or self.project_id is None:
            return
        try:
            with db.connect() as conn:
                db.prune_debug_runs(conn, self.project_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not prune debug runs: %s", exc)


class NullTrace(Trace):
    """A recorder that discards everything (the one-shot API has no project)."""

    def __init__(self) -> None:
        super().__init__(None, None, enabled=False)
