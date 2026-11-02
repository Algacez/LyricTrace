"""Task runner: a single background worker thread with a durable queue.

Tasks are persisted in SQLite, so a crash leaves them marked failed on the
next startup (see ``db.init_db``). One worker keeps the pipeline deterministic
without pulling in Celery/Redis for a local-first tool.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

from . import db
from .config import settings

logger = logging.getLogger("lyrictrace.tasks")

TASK_PIPELINE = "pipeline"
TASK_ALIGN = "align"
TASK_SEARCH = "search"

Handler = Callable[[str, dict[str, Any]], None]


class TaskRunner:
    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._handlers: dict[str, Handler] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def register(self, kind: str, handler: Handler) -> None:
        self._handlers[kind] = handler

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="lyrictrace-worker", daemon=True)
        self._thread.start()
        # Re-queue anything left over from a previous run is intentionally not
        # done: init_db marks those failed so the user can retry explicitly.

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")  # sentinel
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, task_id: str) -> None:
        self._queue.put(task_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if not task_id:
                continue
            self._run(task_id)

    def _run(self, task_id: str) -> None:
        with db.connect() as conn:
            task = db.fetch_task(conn, task_id)
            if task is None or task["status"] != "queued":
                return
            conn.execute(
                "UPDATE tasks SET status='running', started_at=?, progress=0.01 WHERE id=?",
                (db.now_iso(), task_id),
            )
            conn.execute(
                "UPDATE projects SET status='processing', progress=0.01, error=NULL WHERE id=?",
                (task["project_id"],),
            )
            conn.commit()

        handler = self._handlers.get(task["kind"])
        try:
            if handler is None:
                raise RuntimeError(f"No handler registered for task kind '{task['kind']}'")
            handler(task_id, task)
            with db.connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status='succeeded', progress=1, finished_at=?, stage='done' "
                    "WHERE id=? AND status='running'",
                    (db.now_iso(), task_id),
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to the API
            logger.exception("Task %s failed", task_id)
            message = str(exc) or exc.__class__.__name__
            with db.connect() as conn:
                conn.execute(
                    "UPDATE tasks SET status='failed', error=?, finished_at=? WHERE id=?",
                    (message, db.now_iso(), task_id),
                )
                project = db.fetch_project(conn, task["project_id"])
                if project and project["status"] in {"queued", "processing"}:
                    db.update_project(conn, task["project_id"], status="failed", error=message)


runner = TaskRunner()


# --------------------------------------------------------------------------
# Progress helpers used by handlers
# --------------------------------------------------------------------------


def update_progress(
    task_id: str,
    *,
    stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if progress is not None:
        fields.append("progress = ?")
        values.append(max(0.0, min(1.0, progress)))
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    if not fields:
        return
    with db.connect() as conn:
        conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", (*values, task_id))
        task = db.fetch_task(conn, task_id)
        if task:
            project_fields: dict[str, Any] = {}
            if progress is not None:
                project_fields["progress"] = max(0.0, min(1.0, progress))
            if project_fields:
                db.update_project(conn, task["project_id"], **project_fields)


def fail_task(task_id: str, error: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "UPDATE tasks SET status='failed', error=?, finished_at=? WHERE id=?",
            (error, db.now_iso(), task_id),
        )
        task = db.fetch_task(conn, task_id)
        if task:
            db.update_project(conn, task["project_id"], status="failed", error=error)
