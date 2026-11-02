"""SQLite persistence layer.

Deliberately small: a handful of tables, hand-written SQL, one connection per
operation. The database is an implementation detail of the local-first app.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    duration REAL,
    audio_filename TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    audio_size INTEGER NOT NULL DEFAULT 0,
    audio_mime TEXT NOT NULL DEFAULT '',
    original_filename TEXT NOT NULL DEFAULT '',
    metadata_source TEXT NOT NULL DEFAULT 'manual',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    search_query TEXT,
    selected_candidate_id TEXT,
    lyrics_source TEXT,
    lyrics_lrclib_id INTEGER,
    lyrics_synced TEXT,
    lyrics_word_sync TEXT,
    lyrics_word_sync_format TEXT,
    lyrics_word_sync_source TEXT,
    lyrics_word_sync_edition TEXT,
    lyrics_language TEXT,
    original_lyrics TEXT NOT NULL DEFAULT '',
    revised_lyrics TEXT NOT NULL DEFAULT '',
    timeline_json TEXT,
    asr_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    lrclib_id INTEGER,
    track_name TEXT NOT NULL DEFAULT '',
    artist_name TEXT NOT NULL DEFAULT '',
    album_name TEXT NOT NULL DEFAULT '',
    duration REAL,
    instrumental INTEGER NOT NULL DEFAULT 0,
    plain_lyrics TEXT,
    synced_lyrics TEXT,
    score REAL NOT NULL DEFAULT 0,
    selected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS debug_steps (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    run_id TEXT,
    step TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_candidates_project ON candidates(project_id);
CREATE INDEX IF NOT EXISTS idx_debug_project ON debug_steps(project_id, created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex
    return f"{prefix}{value}" if prefix else value


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Columns added after the first release; applied to existing databases.
MIGRATIONS = {
    "projects": {
        "asr_json": "TEXT",
        "asr_language": "TEXT",
        "lyrics_lrclib_id": "INTEGER",
        "lyrics_synced": "TEXT",
        "lyrics_word_sync": "TEXT",
        "lyrics_word_sync_format": "TEXT",
        "lyrics_word_sync_source": "TEXT",
        "lyrics_word_sync_edition": "TEXT",
        "lyrics_external_id": "TEXT",
        "lyrics_language": "TEXT",
        "language_origin": "TEXT",
        "separate_vocals": "INTEGER DEFAULT 1",
        # The ASR service's own forced aligner, off unless the project asks for
        # it: per-character times come from the recogniser's own tokens
        # (see char_align).
        "asr_align": "INTEGER DEFAULT 0",
        # Look for 逐字 lyrics in other editions of the same song when the entry
        # the lyrics came from has none. On by default: the studio cut often has
        # no ``yrc`` while the live edition of the same song does.
        "word_sync_search": "INTEGER DEFAULT 1",
        # Whether the cached tokens came from the separated vocals or the raw
        # mix; switching between them is a different audio, so the cache cannot
        # be reused.
        "asr_separated": "INTEGER",
        # Whether this project aligns the whole song with the forced aligner
        # instead of taking the line positions from the recogniser. Off by
        # default: measured against the official 逐字 timings, the ASR-timed flow
        # is the more accurate one (see ``pipeline.align_handler``).
        "hfa_timing": "INTEGER DEFAULT 0",
        # Whether sung content the lyrics do not contain becomes a line of its
        # own (「识别现场加唱」). Off by default: the timeline holds the original
        # lyrics and nothing else. All flows read it — the aligner-first flow
        # also skips the recogniser entirely when it is off, since finding
        # extras is its only job there.
        "detect_extras": "INTEGER DEFAULT 0",
        # Whether the forced aligner takes part in the per-character times. Off by
        # default: the per-character table is the recogniser's own and no
        # aligner request is made. On, every character is also placed by the
        # aligner and the two are fused (see char_align.FUSE_TOLERANCE), which
        # measures better but costs one forced-alignment request per line.
        "hfa_chars": "INTEGER DEFAULT 0",
        # Whether the finished per-character timeline goes to the LLM to be
        # grouped into words (「按语意合并成词」). Off by default: the result
        # keeps one time per character, which is what was measured. On, the
        # LLM only decides where the boundaries are — every unit takes the
        # span of the characters it covers (see llm_merge).
        "llm_merge": "INTEGER DEFAULT 0",
    },
}


def _apply_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db() -> None:
    settings.ensure_dirs()
    with connect() as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)
        # Any task still marked running belongs to a previous process.
        conn.execute(
            "UPDATE tasks SET status='failed', error='Interrupted by server restart', "
            "finished_at=? WHERE status IN ('queued','running')",
            (now_iso(),),
        )
        conn.execute(
            "UPDATE projects SET status='failed', error='Interrupted by server restart' "
            "WHERE status IN ('queued','processing')"
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if "metadata_json" in data:
        try:
            data["metadata"] = json.loads(data.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            data["metadata"] = {}
    if "timeline_json" in data:
        raw = data.pop("timeline_json")
        try:
            data["timeline"] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data["timeline"] = None
    if "instrumental" in data:
        data["instrumental"] = bool(data["instrumental"])
    if "selected" in data:
        data["selected"] = bool(data["selected"])
    return data


def fetch_project(conn: sqlite3.Connection, project_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return row_to_dict(row)


def update_project(conn: sqlite3.Connection, project_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now_iso()
    assignments = ", ".join(f"{key} = ?" for key in fields)
    conn.execute(
        f"UPDATE projects SET {assignments} WHERE id = ?",
        (*fields.values(), project_id),
    )


def create_task(conn: sqlite3.Connection, project_id: str, kind: str) -> dict[str, Any]:
    task_id = new_id("task_")
    conn.execute(
        "INSERT INTO tasks (id, project_id, kind, status, created_at) VALUES (?, ?, ?, 'queued', ?)",
        (task_id, project_id, kind, now_iso()),
    )
    return dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())


def fetch_task(conn: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------
# Debug trace
# --------------------------------------------------------------------------

#: Keep the trace bounded: a long song's ASR payload is the biggest thing we
#: store, and the debug page only ever needs the last few runs.
MAX_DEBUG_RUNS = 10


def add_debug_step(
    conn: sqlite3.Connection,
    project_id: str,
    run_id: str,
    step: str,
    payload: Any,
) -> None:
    """Append one intermediate result to a run's trace."""
    conn.execute(
        "INSERT INTO debug_steps (id, project_id, run_id, step, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            new_id("dbg_"),
            project_id,
            run_id,
            step,
            json.dumps(payload, ensure_ascii=False, default=str),
            now_iso(),
        ),
    )


def fetch_debug_steps(
    conn: sqlite3.Connection, project_id: str, run_id: str | None = None
) -> list[dict[str, Any]]:
    if run_id:
        rows = conn.execute(
            "SELECT rowid AS seq, * FROM debug_steps WHERE project_id = ? AND run_id = ? "
            "ORDER BY created_at, rowid",
            (project_id, run_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT rowid AS seq, * FROM debug_steps WHERE project_id = ? ORDER BY created_at, rowid",
            (project_id,),
        ).fetchall()
    steps: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except json.JSONDecodeError:
            pass
        steps.append(item)
    return steps


def prune_debug_runs(conn: sqlite3.Connection, project_id: str, keep: int = MAX_DEBUG_RUNS) -> None:
    """Drop the oldest runs so a project does not accumulate traces forever."""
    rows = conn.execute(
        "SELECT run_id, MAX(rowid) AS latest FROM debug_steps WHERE project_id = ? "
        "GROUP BY run_id ORDER BY latest DESC",
        (project_id,),
    ).fetchall()
    stale = [row["run_id"] for row in rows[keep:]]
    for run_id in stale:
        conn.execute(
            "DELETE FROM debug_steps WHERE project_id = ? AND run_id = ?", (project_id, run_id)
        )
