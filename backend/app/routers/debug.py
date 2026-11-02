"""Debug view of an alignment run's intermediate results.

    GET /api/projects/{id}/debug              (every kept run, newest first)
    GET /api/projects/{id}/debug/{run_id}     (one run)

The pipeline records each opaque decision — which lyrics source answered, what
the ASR heard per window, how the kana DP mapped each line, what the LLM was
asked and answered, how the extra content was re-lined — into ``debug_steps``.
This endpoint just replays them in a stable order so the UI can show them.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .. import db
from ..schemas import DebugOut, DebugRun, DebugStep
from ..services.trace import STEP_ORDER

router = APIRouter(prefix="/api", tags=["debug"])


def _group_runs(steps: list[dict[str, Any]]) -> list[DebugRun]:
    """Fold the stored rows into runs, newest first, steps in canonical order."""
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in steps:
        by_run.setdefault(row.get("run_id") or "", []).append(row)

    rank = {name: index for index, name in enumerate(STEP_ORDER)}
    runs: list[tuple[int, DebugRun]] = []
    for run_id, rows in by_run.items():
        rows.sort(key=lambda row: (rank.get(row["step"], len(rank)), row["created_at"]))
        # Insertion order, not the millisecond timestamp: several steps of a
        # run (and short runs) share the same stamp.
        latest = max(row.get("seq") or 0 for row in rows)
        runs.append(
            (
                latest,
                DebugRun(
                    run_id=run_id,
                    created_at=min(row["created_at"] for row in rows),
                    finished_at=max(row["created_at"] for row in rows) if len(rows) > 1 else None,
                    steps=[
                        DebugStep(step=row["step"], payload=row.get("payload"), created_at=row["created_at"])
                        for row in rows
                    ],
                ),
            )
        )
    runs.sort(key=lambda item: item[0], reverse=True)
    return [run for _, run in runs]


@router.get("/projects/{project_id}/debug", response_model=DebugOut)
def project_debug(project_id: str) -> DebugOut:
    """Every kept alignment run of a project, newest first."""
    with db.connect() as conn:
        if db.fetch_project(conn, project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
        steps = db.fetch_debug_steps(conn, project_id)
    return DebugOut(project_id=project_id, order=list(STEP_ORDER), runs=_group_runs(steps))


@router.get("/projects/{project_id}/debug/{run_id}", response_model=DebugOut)
def run_debug(project_id: str, run_id: str) -> DebugOut:
    """One run; an unknown id is a 404 rather than an empty payload."""
    with db.connect() as conn:
        if db.fetch_project(conn, project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
        steps = db.fetch_debug_steps(conn, project_id, run_id)
    if not steps:
        raise HTTPException(status_code=404, detail="没有这次运行的调试记录。")
    return DebugOut(project_id=project_id, order=list(STEP_ORDER), runs=_group_runs(steps))
