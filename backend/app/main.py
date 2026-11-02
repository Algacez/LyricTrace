"""FastAPI application entry point.

Run with:  uvicorn app.main:app --reload --port 8000   (from backend/)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import db
from .config import settings
from .routers import align, debug, projects, system
from .services.housekeeping import housekeeping
from .services.oneshot import jobs
from .services.pipeline import align_handler, pipeline_handler
from .tasks import TASK_ALIGN, TASK_PIPELINE, runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    runner.register(TASK_PIPELINE, pipeline_handler)
    runner.register(TASK_ALIGN, align_handler)
    runner.start()
    jobs.recover_stale()
    # The caches (job dirs, work dirs, stale uploads) are bounded and swept on
    # a timer; see services/housekeeping.
    housekeeping.start()
    yield
    housekeeping.stop()
    runner.stop()
    jobs.shutdown()


app = FastAPI(
    title="LyricTrace API",
    version="0.1.0",
    description=(
        "Word-level lyric synchronization API.\n\n"
        "Upload audio, extract song metadata from the filename/tags, search LRCLIB, "
        "edit lyrics and word timestamps, then export JSON / LRC / ASS.\n\n"
        "**Timing precision**: the canonical model stores float seconds (millisecond "
        "precision in JSON). LRC and ASS are centisecond formats and are rounded on export."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(system.router)
app.include_router(projects.router)
app.include_router(align.router)
app.include_router(debug.router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "LyricTrace",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/api/health",
    }
