"""HTTP client for the Qwen3-ASR subtitle service.

The service accepts an audio upload and returns a job; when the job finishes it
exposes a JSON document with ``words[]`` (per character / per word, seconds).
See ``docs/音频转逐字时间戳JSON.md`` in the Qwen-ASR-Subtitle repository.

Its own forced aligner is off by default: on singing it is the wrong tool
(upstream measured 80% of Japanese characters coming back zero-width). With
``align=False`` the service exposes no subtitle document, so this provider then
returns text only and the line mapping uses the recogniser's plain transcription;
turning the aligner on (the per-project ``asr_align`` switch) keeps these tokens
as the per-character source, which ``char_align`` re-slices per line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from ...schemas import Word
from .base import ASRProvider, ASRResult

logger = logging.getLogger("lyrictrace.asr")

POLL_INTERVAL = 2.0


class ASRError(RuntimeError):
    pass


class QwenASRProvider(ASRProvider):
    name = "qwen-asr-http"
    description = "Qwen3-ASR + Qwen3-ForcedAligner via the subtitle service HTTP API"

    def __init__(
        self,
        base_url: str,
        *,
        language: str = "",
        timeout: float = 1800.0,
        align: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.language = language
        self.timeout = timeout
        self.align = align

    def with_align(self, align: bool) -> "QwenASRProvider":
        """The same service with the forced aligner switched on or off."""
        if align == self.align:
            return self
        return QwenASRProvider(
            self.base_url, language=self.language, timeout=self.timeout, align=align
        )

    def available(self) -> bool:
        return bool(self.base_url)

    def unavailable_detail(self) -> str:
        return "Set LYRICTRACE_ASR_URL to the Qwen3-ASR service base URL (e.g. http://asr.local:3090)."

    async def transcribe(self, audio_path: Path, language: str | None = None) -> ASRResult:
        """``language=None`` means "use the configured default"; ``""`` means
        "let the service auto-detect" — the two are deliberately distinct."""
        if not self.available():
            raise ASRError(self.unavailable_detail())

        params = {"align": "true" if self.align else "false"}
        lang = (self.language if language is None else language).strip()
        if lang:
            params["language"] = lang

        # A unique upload name keeps result files from colliding with earlier runs.
        upload_name = f"lyrictrace-{int(time.time())}-{audio_path.name}"
        async with httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            with audio_path.open("rb") as handle:
                response = await client.post(
                    "/api/upload",
                    params=params,
                    files={"files": (upload_name, handle, "application/octet-stream")},
                )
            response.raise_for_status()
            payload = response.json()
            jobs = payload.get("jobs") or []
            if not jobs:
                raise ASRError(f"ASR service rejected the upload: {payload.get('failed') or payload}")
            job_id = jobs[0]["job_id"]

            logger.info("ASR job %s queued for %s", job_id, audio_path.name)
            deadline = time.monotonic() + self.timeout
            while True:
                await asyncio.sleep(POLL_INTERVAL)
                status = (await client.get("/api/status")).json()
                job = next((item for item in status.get("jobs", []) if item.get("id") == job_id), None)
                if job is None:
                    raise ASRError(f"ASR job {job_id} disappeared from the service queue.")
                state = job.get("status")
                if state == "failed":
                    raise ASRError(job.get("error") or "ASR job failed.")
                if state == "done":
                    if job.get("align_error"):
                        raise ASRError(f"ASR alignment failed: {job['align_error']}")
                    break
                if time.monotonic() > deadline:
                    raise ASRError(f"ASR job {job_id} timed out after {self.timeout:.0f}s.")

            if not self.align:
                # No subtitle document exists without the aligner, and there is
                # nothing else here we use: the text is re-derived from the
                # tokens by the caller, and the tokens come from elsewhere.
                return ASRResult(
                    text="",
                    words=[],
                    language=lang or None,
                    model="Qwen3-ASR-1.7B",
                )

            subtitle = (await client.get("/api/subtitle", params={"job_id": job_id, "fmt": "json"})).json()
            document = json.loads(subtitle["text"])

        words = [
            Word(text=str(item.get("text") or ""), start=float(item["start"]), end=float(item["end"]), source="asr")
            for item in document.get("words", [])
            if item.get("text")
        ]
        text = "".join(word.text for word in words)
        return ASRResult(
            text=text,
            words=words,
            cues=document.get("cues", []),
            language=(document.get("meta") or {}).get("language"),
            model="Qwen3-ASR-1.7B",
        )
