"""Vocal separation through the BSRoformer.cpp HTTP service.

The service runs a job and exposes the resulting stems. We only need the
vocals stem: feeding the full mix to the ASR produces garbled, zero-timed
output, while the isolated vocals transcribe cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from .base import SeparatorProvider

logger = logging.getLogger("lyrictrace.separator")

POLL_INTERVAL = 2.0
STALL_SEC = 420.0
"""Give up when the service reports no progress for this long.

The GPU service occasionally wedges mid-job (progress frozen, GPU idle) and
then holds the queue forever; waiting out the full 30 min timeout is useless.
"""


class SeparationError(RuntimeError):
    pass


class BsRoformerSeparator(SeparatorProvider):
    name = "bs-roformer"
    description = "Mel-band RoFormer vocal separation (BSRoformer.cpp service)"

    def __init__(self, base_url: str, *, model: str = "", timeout: float = 1800.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.base_url)

    def unavailable_detail(self) -> str:
        return "Set LYRICTRACE_SEPARATOR_URL to the separation service base URL (e.g. http://separator.local:3095)."

    async def separate(self, audio_path: Path, out_dir: Path) -> dict[str, Path]:
        if not self.available():
            raise SeparationError(self.unavailable_detail())
        out_dir.mkdir(parents=True, exist_ok=True)

        data = {"output_format": "wav"}
        if self.model:
            data["model"] = self.model
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=15.0)
        ) as client:
            with audio_path.open("rb") as handle:
                response = await client.post(
                    "/v1/separate",
                    data=data,
                    files={"file": (audio_path.name, handle, "application/octet-stream")},
                )
            response.raise_for_status()
            job_id = response.json().get("id")
            if not job_id:
                raise SeparationError(f"Separation service returned no job id: {response.text[:200]}")

            logger.info("separation job %s queued for %s", job_id, audio_path.name)
            deadline = time.monotonic() + self.timeout
            last_progress: float | None = None
            last_change = time.monotonic()
            while True:
                await asyncio.sleep(POLL_INTERVAL)
                response = await client.get(f"/v1/jobs/{job_id}")
                if response.status_code == 404:
                    # The service dropped the job (restarted, or someone deleted
                    # it). Polling on would just burn the whole timeout.
                    raise SeparationError(f"人声分离任务 {job_id} 在服务端已不存在（可能被重启或删除）。")
                response.raise_for_status()
                job = response.json()
                state = job.get("status")
                if state == "failed":
                    raise SeparationError(job.get("error") or "Separation job failed.")
                if state in {"completed", "done"}:
                    break
                progress = job.get("progress")
                if progress != last_progress:
                    last_progress = progress
                    last_change = time.monotonic()
                elif time.monotonic() - last_change > STALL_SEC:
                    raise SeparationError(
                        f"人声分离任务 {job_id} 卡在 {progress}% 超过 {STALL_SEC:.0f}s，已放弃。"
                    )
                if time.monotonic() > deadline:
                    raise SeparationError(f"Separation job {job_id} timed out after {self.timeout:.0f}s.")

            stems = {stem["name"]: stem for stem in job.get("stems", []) if stem.get("name")}
            if "vocals" not in stems:
                raise SeparationError("Separation produced no vocals stem.")
            target = out_dir / "vocals.wav"
            async with client.stream("GET", stems["vocals"]["url"]) as stream:
                stream.raise_for_status()
                with target.open("wb") as handle:
                    async for chunk in stream.aiter_bytes(1024 * 1024):
                        handle.write(chunk)
        logger.info("vocals stem saved to %s", target)
        return {"vocals": target}


def to_asr_wav(source: Path, target: Path, *, sample_rate: int = 16000) -> Path:
    """Downmix to mono 16 kHz WAV, which is what the ASR service expects."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SeparationError("ffmpeg is required to prepare audio for ASR.")
    command = [
        ffmpeg, "-y", "-v", "error", "-i", str(source),
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        raise SeparationError(f"ffmpeg failed: {completed.stderr.strip()[:400]}")
    return target
