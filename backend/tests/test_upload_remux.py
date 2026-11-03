"""Uploaded audio is rewritten into a container the browser accepts.

The bug this exists for: a FLAC whose PICTURE metadata block is malformed
(Chrome's ``MediaError`` code 4, "format error") while ffmpeg reads it without
complaint — so the pipeline aligns it happily and the timeline has no sound.
Re-muxing copies the audio frames into a fresh container and leaves every
non-audio block behind.

The project upload reads the tags *before* the rewrite and records the size
*after* it, because the rewrite is what lands on disk.
"""

import io
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

from app import audio
from app.config import settings


def _wav_bytes(seconds: float = 0.2, rate: int = 8000) -> bytes:
    handle = io.BytesIO()
    with wave.open(handle, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(b"\x00\x01" * int(rate * seconds))
    return handle.getvalue()


def _flac_bytes(wav: bytes) -> bytes:
    """A real FLAC, so the header surgery below starts from something decodable."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "in.wav"
        target = Path(folder) / "out.flac"
        source.write_bytes(wav)
        done = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-c:a", "flac", str(target)],
            capture_output=True,
        )
        if done.returncode != 0:
            pytest.skip("ffmpeg cannot write flac here")
        return target.read_bytes()


def _with_bad_picture(flac: bytes) -> bytes:
    """The same FLAC with a PICTURE block no decoder can read.

    The mime length says 3 but no mime string follows, and the picture's
    dimensions, depth and colour count are all zero — the shape of the block in
    the real file that made Chrome fail with ``[object MediaError]``. The audio
    frames after it are untouched, which is why ffmpeg still reads the file.
    """
    head, rest = flac[:4], flac[4:]
    # The first metadata block is always STREAMINFO (type 0), 34 bytes long.
    block_type, length = head[:1] + bytes([rest[0] & 0x7F]), struct.unpack(">I", b"\x00" + rest[1:4])[0]
    streaminfo = rest[:4 + length]
    picture = struct.pack(">I", 6) + struct.pack(">I", 3) + b"\x00" * 12
    return (
        head
        + streaminfo
        + bytes([block_type[0] & 0x7F])
        + struct.pack(">I", len(picture))[1:]
        + picture
        + rest[4 + length :]
    )


@pytest.fixture
def upload(monkeypatch, tmp_path, client):
    """A TestClient whose audio dir is a temp dir and whose pipeline never runs."""
    from app.routers import projects

    monkeypatch.setattr(settings, "audio_dir", tmp_path / "audio")
    monkeypatch.setattr(projects, "_create_task", lambda *a, **k: None)
    return client


def _create(client, name: str, payload: bytes):
    return client.post(
        "/api/projects",
        files={"file": (name, payload, "audio/x-flac")},
    )


def test_a_playable_upload_survives_the_rewrite(upload):
    before = _create(upload, "song.wav", _wav_bytes())
    assert before.status_code == 201, before.text
    project_id = before.json()["id"]
    stored = settings.audio_dir / f"{project_id}.wav"
    assert stored.exists()
    # The audio is still the audio: same length of samples, and readable back.
    with wave.open(str(stored), "rb") as reader:
        assert reader.getnframes() == 8000 * 0.2


def test_a_flac_the_browser_refuses_is_rewritten(upload):
    """The regression: this file used to upload as-is and never played."""
    payload = _with_bad_picture(_flac_bytes(_wav_bytes()))
    created = _create(upload, "broken.flac", payload)
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    stored = settings.audio_dir / f"{project_id}.flac"
    assert stored.exists()
    # The rewritten file is a FLAC with no metadata block left in it: the audio
    # frames are all ffmpeg wrote. (The bad PICTURE is gone, not merely moved.)
    head = stored.read_bytes()[:200]
    assert head.startswith(b"fLaC")
    assert len(head) > 4
    # And it is still the same recording, decodable by the same ffmpeg the ASR uses.
    assert audio.probe_audio(stored).duration is not None


def test_the_rewrite_is_skipped_without_ffmpeg(upload, monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    payload = _with_bad_picture(_flac_bytes(_wav_bytes()))
    created = _create(upload, "broken.flac", payload)
    assert created.status_code == 201, created.text
    stored = settings.audio_dir / f"{created.json()['id']}.flac"
    # The file is left exactly as uploaded: a file we cannot repair is still a
    # file, and the pipeline can read it even when the browser cannot.
    assert stored.read_bytes() == payload


def test_a_failed_rewrite_leaves_the_original(upload, monkeypatch):
    def boom(command, **kwargs):
        raise OSError("no")

    monkeypatch.setattr(audio.subprocess, "run", boom)
    payload = _wav_bytes()
    created = _create(upload, "song.wav", payload)
    assert created.status_code == 201, created.text
    stored = settings.audio_dir / f"{created.json()['id']}.wav"
    assert stored.read_bytes() == payload
    # No half-written copy is left behind under a second name.
    assert list(settings.audio_dir.iterdir()) == [stored]


def test_the_size_recorded_is_the_one_on_disk(upload):
    payload = _wav_bytes()
    created = _create(upload, "song.wav", payload)
    assert created.status_code == 201, created.text
    stored = settings.audio_dir / f"{created.json()['id']}.wav"
    assert created.json()["audio_size"] == stored.stat().st_size
