import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="lyrictrace-tests-"))
os.environ["LYRICTRACE_DATA_DIR"] = str(_TMP / "data")
os.environ["LYRICTRACE_DB_PATH"] = str(_TMP / "data" / "test.db")
os.environ["LYRICTRACE_AUTO_SEARCH"] = "false"
os.environ["LYRICTRACE_LRCLIB_URL"] = "https://lrclib.invalid"
# No fallback source may reach the network during tests.
os.environ["LYRICTRACE_LRCAPI_URL"] = ""
os.environ["LYRICTRACE_NETEASE_URL"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402

AUDIO_FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp3"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project(client):
    with AUDIO_FIXTURE.open("rb") as handle:
        response = client.post(
            "/api/projects", files={"file": ("Artist - Test Song [Official MV].mp3", handle, "audio/mpeg")}
        )
    assert response.status_code == 201, response.text
    return response.json()


def wait_for_task(client, task_id: str, timeout: float = 10.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in {"succeeded", "failed"}:
            return task
        time.sleep(0.05)
    raise AssertionError("task did not finish in time")


def wait_for_project(client, project_id: str, timeout: float = 10.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/projects/{project_id}").json()
        if data["status"] not in {"queued", "processing"}:
            return data
        time.sleep(0.05)
    raise AssertionError("project did not settle in time")


@pytest.fixture()
def fresh_db():
    db.init_db()
    return db

@pytest.fixture()
def fake_aligner(monkeypatch):
    """Make the forced aligner available and answer with evenly spaced characters.

    Every time in the timeline comes from the aligner now, so a test that stubs
    only the ASR and the LLM aligner fails on availability instead of on the
    thing it means to check. Characters are laid out 0.4 s apart from ``offset``;
    the exact numbers do not matter to the tests that use this, only that they
    exist.
    """
    from app.config import settings
    from app.services import char_align

    def fake_align_text(audio_path, text, language, *, offset=0.0, timeout=None):
        step = 0.4
        chars = [
            {
                "line": 0,
                "c": char,
                "start": offset + index * step,
                "end": offset + (index + 1) * step,
            }
            for index, char in enumerate(text)
        ]
        return {
            "chars": chars,
            "duration": len(text) * step,
            "device": "test",
            "base_offset": offset,
        }

    monkeypatch.setattr(settings, "char_aligner_url", "http://aligner.test")
    monkeypatch.setattr(char_align, "align_text", fake_align_text)
    return fake_align_text
