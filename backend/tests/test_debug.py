"""Debug trace: the recorded intermediate results and the endpoint that replays them."""

import pytest

from app import db
from app.schemas import Line, Word
from app.services.providers.llm_aligner import AlignReport, LLMLyricAligner
from app.services.trace import STEP_ORDER, Trace


def _tokens(*items):
    return [Word(text=text, start=start, end=end, source="asr") for text, start, end in items]


def _aligner(reply):
    class FakeClient:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return reply

    aligner = LLMLyricAligner.__new__(LLMLyricAligner)
    aligner.client = FakeClient()
    return aligner


# --------------------------------------------------------------------------
# Only newly sung content is re-lined
# --------------------------------------------------------------------------


def test_split_extras_uses_the_llm_break_and_keeps_the_token_times():
    tokens = _tokens(("ラ", 5.0, 5.2), ("ラ", 5.2, 5.4), ("ラ", 9.0, 9.2), ("ラ", 9.2, 9.4))
    extras = [
        Line(text="ララ", start=5.0, end=5.4, words=tokens[:2], source="asr-extra"),
        Line(text="ララ", start=9.0, end=9.4, words=tokens[2:], source="asr-extra"),
    ]
    aligner = _aligner({"lines": [[1, 2, 3], [4]]})
    report = AlignReport()
    split = aligner.split_extras(extras, tokens, report, [Line(text="あ")])

    assert [line.text for line in split] == ["ラララ", "ラ"]
    assert [(line.start, line.end) for line in split] == [(5.0, 9.2), (9.2, 9.4)]
    assert all(line.source == "asr-extra" for line in split)
    assert report.llm_calls == 1


def test_split_extras_rejects_a_non_partition_and_falls_back():
    tokens = _tokens(("ラ", 5.0, 5.2), ("ラ", 5.2, 5.4))
    extras = [Line(text="ララ", start=5.0, end=5.4, words=tokens, source="asr-extra")]

    # Token 1 twice, token 2 missing: not a partition, so the pause-based
    # grouping is kept rather than losing or duplicating a sung syllable.
    aligner = _aligner({"lines": [[1], [1]]})
    assert aligner.split_extras(extras, tokens, AlignReport(), []) is extras

    aligner = _aligner({"lines": [[1, 2, 3]]})  # out of range
    assert aligner.split_extras(extras, tokens, AlignReport(), []) is extras

    aligner = _aligner({"lines": "nope"})
    assert aligner.split_extras(extras, tokens, AlignReport(), []) is extras


def test_split_extras_never_touches_original_lyric_lines():
    tokens = _tokens(("ラ", 5.0, 5.2), ("ラ", 5.2, 5.4))
    lyrics = [Line(text="あいう", start=0.0, end=4.0), Line(text="えお", start=4.0, end=4.5)]
    extras = [Line(text="ララ", start=5.0, end=5.4, words=tokens, source="asr-extra")]
    aligner = _aligner({"lines": [[1], [2]]})
    split = aligner.split_extras(extras, tokens, AlignReport(), lyrics)
    # The original lines keep the source's own breaks; only the sung-but-not-
    # written content is re-lined, and it comes back as new lines.
    assert [line.text for line in lyrics] == ["あいう", "えお"]
    assert [line.text for line in split] == ["ラ", "ラ"]
    assert all(line.source == "asr-extra" for line in split)


# --------------------------------------------------------------------------
# The recorder and the endpoint
# --------------------------------------------------------------------------


def test_trace_records_steps_and_ignores_failures(fresh_db):
    project_id = fresh_db.new_id("proj_")
    with fresh_db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, title, audio_filename, audio_path, created_at, updated_at) "
            "VALUES (?, '', 'a.mp3', 'a.mp3', ?, ?)",
            (project_id, fresh_db.now_iso(), fresh_db.now_iso()),
        )

    trace = Trace(project_id, "run_test")
    trace.step("metadata", {"title": "x"})
    trace.step("asr_words", {"count": 0, "words": []})

    with fresh_db.connect() as conn:
        steps = fresh_db.fetch_debug_steps(conn, project_id, "run_test")
    assert [step["step"] for step in steps] == ["metadata", "asr_words"]
    assert steps[0]["payload"] == {"title": "x"}


def test_a_disabled_trace_writes_nothing(fresh_db):
    project_id = fresh_db.new_id("proj_")
    with fresh_db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, title, audio_filename, audio_path, created_at, updated_at) "
            "VALUES (?, '', 'a.mp3', 'a.mp3', ?, ?)",
            (project_id, fresh_db.now_iso(), fresh_db.now_iso()),
        )
    Trace(None, None, enabled=False).step("metadata", {})
    with fresh_db.connect() as conn:
        assert fresh_db.fetch_debug_steps(conn, project_id) == []


def test_prune_keeps_the_newest_runs(fresh_db):
    project_id = fresh_db.new_id("proj_")
    with fresh_db.connect() as conn:
        conn.execute(
            "INSERT INTO projects (id, title, audio_filename, audio_path, created_at, updated_at) "
            "VALUES (?, '', 'a.mp3', 'a.mp3', ?, ?)",
            (project_id, fresh_db.now_iso(), fresh_db.now_iso()),
        )
        for index in range(4):
            fresh_db.add_debug_step(conn, project_id, f"run_{index}", "metadata", {"i": index})
        fresh_db.prune_debug_runs(conn, project_id, keep=2)
        remaining = fresh_db.fetch_debug_steps(conn, project_id)
    assert {row["run_id"] for row in remaining} == {"run_2", "run_3"}


def test_debug_endpoint_orders_steps_and_runs(client, project):
    with db.connect() as conn:
        # Written out of order on purpose: the page must render in pipeline order.
        db.add_debug_step(conn, project["id"], "run_a", "timeline", {"lines": []})
        db.add_debug_step(conn, project["id"], "run_a", "metadata", {"title": "t"})
        db.add_debug_step(conn, project["id"], "run_b", "metadata", {"title": "t2"})

    data = client.get(f"/api/projects/{project['id']}/debug").json()
    assert data["order"] == list(STEP_ORDER)
    assert [run["run_id"] for run in data["runs"]] == ["run_b", "run_a"]
    assert [step["step"] for step in data["runs"][1]["steps"]] == ["metadata", "timeline"]

    one = client.get(f"/api/projects/{project['id']}/debug/run_a").json()
    assert len(one["runs"]) == 1
    assert one["runs"][0]["steps"][0]["payload"] == {"title": "t"}


def test_debug_endpoint_404s_for_unknown_project_and_run(client, project):
    assert client.get("/api/projects/nope/debug").status_code == 404
    assert client.get(f"/api/projects/{project['id']}/debug/run_missing").status_code == 404


@pytest.mark.parametrize("step", ["metadata", "timeline"])
def test_step_names_are_in_the_canonical_order(step):
    assert step in STEP_ORDER
