from app.schemas import Line, Timeline, Word


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_upload_rejects_non_audio(client):
    response = client.post("/api/projects", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    response = client.post("/api/projects", files={"file": ("empty.mp3", b"", "audio/mpeg")})
    assert response.status_code == 400


def test_pipeline_extracts_metadata_and_marks_untimed(client, project):
    from tests.conftest import wait_for_project, wait_for_task

    tasks = client.get(f"/api/projects/{project['id']}/tasks").json()
    task = wait_for_task(client, tasks[0]["id"])
    assert task["status"] == "succeeded", task
    assert task["progress"] == 1

    data = wait_for_project(client, project["id"])
    # Embedded ID3 tags win over the filename.
    assert data["title"] == "Test Song"
    assert data["artist"] == "Test Artist"
    assert data["duration"] and data["duration"] > 2
    # LRCLIB was disabled in tests, so there must be no invented timeline.
    assert data["timeline"] is None
    assert data["timeline_status"] == "none"
    assert data["status"] == "needs_review"


def test_audio_streams(client, project):
    response = client.get(f"/api/projects/{project['id']}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/")
    assert len(response.content) > 1000


def test_manual_lyrics_creates_untimed_timeline(client, project):
    response = client.put(
        f"/api/projects/{project['id']}/lyrics",
        json={"original_lyrics": "line one\nline two", "revised_lyrics": "line one\nline two"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["timeline_status"] == "untimed"
    assert [line["text"] for line in body["timeline"]["lines"]] == ["line one", "line two"]
    assert all(line["start"] is None for line in body["timeline"]["lines"])


def test_timeline_update_round_trip(client, project):
    timeline = Timeline(
        lines=[
            Line(
                text="Hello",
                start=1.0,
                end=2.0,
                words=[Word(text="Hel", start=1.0, end=1.5), Word(text="lo", start=1.5, end=2.0)],
            )
        ]
    )
    response = client.put(f"/api/projects/{project['id']}/timeline", json={"timeline": timeline.model_dump()})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["timeline_status"] == "complete"
    assert body["status"] == "ready"
    assert body["revised_lyrics"] == "Hello"
    assert body["timeline"]["lines"][0]["words"][0]["start"] == 1.0


def test_timeline_validation_rejects_bad_order(client, project):
    bad = Timeline(
        lines=[
            Line(
                text="x",
                start=1.0,
                words=[Word(text="a", start=2.0, end=2.5), Word(text="b", start=1.0, end=2.0)],
            )
        ]
    )
    response = client.put(f"/api/projects/{project['id']}/timeline", json={"timeline": bad.model_dump()})
    assert response.status_code == 422


def test_timeline_validation_rejects_negative(client, project):
    bad = Timeline(lines=[Line(text="x", start=-1.0)])
    response = client.put(f"/api/projects/{project['id']}/timeline", json={"timeline": bad.model_dump()})
    assert response.status_code == 422


def test_export_json_lrc_ass(client, project):
    timeline = Timeline(
        lines=[
            Line(
                text="Hello",
                start=1.0,
                end=2.0,
                words=[Word(text="Hello", start=1.0, end=2.0)],
            )
        ]
    )
    client.put(f"/api/projects/{project['id']}/timeline", json={"timeline": timeline.model_dump()})

    json_export = client.get(f"/api/projects/{project['id']}/export?format=json")
    assert json_export.status_code == 200
    payload = json_export.json()
    assert payload["lines"][0]["words"][0]["start"] == 1.0
    assert payload["precision"].startswith("milliseconds")

    lrc_export = client.get(f"/api/projects/{project['id']}/export?format=lrc")
    assert lrc_export.status_code == 200
    assert "[00:01.00]" in lrc_export.text
    assert "<00:01.00>Hello" in lrc_export.text

    ass_export = client.get(f"/api/projects/{project['id']}/export?format=ass")
    assert ass_export.status_code == 200
    assert "[Events]" in ass_export.text
    assert "Dialogue:" in ass_export.text


def test_export_without_timeline_conflicts(client):
    with open("tests/fixtures/sample.mp3", "rb") as handle:
        created = client.post("/api/projects", files={"file": ("fresh.mp3", handle, "audio/mpeg")}).json()
    response = client.get(f"/api/projects/{created['id']}/export?format=lrc")
    assert response.status_code == 409


def test_align_accepts_a_project_without_lyrics(client, project):
    """No lyrics is not an error: the ASR transcript becomes the text then."""
    response = client.post(f"/api/projects/{project['id']}/align", json={})
    assert response.status_code == 202, response.text
    assert response.json()["kind"] == "align"


def test_formats_and_providers(client):
    formats = client.get("/api/formats").json()
    assert {item["format"] for item in formats} == {"json", "lrc", "ass"}

    providers = client.get("/api/providers").json()["providers"]
    by_name = {item["name"]: item for item in providers}
    assert by_name["lrclib"]["available"] is True
    assert by_name["filename"]["available"] is True
    assert by_name["qwen-asr-http"]["available"] is False
    assert by_name["llm-aligner"]["available"] is False
    assert by_name["bs-roformer"]["available"] is False


def test_patch_project_marks_metadata_manual(client, project):
    response = client.patch(f"/api/projects/{project['id']}", json={"title": "Renamed"})
    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"
    assert response.json()["metadata_source"] == "manual"


def test_separation_is_a_project_setting_and_drops_the_cached_asr(client, project, fresh_db):
    """Recognising the raw mix is a different audio, so old tokens must go."""
    pid = project["id"]
    with fresh_db.connect() as conn:
        fresh_db.update_project(conn, pid, asr_json='{"words":[{"text":"あ","start":0,"end":1}]}')
        conn.commit()

    # Separation is on unless turned off: a loud backing track is otherwise
    # misheard as words.
    assert project["separate_vocals"] is True
    assert project["asr_align"] is False
    response = client.patch(f"/api/projects/{pid}", json={"separate_vocals": False})
    assert response.status_code == 200
    assert response.json()["separate_vocals"] is False
    assert client.get(f"/api/projects/{pid}/asr").json()["words"] == []

    # Re-saving the same value keeps the cache; it describes the same audio.
    with fresh_db.connect() as conn:
        fresh_db.update_project(conn, pid, asr_json='{"words":[{"text":"あ","start":0,"end":1}]}')
        conn.commit()
    client.patch(f"/api/projects/{pid}", json={"separate_vocals": False})
    assert len(client.get(f"/api/projects/{pid}/asr").json()["words"]) == 1


def test_asr_align_is_a_project_setting_that_keeps_the_cache(client, project, fresh_db):
    """It only decides which recogniser runs, so the word stream stays usable."""
    pid = project["id"]
    with fresh_db.connect() as conn:
        fresh_db.update_project(conn, pid, asr_json='{"words":[{"text":"あ","start":0,"end":1}]}')
        conn.commit()
    response = client.patch(f"/api/projects/{pid}", json={"asr_align": True})
    assert response.status_code == 200
    assert response.json()["asr_align"] is True
    assert len(client.get(f"/api/projects/{pid}/asr").json()["words"]) == 1


def test_word_sync_search_is_a_project_setting(client, project, fresh_db):
    """Off means "only the entry the lyrics came from", so a foreign payload goes."""
    pid = project["id"]
    # Off unless turned on: the search costs a second chain of source queries
    # and only pays off when the studio entry has no 逐字 data.
    assert project["word_sync_search"] is False

    with fresh_db.connect() as conn:
        fresh_db.update_project(
            conn,
            pid,
            lyrics_word_sync="[1000,500](1000,500,0)あ\n",
            lyrics_word_sync_format="yrc",
            lyrics_word_sync_source="netease",
            lyrics_word_sync_edition="テスト (Live) — 歌手",
        )
        conn.commit()

    response = client.patch(f"/api/projects/{pid}", json={"word_sync_search": False})
    assert response.status_code == 200
    assert response.json()["word_sync_search"] is False
    # The payload came from another edition; with the search off nothing would
    # replace it, so keeping it would silently keep using that edition.
    assert response.json()["lyrics_word_sync"] is False
    assert response.json()["lyrics_word_sync_edition"] is None

    with fresh_db.connect() as conn:
        fresh_db.update_project(
            conn,
            pid,
            lyrics_word_sync="[1000,500](1000,500,0)あ\n",
            lyrics_word_sync_format="yrc",
            lyrics_word_sync_edition=None,
        )
        conn.commit()
    # The lyrics' own entry supplied this one: turning the search off keeps it.
    assert client.patch(f"/api/projects/{pid}", json={"word_sync_search": False}).json()[
        "lyrics_word_sync"
    ] is True


def test_align_request_can_switch_the_other_edition_search(client, project):
    pid = project["id"]
    response = client.post(f"/api/projects/{pid}/align", json={"word_sync_search": False})
    assert response.status_code == 202, response.text
    assert client.get(f"/api/projects/{pid}").json()["word_sync_search"] is False


def test_align_request_can_switch_separation(client, project):
    pid = project["id"]
    response = client.post(f"/api/projects/{pid}/align", json={"separate": False})
    assert response.status_code == 202, response.text
    assert client.get(f"/api/projects/{pid}").json()["separate_vocals"] is False


def test_list_and_delete_project(client):
    with open("tests/fixtures/sample.mp3", "rb") as handle:
        created = client.post("/api/projects", files={"file": ("deletable.mp3", handle, "audio/mpeg")}).json()
    listed = client.get("/api/projects").json()
    assert listed["total"] >= 1
    assert any(item["id"] == created["id"] for item in listed["items"])

    assert client.delete(f"/api/projects/{created['id']}").status_code == 204
    assert client.get(f"/api/projects/{created['id']}").status_code == 404


def test_batch_delete_removes_every_project_and_what_it_owns(client):
    """The home page's batch delete: rows, audio and ``work/`` all go."""
    from tests.conftest import wait_for_project

    from app.config import settings

    ids = []
    for name in ("batch-a.mp3", "batch-b.mp3"):
        with open("tests/fixtures/sample.mp3", "rb") as handle:
            created = client.post("/api/projects", files={"file": (name, handle, "audio/mpeg")}).json()
        wait_for_project(client, created["id"])
        ids.append(created["id"])
        # Something the run would have written, so the sweep has a target.
        work = settings.data_dir / "work" / created["id"]
        work.mkdir(parents=True, exist_ok=True)
        (work / "vocals.wav").write_bytes(b"x")

    response = client.post("/api/projects/delete", json={"ids": ids})
    assert response.status_code == 204, response.text
    for project_id in ids:
        assert client.get(f"/api/projects/{project_id}").status_code == 404
        assert not (settings.data_dir / "work" / project_id).exists()
    listed = client.get("/api/projects").json()
    assert not [item for item in listed["items"] if item["id"] in ids]

    # An unknown id in the batch is a 404, not a silent skip.
    assert client.post("/api/projects/delete", json={"ids": ["prj_missing"]}).status_code == 404
    # An empty batch is rejected by the schema, not by the handler.
    assert client.post("/api/projects/delete", json={"ids": []}).status_code == 422


def test_batch_delete_refuses_a_project_that_is_still_running(client):
    """A worker is writing into ``work/`` right now; the whole batch waits."""
    with open("tests/fixtures/sample.mp3", "rb") as handle:
        created = client.post("/api/projects", files={"file": ("busy.mp3", handle, "audio/mpeg")}).json()

    # Stop the pipeline mid-run rather than mock it: the status is what counts.
    from app import db

    with db.connect() as conn:
        db.update_project(conn, created["id"], status="processing")

    response = client.post("/api/projects/delete", json={"ids": [created["id"]]})
    assert response.status_code == 409
    assert client.get(f"/api/projects/{created['id']}").status_code == 200

    with db.connect() as conn:
        db.update_project(conn, created["id"], status="ready")
    assert client.post("/api/projects/delete", json={"ids": [created["id"]]}).status_code == 204


def test_export_with_a_cjk_title_keeps_headers_latin1(client, project):
    """CJK titles used to crash the export: HTTP headers are latin-1 only."""
    from app.schemas import Line, Timeline

    client.patch(f"/api/projects/{project['id']}", json={"title": "残機"})
    client.put(
        f"/api/projects/{project['id']}/timeline",
        json={"timeline": Timeline(lines=[Line(text="あ", start=0.0, end=1.0)]).model_dump()},
    )
    response = client.get(f"/api/projects/{project['id']}/export?format=lrc")
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert 'filename="lyrics.lrc"' in disposition
    assert "filename*=UTF-8''%E6%AE%8B%E6%A9%9F.lrc" in disposition
    assert "[ti:残機]" in response.text


def test_original_lyrics_download_keeps_the_source_timeline(client, project):
    from app import db

    synced = "[00:01.00]line one\n[00:03.00]line two"
    with db.connect() as conn:
        db.update_project(
            conn,
            project["id"],
            original_lyrics="line one\nline two",
            lyrics_source="lrclib",
            lyrics_synced=synced,
        )

    body = client.get(f"/api/projects/{project['id']}").json()
    assert body["lyrics_reference"] is True
    assert body["lyrics_reference_lrc"] == synced
    assert [(item["index"], item["start"]) for item in body["reference_lines"]] == [(0, 1.0), (1, 3.0)]

    lrc = client.get(f"/api/projects/{project['id']}/lyrics/export?format=lrc")
    assert lrc.status_code == 200
    assert lrc.text == synced + "\n"
    assert "attachment" in lrc.headers["content-disposition"]

    txt = client.get(f"/api/projects/{project['id']}/lyrics/export?format=txt")
    assert txt.status_code == 200
    assert txt.text == "line one\nline two\n"


def test_original_lyrics_download_falls_back_to_plain_text(client, project):
    client.put(
        f"/api/projects/{project['id']}/lyrics",
        json={"original_lyrics": "手写的词\n第二行"},
    )
    lrc = client.get(f"/api/projects/{project['id']}/lyrics/export?format=lrc")
    assert lrc.status_code == 200
    # No source timeline was ever supplied: nothing is invented.
    assert lrc.text == "手写的词\n第二行\n"


def test_original_lyrics_download_needs_lyrics(client):
    with open("tests/fixtures/sample.mp3", "rb") as handle:
        created = client.post("/api/projects", files={"file": ("empty-lyrics.mp3", handle, "audio/mpeg")}).json()
    response = client.get(f"/api/projects/{created['id']}/lyrics/export?format=lrc")
    assert response.status_code == 409


def test_pasted_lyrics_are_stored_without_furigana(client, project):
    """Readings would otherwise double the line length the aligner has to place."""
    response = client.put(
        f"/api/projects/{project['id']}/lyrics",
        json={"original_lyrics": "残機（ざんき）を数える\n運命(さだめ)"},
    )
    assert response.status_code == 200
    assert response.json()["original_lyrics"] == "残機を数える\n運命"
