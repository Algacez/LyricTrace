"""Original lyrics: the NetEase → LRCLIB → LrcApi chain, ranking and selection."""

import asyncio

import pytest

from app.services import lyrics_fetch
from app.services.providers.base import LyricsCandidate


def _candidate(**fields):
    base = dict(track_name="残機", artist_name="ずっと真夜中でいいのに。", album_name="残機", duration=261.0)
    base.update(fields)
    return LyricsCandidate(**base)


def test_plain_lyrics_are_preferred_and_lrc_stamps_are_stripped():
    candidate = _candidate(
        plain_lyrics="ああ\nいい",
        synced_lyrics="[00:01.00]ああ\n[00:02.00]いい",
    )
    assert lyrics_fetch.candidate_lyrics(candidate) == "ああ\nいい"


def test_synced_only_entry_is_reduced_to_text():
    candidate = _candidate(plain_lyrics=None, synced_lyrics="[00:01.00]ああ\n[00:02.00]いい")
    # The API timings must never reach the aligner: only the text survives.
    assert lyrics_fetch.candidate_lyrics(candidate) == "ああ\nいい"


def test_candidate_without_any_text_is_dropped():
    assert lyrics_fetch.candidate_lyrics(_candidate(plain_lyrics=None, synced_lyrics=None)) == ""
    matches = lyrics_fetch.rank_candidates([_candidate(plain_lyrics=None)], title="残機")
    assert matches == []


def test_rank_candidates_prefers_metadata_match():
    good = _candidate(plain_lyrics="ああ", synced_lyrics="[00:01.00]ああ")
    plain = _candidate(plain_lyrics="ああ", album_name="别的专辑", duration=300.0)
    matches = lyrics_fetch.rank_candidates(
        [plain, good], title="残機", artist="ずっと真夜中でいいのに。", duration=261.0
    )
    assert matches[0].candidate.synced_lyrics
    assert matches[0].score > matches[1].score


def test_rank_candidates_drops_a_different_artist():
    """A same-titled song by someone else must not reach the aligner."""
    good = _candidate(plain_lyrics="ああ")
    wrong_artist = _candidate(artist_name="別の人", plain_lyrics="ああ")
    matches = lyrics_fetch.rank_candidates(
        [wrong_artist, good], title="残機", artist="ずっと真夜中でいいのに。"
    )
    assert [match.candidate.artist_name for match in matches] == ["ずっと真夜中でいいのに。"]


def test_a_cover_finds_the_original_when_the_running_time_matches():
    """【秋葵AKIAOI】怪物（TV size） is a cover: the lyrics are YOASOBI's."""
    original = _candidate(
        track_name="怪物",
        artist_name="YOASOBI",
        album_name="",
        duration=90.0,
        plain_lyrics="ああ",
    )
    matches = lyrics_fetch.rank_candidates(
        [original], title="怪物", artist="秋葵AKIAOI", duration=90.12
    )
    assert [match.candidate.artist_name for match in matches] == ["YOASOBI"]


def test_a_live_cut_finds_the_studio_entry_it_runs_past():
    """「勾指起誓」 at 331 s is the 184 s studio song plus talk: same lyrics.

    The rule is one-sided because only a *longer* entry is harmful. This
    direction is what used to fail: every candidate was dropped for being
    shorter than the audio, and the run fell back to the ASR transcript.
    """
    studio = _candidate(
        track_name="勾指起誓",
        artist_name="洛天依Official / ilem",
        album_name="",
        duration=183.536,
        plain_lyrics="你是信的开头",
    )
    matches = lyrics_fetch.rank_candidates(
        [studio], title="勾指起誓", artist="鸟憨", duration=331.33
    )
    assert [match.candidate.track_name for match in matches] == ["勾指起誓"]


def test_a_same_titled_song_of_another_length_is_still_dropped():
    """A cover does not make every namesake the same song: 90 s is not 206 s."""
    full_length = _candidate(
        track_name="怪物", artist_name="YOASOBI", album_name="", duration=206.0, plain_lyrics="ああ"
    )
    matches = lyrics_fetch.rank_candidates(
        [full_length], title="怪物", artist="秋葵AKIAOI", duration=90.12
    )
    assert matches == []


def test_a_cover_candidate_needs_a_title_match_too():
    """Title is what carries the cover case; a different title is a different song."""
    other = _candidate(
        track_name="残機", artist_name="YOASOBI", album_name="", duration=90.0, plain_lyrics="ああ"
    )
    matches = lyrics_fetch.rank_candidates(
        [other], title="怪物", artist="秋葵AKIAOI", duration=90.12
    )
    assert matches == []


def test_rank_candidates_accepts_a_partially_written_artist():
    """泠鸢 vs 泠鸢yousa is the same singer: never hide that entry."""
    candidate = _candidate(track_name="我的故事未写完", artist_name="泠鸢yousa", plain_lyrics="ああ")
    matches = lyrics_fetch.rank_candidates([candidate], title="我的故事未写完", artist="泠鸢")
    assert matches and matches[0].candidate.artist_name == "泠鸢yousa"


def test_fetch_best_respects_the_auto_select_threshold(monkeypatch):
    from app.config import settings

    def fake_search(score):
        async def search(project, *, query=None):
            return [lyrics_fetch.LyricsMatch(candidate=_candidate(), score=score, lyrics="ああ")]

        return search

    import asyncio

    monkeypatch.setattr(settings, "auto_select_threshold", 0.6)
    monkeypatch.setattr(lyrics_fetch, "search", fake_search(0.59))
    assert asyncio.run(lyrics_fetch.fetch_best({"title": "残機"})) is None

    monkeypatch.setattr(lyrics_fetch, "search", fake_search(0.6))
    match = asyncio.run(lyrics_fetch.fetch_best({"title": "残機"}))
    assert match is not None and match.lyrics == "ああ"


def test_auto_fetch_is_skipped_when_disabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "auto_search", False)
    assert lyrics_fetch.auto_fetch({"title": "残機"}) is None


def test_auto_fetch_swallows_network_errors(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "auto_search", True)

    async def boom(project, *, query=None):
        raise RuntimeError("lrclib down")

    monkeypatch.setattr(lyrics_fetch, "search", boom)
    # A dead API must never fail the upload task.
    assert lyrics_fetch.auto_fetch({"title": "残機"}) is None


@pytest.fixture()
def fake_lrclib(monkeypatch):
    """Answer LRCLIB lookups locally and record the calls."""

    calls: list[dict] = []

    async def fake_search(project, *, query=None):
        calls.append({"project": project, "query": query})
        return [
            lyrics_fetch.LyricsMatch(
                candidate=_candidate(
                    plain_lyrics="しょっぱいぜ\n初めて嗅いで舐めた出会い",
                    lrclib_id=34756429,
                ),
                score=1.0,
                lyrics="しょっぱいぜ\n初めて嗅いで舐めた出会い",
            )
        ]

    monkeypatch.setattr(lyrics_fetch, "search", fake_search)
    monkeypatch.setattr(lyrics_fetch, "search_all", fake_search)
    return calls


def test_align_fills_lyrics_from_the_api(client, fake_lrclib, monkeypatch):
    """The lookup happens when the run starts, not when the file was uploaded."""
    from app.config import settings
    from tests.conftest import AUDIO_FIXTURE, wait_for_project, wait_for_task

    # The lookup is what this test is about; the ASR that would follow it is not.
    from app.services import pipeline

    def no_asr(*_args, **_kwargs):
        raise AssertionError("the align run must not reach the ASR in this test")

    monkeypatch.setattr(pipeline, "_align_from_asr", no_asr)
    settings.auto_search = True
    try:
        with AUDIO_FIXTURE.open("rb") as handle:
            response = client.post(
                "/api/projects",
                files={"file": ("ずっと真夜中でいいかに。 - 残機.mp3", handle, "audio/mpeg")},
            )
        assert response.status_code == 201, response.text
        project_id = response.json()["id"]
        task = wait_for_task(client, client.get(f"/api/projects/{project_id}/tasks").json()[0]["id"])
        assert task["status"] == "succeeded", task
        # Uploading the file must not have looked anything up: no lyrics yet.
        data = wait_for_project(client, project_id)
        assert data["lyrics_source"] is None
        assert data["original_lyrics"] == ""
        assert not fake_lrclib, "the upload asked the lyric sources"
        # Starting the run is what fetches them.
        align = client.post(f"/api/projects/{project_id}/align", json={})
        assert align.status_code == 202, align.text
        wait_for_task(client, align.json()["id"])
        data = wait_for_project(client, project_id)
    finally:
        settings.auto_search = False

    assert data["lyrics_source"] == "lrclib"
    assert data["lyrics_lrclib_id"] == 34756429
    assert data["original_lyrics"].startswith("しょっぱいぜ")
    # Text only: the timeline exists but carries no invented timing.
    assert data["timeline_status"] == "untimed"
    assert all(line["start"] is None for line in data["timeline"]["lines"])
    assert fake_lrclib and fake_lrclib[0]["project"]["title"]


def test_pipeline_skips_the_lyrics_lookup_when_fetch_lyrics_is_off(client, monkeypatch):
    """``fetch_lyrics=False`` on a project with no pasted lyrics → no source call."""
    from app.config import settings
    from app.services import lyrics_fetch
    from app.services.pipeline import _fetch_lyrics_fields
    from tests.conftest import AUDIO_FIXTURE

    monkeypatch.setattr(settings, "auto_search", True)
    searched = []

    def should_not_fetch(_project):
        searched.append(True)
        raise AssertionError("auto_fetch ran while fetch_lyrics was off")

    monkeypatch.setattr(lyrics_fetch, "auto_fetch", should_not_fetch)
    try:
        fields = _fetch_lyrics_fields(
            {"title": "残機", "original_lyrics": "", "fetch_lyrics": False}
        )
        assert fields == {}
        # A real upload with no pasted lyrics and fetch_lyrics off must also skip.
        with AUDIO_FIXTURE.open("rb") as handle:
            created = client.post(
                "/api/projects",
                files={"file": ("原创曲 - 泠鸢.mp3", handle, "audio/mpeg")},
            ).json()
        client.patch(f"/api/projects/{created['id']}", json={"fetch_lyrics": False})
        assert _fetch_lyrics_fields(
            {"title": "残機", "original_lyrics": "", "fetch_lyrics": False}
        ) == {}
    finally:
        settings.auto_search = False
    assert searched == []


def test_pipeline_keeps_user_lyrics_over_the_api(client, project, fake_lrclib):
    from app.config import settings
    from app.services.pipeline import _fetch_lyrics_fields

    client.put(f"/api/projects/{project['id']}/lyrics", json={"original_lyrics": "我的歌词\n第二行"})
    settings.auto_search = True
    try:
        # A second pipeline run must not overwrite what the user typed.
        fields = _fetch_lyrics_fields({"title": "残機", "original_lyrics": "我的歌词\n第二行"})
    finally:
        settings.auto_search = False

    assert fields == {}
    data = client.get(f"/api/projects/{project['id']}").json()
    assert data["original_lyrics"] == "我的歌词\n第二行"
    assert data["lyrics_source"] == "manual"


def test_lyrics_search_endpoint_returns_ranked_candidates(client, project, fake_lrclib):
    response = client.post(f"/api/projects/{project['id']}/lyrics/search", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidates"][0]["lrclib_id"] == 34756429
    assert body["candidates"][0]["line_count"] == 2
    assert "しょっぱいぜ" in body["candidates"][0]["preview"]


def test_selecting_lyrics_stores_text_and_clears_asr_cache(client, project, monkeypatch):
    async def fake_get_external(source, external_id, project=None):
        assert source == "lrclib"
        assert external_id == "34756429"
        return lyrics_fetch.LyricsMatch(
            candidate=_candidate(lrclib_id=int(external_id), plain_lyrics="ああ\nいい"),
            score=1.0,
            lyrics="ああ\nいい",
        )

    monkeypatch.setattr(lyrics_fetch, "fetch_by_external", fake_get_external)
    response = client.post(f"/api/projects/{project['id']}/lyrics/select", json={"lrclib_id": 34756429})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["original_lyrics"] == "ああ\nいい"
    assert body["lyrics_source"] == "lrclib"
    assert body["lyrics_lrclib_id"] == 34756429
    assert [line["text"] for line in body["timeline"]["lines"]] == ["ああ", "いい"]


def test_editing_lyrics_marks_them_manual(client, project, fake_lrclib):
    from app import db
    from app.config import settings
    from app.services.pipeline import _fetch_lyrics_fields

    settings.auto_search = True
    try:
        fields = _fetch_lyrics_fields({"title": "残機"})
    finally:
        settings.auto_search = False
    assert fields["lyrics_source"] == "lrclib"
    with db.connect() as conn:
        db.update_project(conn, project["id"], **fields)

    body = client.put(f"/api/projects/{project['id']}/lyrics", json={"original_lyrics": "手改的词"}).json()
    assert body["lyrics_source"] == "manual"
    assert body["lyrics_lrclib_id"] is None


def _lrcapi_candidate(**fields):
    base = dict(
        track_name="我的故事未写完",
        artist_name="泠鸢yousa",
        plain_lyrics="记忆就像是画片\n被风吹起又飞远",
        synced_lyrics="[00:25.14]记忆就像是画片\n[00:30.82]被风吹起又飞远",
        external_id="189867024255956882195958113408768915680",
        source="lrcapi",
    )
    base.update(fields)
    return LyricsCandidate(**base)


def test_lrclib_tries_the_track_name_alone_before_giving_up(monkeypatch):
    """A near-miss artist (泠鸢 vs 泠鸢yousa) must not hide an existing entry."""
    from app.services.providers import lrclib

    calls: list[dict] = []

    async def fake_search(self, **params):
        calls.append(params)
        if "artist_name" in params:
            return []  # the strict artist filter misses
        return [
            _candidate(track_name="我的故事未写完", artist_name="泠鸢yousa", plain_lyrics="ああ", lrclib_id=1)
        ]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", fake_search)
    matches = asyncio.run(
        lyrics_fetch.search({"title": "我的故事未写完", "artist": "泠鸢"})
    )
    assert matches and matches[0].source == "lrclib"
    assert calls[0]["artist_name"] == "泠鸢"
    assert calls[1] == {"track_name": "我的故事未写完"}


def test_netease_searches_by_title_when_the_cover_artist_finds_nothing(monkeypatch):
    """The cover credit only reaches the singer's own songs; the title does not."""
    from app.services.providers import lrclib, lrcapi, netease

    async def empty(self, **params):
        return []

    calls: list[dict] = []
    original = LyricsCandidate(
        track_name="怪物",
        artist_name="YOASOBI",
        duration=90.0,
        plain_lyrics="嘘呼\n素晴らしき世界に今日も乾杯",
        external_id="1",
        source="netease",
    )
    cover_singer = LyricsCandidate(
        track_name="花束（吉他弹唱）",
        artist_name="秋葵AKIAOI",
        duration=134.2,
        plain_lyrics="别的歌",
        external_id="2",
        source="netease",
    )

    async def fake_search(self, **params):
        calls.append(params)
        if params.get("artist_name"):
            return [cover_singer]  # the singer's own catalogue, wrong song
        return [original]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: False)
    monkeypatch.setattr(netease.NeteaseProvider, "available", lambda self: True)
    monkeypatch.setattr(netease.NeteaseProvider, "search", fake_search)

    matches = asyncio.run(
        lyrics_fetch.search({"title": "怪物", "artist": "秋葵AKIAOI", "duration": 90.12})
    )
    assert matches and matches[0].candidate.artist_name == "YOASOBI"
    assert [call.get("artist_name") for call in calls] == ["秋葵AKIAOI", None]


def test_netease_keeps_one_search_when_the_artist_is_already_right(monkeypatch):
    """No second round trip when the first one answered usefully."""
    from app.services.providers import lrclib, lrcapi, netease

    async def empty(self, **params):
        return []

    calls: list[dict] = []

    async def fake_search(self, **params):
        calls.append(params)
        return [
            LyricsCandidate(
                track_name="残機",
                artist_name="ずっと真夜中でいいのに。",
                duration=261.0,
                plain_lyrics="ああ",
                external_id="3",
                source="netease",
            )
        ]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: False)
    monkeypatch.setattr(netease.NeteaseProvider, "available", lambda self: True)
    monkeypatch.setattr(netease.NeteaseProvider, "search", fake_search)

    matches = asyncio.run(
        lyrics_fetch.search(
            {"title": "残機", "artist": "ずっと真夜中でいいのに。", "duration": 261.0}
        )
    )
    assert matches
    assert len(calls) == 1


def test_falls_through_to_the_internal_lrcapi(monkeypatch):
    from app.services.providers import lrclib, lrcapi

    async def empty(self, **params):
        return []

    async def found(self, **params):
        assert params["track_name"] == "我的故事未写完"
        return [_lrcapi_candidate()]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: True)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "search", found)

    match = asyncio.run(lyrics_fetch.fetch_best({"title": "我的故事未写完", "artist": "泠鸢"}))
    assert match is not None
    assert match.source == "lrcapi"
    assert match.lyrics.startswith("记忆就像是画片")
    # Its LRC timeline survives as the positional hint.
    assert "[00:25.14]" in match.synced


def test_falls_through_to_netease_when_the_others_are_empty(monkeypatch):
    from app.services.providers import lrclib, lrcapi, netease

    async def empty(self, **params):
        return []

    async def found(self, **params):
        return [
            LyricsCandidate(
                track_name="我的故事未写完",
                artist_name="泠鸢yousa",
                plain_lyrics="记忆就像是画片",
                synced_lyrics="[00:25.40]记忆就像是画片",
                external_id="2664421415",
                source="netease",
            )
        ]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: True)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "search", empty)
    monkeypatch.setattr(netease.NeteaseProvider, "available", lambda self: True)
    monkeypatch.setattr(netease.NeteaseProvider, "search", found)

    match = asyncio.run(lyrics_fetch.fetch_best({"title": "我的故事未写完"}))
    assert match is not None and match.source == "netease"


def test_a_weak_source_hit_is_skipped_for_the_next_source(monkeypatch):
    """A same-titled song by someone else is worse than asking the next source."""
    from app.services.providers import lrclib, lrcapi

    async def wrong_song(self, **params):
        return [LyricsCandidate(track_name="我的故事未写完", artist_name="别人", plain_lyrics="别的词")]

    async def right_song(self, **params):
        return [_lrcapi_candidate()]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", wrong_song)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: True)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "search", right_song)

    match = asyncio.run(lyrics_fetch.fetch_best({"title": "我的故事未写完", "artist": "泠鸢yousa"}))
    assert match is not None and match.source == "lrcapi"


def test_search_all_keeps_results_from_every_source(monkeypatch):
    from app.services.providers import lrclib, lrcapi

    async def from_lrclib(self, **params):
        return [_candidate(plain_lyrics="ああ", lrclib_id=34756429)]

    async def from_lrcapi(self, **params):
        return [_lrcapi_candidate(track_name="残機")]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", from_lrclib)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: True)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "search", from_lrcapi)

    matches = asyncio.run(lyrics_fetch.search_all({"title": "残機"}))
    assert {match.source for match in matches} == {"lrclib", "lrcapi"}


def test_selecting_a_lrcapi_entry_repeats_the_search_by_id(monkeypatch):
    from app.services.providers import lrcapi

    async def search(self, **params):
        return [_lrcapi_candidate()]

    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: True)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "search", search)

    match = asyncio.run(
        lyrics_fetch.fetch_by_external(
            "lrcapi", "189867024255956882195958113408768915680", {"title": "我的故事未写完"}
        )
    )
    assert match is not None and match.source == "lrcapi"
    assert asyncio.run(lyrics_fetch.fetch_by_external("lrcapi", "missing", {"title": "x"})) is None


def test_search_keeps_looking_past_a_text_only_source_for_word_timings(monkeypatch):
    """A later source may carry 逐字 timings for the same song: that wins."""
    plain = _candidate(plain_lyrics="ああ", source="lrclib", external_id="1")
    synced = _candidate(
        plain_lyrics="ああ",
        source="netease",
        external_id="2",
        word_sync_raw="[1000,2000](1000,500,0)あ\n",
        word_sync_format="yrc",
    )

    async def fake_provider(source, fields, query):
        return [plain] if source == "lrclib" else [synced]

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert matches[0].source == "netease"
    assert matches[0].has_word_sync


def test_search_falls_back_to_text_when_no_source_has_word_timings(monkeypatch):
    plain = _candidate(plain_lyrics="ああ", source="lrclib", external_id="1")

    async def fake_provider(source, fields, query):
        return [plain] if source == "lrclib" else []

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert matches and matches[0].source == "lrclib"


def test_search_prefers_a_lower_ranked_entry_that_has_word_timings(monkeypatch):
    """Two editions score alike; the one with yrc is taken even if ranked lower."""
    studio = _candidate(
        track_name="残機",
        plain_lyrics="ああ",
        synced_lyrics="[00:10.00]ああ",
        external_id="1",
        source="netease",
    )
    live = _candidate(
        track_name="残機",
        plain_lyrics="ああ",
        external_id="2",
        source="netease",
        word_sync_raw="[1000,2000](1000,500,0)あ\n",
        word_sync_format="yrc",
    )

    async def fake_provider(source, fields, query):
        return [studio, live] if source == "netease" else []

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert matches[0].external_id == "2"
    assert matches[0].has_word_sync
    # The text-only entry is still offered to the manual picker, after it.
    assert [match.external_id for match in matches[1:]] == ["1"]


def test_search_keeps_the_better_text_when_the_word_timed_edition_is_weaker(monkeypatch):
    """A live cut has yrc but also its MC section: its text is not the lyrics.

    The best text wins and the 逐字 timings are fetched from that edition later
    by ``find_word_sync``, which only needs the timings, not the text.
    """
    studio = _candidate(track_name="残機", plain_lyrics="ああ", external_id="1", source="netease")
    live = _candidate(
        track_name="残機 (Live)",
        plain_lyrics="ああ\nMC 感谢大家",
        external_id="2",
        source="netease",
        word_sync_raw="[1000,2000](1000,500,0)あ\n",
        word_sync_format="yrc",
    )

    async def fake_provider(source, fields, query):
        return [studio, live] if source == "netease" else []

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert matches[0].external_id == "1"
    assert not matches[0].has_word_sync
    # The word-timed edition is still listed for the manual picker.
    assert any(match.external_id == "2" and match.has_word_sync for match in matches)


def test_netease_is_searched_before_lrclib(monkeypatch):
    """NetEase carries the 逐字 catalogue, so it is asked first."""
    assert lyrics_fetch.SOURCE_ORDER[0] == "netease"
    seen: list[str] = []

    async def fake_provider(source, fields, query):
        seen.append(source)
        return [_candidate(plain_lyrics="ああ", source=source, external_id=source)]

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert seen[0] == "netease"
    assert matches[0].source == "netease"


def test_search_all_puts_word_timed_entries_first(monkeypatch):
    plain = _candidate(plain_lyrics="ああ", source="lrclib", external_id="1")
    synced = _candidate(
        plain_lyrics="ああ",
        source="netease",
        external_id="2",
        word_sync_raw="[1000,2000](1000,500,0)あ\n",
        word_sync_format="yrc",
    )

    async def fake_provider(source, fields, query):
        return [plain] if source == "lrclib" else [synced] if source == "netease" else []

    monkeypatch.setattr(lyrics_fetch, "_search_provider", fake_provider)
    matches = asyncio.run(lyrics_fetch.search_all({"title": "残機", "artist": "ずっと真夜中でいいのに。"}))
    assert matches[0].has_word_sync


def test_lrclib_candidate_keeps_the_lyricsfile_when_it_has_word_sync():
    from app.services.providers.lrclib import LrclibProvider

    item = {
        "id": 7,
        "trackName": "残機",
        "artistName": "ずっと真夜中でいいのに。",
        "duration": 261.0,
        "hasWordSync": True,
        "plainLyrics": "しょっぱいぜ",
        "syncedLyrics": "[00:10.00]しょっぱいぜ",
        "lyricsfile": "lines:\n- text: しょっぱいぜ\n  start_ms: 10000\n  words:\n  - text: しょ\n    start_ms: 10000\n",
    }
    candidate = LrclibProvider._to_candidate(item)
    assert candidate.word_sync_format == "lyricsfile"
    assert candidate.raw["hasWordSync"] is True
    assert [line.text for line in _match_lines(candidate)] == ["しょっぱいぜ"]


def _match_lines(candidate):
    from app.services import word_sync

    return word_sync.parse_lyricsfile(candidate.word_sync_raw)


def test_lrclib_candidate_without_word_sync_carries_no_payload():
    from app.services.providers.lrclib import LrclibProvider

    item = {
        "id": 8,
        "trackName": "残機",
        "artistName": "ずっと真夜中でいいのに。",
        "hasWordSync": False,
        "lyricsfile": "lines:\n- text: しょっぱいぜ\n  start_ms: 10000\n",
    }
    candidate = LrclibProvider._to_candidate(item)
    assert candidate.word_sync_raw is None
    assert candidate.word_sync_format is None


def test_fetch_lyrics_fields_stores_the_word_sync_payload(monkeypatch):
    """The align handler needs the payload later; it must survive the lookup."""
    from app.config import settings
    from app.services import pipeline

    match = lyrics_fetch.LyricsMatch(
        candidate=_candidate(
            plain_lyrics="しょっぱいぜ",
            source="netease",
            external_id="42",
            word_sync_raw="[10000,2000](10000,500,0)しょっぱいぜ\n",
            word_sync_format="yrc",
        ),
        score=1.0,
        lyrics="しょっぱいぜ",
    )
    monkeypatch.setattr(settings, "auto_search", True)
    monkeypatch.setattr(lyrics_fetch, "auto_fetch", lambda project: match)
    fields = pipeline._fetch_lyrics_fields({"title": "残機"})
    assert fields["lyrics_source"] == "netease"
    assert fields["lyrics_word_sync_format"] == "yrc"
    assert fields["lyrics_word_sync"].startswith("[10000,2000]")


def test_find_word_sync_uses_another_edition_of_the_same_song(monkeypatch):
    """The studio entry has no yrc; the live edition of the same song does."""
    from app.services.providers import lrclib, lrcapi, netease

    async def empty(self, **params):
        return []

    studio = _candidate(
        track_name="我的故事未写完",
        artist_name="泠鸢yousa",
        plain_lyrics="记忆就像是画片\n被风吹起又飞远",
        source="netease",
        external_id="2664421415",
    )
    live = _candidate(
        track_name="我的故事未写完 (Live)",
        artist_name="泠鸢yousa",
        plain_lyrics="记忆就像是画片\n被风吹起又飞远",
        source="netease",
        external_id="2020316718",
        word_sync_raw="[22310,4300](22310,2880,0)记忆\n[28570,2000](28570,600,0)就像是画片\n",
        word_sync_format="yrc",
    )

    async def search(self, **params):
        return [studio, live]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: False)
    monkeypatch.setattr(netease.NeteaseProvider, "available", lambda self: True)
    monkeypatch.setattr(netease.NeteaseProvider, "search", search)

    from app.schemas import Line

    found = asyncio.run(
        lyrics_fetch.find_word_sync(
            {"title": "我的故事未写完", "artist": "泠鸢yousa"},
            [Line(text="记忆就像是画片"), Line(text="被风吹起又飞远")],
            exclude="2664421415",
        )
    )
    assert found is not None
    assert found.external_id == "2020316718"
    assert found.agreement == 1.0
    assert found.edition_label.startswith("我的故事未写完 (Live)")
    assert found.word_sync_lines()


def test_find_word_sync_ignores_an_edition_with_different_lyrics(monkeypatch):
    from app.services.providers import lrclib, lrcapi, netease
    from app.schemas import Line

    async def empty(self, **params):
        return []

    other = _candidate(
        track_name="我的故事未写完",
        artist_name="泠鸢yousa",
        plain_lyrics="完全不同的歌词\n和另一段内容",
        source="netease",
        external_id="99",
        word_sync_raw="[1000,500](1000,500,0)完全不同的歌词\n",
        word_sync_format="yrc",
    )

    async def search(self, **params):
        return [other]

    monkeypatch.setattr(lrclib.LrclibProvider, "search", empty)
    monkeypatch.setattr(lrcapi.LrcApiProvider, "available", lambda self: False)
    monkeypatch.setattr(netease.NeteaseProvider, "available", lambda self: True)
    monkeypatch.setattr(netease.NeteaseProvider, "search", search)

    assert (
        asyncio.run(
            lyrics_fetch.find_word_sync(
                {"title": "我的故事未写完", "artist": "泠鸢yousa"},
                [Line(text="记忆就像是画片"), Line(text="被风吹起又飞远")],
            )
        )
        is None
    )


def test_netease_credit_lines_are_not_lyrics():
    """NetEase ships credits as timed lines; they must not reach the aligner."""
    from app.services.lyrics import strip_credits
    from app.schemas import Line

    lines = [
        Line(text="词曲唱：泠鸢yousa"),
        Line(text="混音&母带 : 陆本平/喜宝Kiho"),
        Line(text="混音母带：陆本平、喜宝Kiho"),
        Line(text="记忆就像是画片"),
        Line(text="我的故事未写完"),
    ]
    assert [line.text for line in strip_credits(lines)] == ["记忆就像是画片", "我的故事未写完"]
