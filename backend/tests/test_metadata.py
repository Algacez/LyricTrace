from app.services.filename_parser import parse_filename
from app.services.matching import score_candidate
from app.services.providers.base import LyricsCandidate


def test_parse_artist_dash_title_with_noise():
    parsed = parse_filename("周杰伦 - 晴天 [Official MV] [1080p].mp3")
    assert parsed.artist == "周杰伦"
    assert parsed.title == "晴天"


def test_parse_video_title_with_uploader_and_quality():
    parsed = parse_filename("bilibili_【4K】- Artist - Song Name (Official Video) 1080p.mp3")
    assert parsed.title == "Song Name"
    assert parsed.artist == "Artist"


def test_parse_track_number_prefix():
    parsed = parse_filename("01. Radiohead - Creep.flac")
    assert parsed.artist == "Radiohead"
    assert parsed.title == "Creep"


def test_parse_title_only():
    parsed = parse_filename("Some Instrumental Track.mp3")
    assert parsed.title == "Some Instrumental Track"
    assert parsed.artist == ""


def test_parse_album_bracket():
    parsed = parse_filename("Band - Track (album: Great Hits).mp3")
    assert parsed.album == "Great Hits"
    assert parsed.title == "Track"
    assert parsed.artist == "Band"


def test_parse_keeps_feat():
    parsed = parse_filename("Singer - Tune (feat. Other).mp3")
    assert "feat" in parsed.title.lower()


def test_scoring_prefers_exact_match_with_sync():
    exact = LyricsCandidate(track_name="Creep", artist_name="Radiohead", duration=238, synced_lyrics="[00:01.00]x")
    wrong = LyricsCandidate(track_name="Creep (Live)", artist_name="Someone", duration=300, synced_lyrics="[00:01.00]x")
    score_exact = score_candidate(exact, title="Creep", artist="Radiohead", duration=238)
    score_wrong = score_candidate(wrong, title="Creep", artist="Radiohead", duration=238)
    assert score_exact > score_wrong
    assert 0.8 <= score_exact <= 1.0


def test_scoring_cjk_titles():
    candidate = LyricsCandidate(track_name="晴天", artist_name="周杰伦", synced_lyrics="[00:01.00]x")
    assert score_candidate(candidate, title="晴天", artist="周杰伦") > 0.8


def test_scoring_penalizes_missing_lyrics():
    with_lyrics = LyricsCandidate(track_name="X", artist_name="Y", synced_lyrics="[00:01.00]x")
    without = LyricsCandidate(track_name="X", artist_name="Y")
    assert score_candidate(with_lyrics, title="X", artist="Y") > score_candidate(
        without, title="X", artist="Y"
    )


def test_scoring_does_not_punish_an_entry_shorter_than_the_audio():
    """A live cut runs past the studio entry; only a *longer* entry is harmful."""
    studio = LyricsCandidate(
        track_name="勾指起誓", artist_name="洛天依", duration=183.5, synced_lyrics="[00:01.00]x"
    )
    score = score_candidate(studio, title="勾指起誓", artist="鸟憨", duration=331.33)
    assert score >= 0.7
    # The other direction keeps the exact penalty: 206 s of lyrics on a 90 s cut.
    full = LyricsCandidate(
        track_name="怪物", artist_name="YOASOBI", duration=206.0, synced_lyrics="[00:01.00]x"
    )
    assert score_candidate(full, title="怪物", artist="秋葵AKIAOI", duration=90.12) < score
"""LLM metadata extraction: the messy filename -> title/artist."""

import pytest

from app.schemas import SongMetadata
from app.services.providers.llm_metadata import LLMMetadataExtractor

MESSY = (
    "_ACAね最累的一集，也是听得最爽的一集_「残機 _电锯人ED2」"
    "【ずっと真夜中でいいのに。】_羅武 - 1.….mp3"
)


class FakeClient:
    def __init__(self, payload, *, fail=False):
        self.payload = payload
        self.fail = fail
        self.calls = 0

    def available(self):
        return True

    def chat_json(self, prompt, *, max_tokens=1500):
        self.calls += 1
        if self.fail:
            from app.services.llm import LLMError

            raise LLMError("boom")
        return self.payload


def test_llm_extractor_uses_the_model_answer():
    client = FakeClient({"title": "残機", "artist": "ずっと真夜中でいいのに。", "album": ""})
    meta = LLMMetadataExtractor(client).extract(MESSY)
    assert meta.title == "残機"
    assert meta.artist == "ずっと真夜中でいいのに。"
    assert meta.source == "llm"
    assert client.calls == 1


def test_llm_extractor_falls_back_when_the_call_fails():
    client = FakeClient({}, fail=True)
    meta = LLMMetadataExtractor(client).extract(MESSY)
    assert meta.title  # filename heuristic still produced something
    assert meta.source == "filename"


def test_llm_extractor_rejects_junk_fields():
    client = FakeClient({"title": "unknown", "artist": "N/A", "album": None})
    meta = LLMMetadataExtractor(client).extract("Artist - Song.mp3")
    assert meta.title == "Song"
    assert meta.artist == "Artist"


def test_llm_extractor_skips_the_call_when_tags_are_complete():
    client = FakeClient({"title": "ignored"})
    meta = LLMMetadataExtractor(client).extract("x.mp3", {"title": "T", "artist": "A"})
    assert (meta.title, meta.artist) == ("T", "A")
    assert client.calls == 0


def test_llm_extractor_reports_unavailable_without_llm_config(monkeypatch):
    from app.services import llm

    monkeypatch.setattr(llm.LLMClient, "available", lambda self: False)
    extractor = LLMMetadataExtractor()
    assert extractor.available() is False
    assert "LYRICTRACE_LLM" in extractor.unavailable_detail()
