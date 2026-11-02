"""Song language inference: LLM first, script heuristic second, auto otherwise."""

from app.services import language


def test_normalize_maps_aliases_onto_asr_names():
    assert language.normalize("ja") == "Japanese"
    assert language.normalize("日本語") == "Japanese"
    assert language.normalize("japanese") == "Japanese"
    assert language.normalize("中文") == "Chinese"
    assert language.normalize("粤语") == "Cantonese"
    # Anything the ASR cannot be asked for is not a language we pass on.
    assert language.normalize("Klingon") is None


def test_auto_values_mean_let_the_asr_decide():
    for value in (None, "", "auto", "自动", "自动识别"):
        assert language.is_auto(value)
        assert language.normalize(value) is None
    assert not language.is_auto("Japanese")


def test_script_heuristic_reads_kana_hangul_and_han():
    assert language.guess_from_script("残機 ずっと真夜中でいいのに。") == "Japanese"
    assert language.guess_from_script("사랑해요 우리") == "Korean"
    assert language.guess_from_script("我的故事未写完 泠鸢") == "Chinese"
    # Latin script says nothing useful: that is exactly where a guess is worst.
    assert language.guess_from_script("Radiohead - Creep") is None


def test_llm_answer_wins_over_the_script_heuristic():
    class FakeLLM:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            assert "残機" in prompt
            return {"language": "Japanese", "confidence": 0.92}

    detected, origin = language.detect({"title": "残機", "artist": "ずっと真夜中でいいのに。"}, client=FakeLLM())
    assert (detected, origin) == ("Japanese", "llm")


def test_low_confidence_llm_answer_falls_back_to_script():
    class UnsureLLM:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"language": "English", "confidence": 0.2}

    detected, origin = language.detect({"title": "我的故事未写完"}, client=UnsureLLM())
    assert (detected, origin) == ("Chinese", "script")


def test_unanswerable_song_returns_auto():
    class UnknownLLM:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            return {"language": "unknown", "confidence": 0.0}

    detected, origin = language.detect({"title": "Track 03", "artist": ""}, client=UnknownLLM())
    assert detected is None and origin == "auto"


def test_a_stored_llm_guess_is_asked_again():
    """An earlier run's guess is not a decision: the title alone may have misled it.

    「怪物」 is a Japanese title written in Han characters, and the upload-time
    guess (made before any lyrics were known) called it Chinese. Storing that
    as final would recognise a Japanese song as Chinese on every later run.
    """

    class JapaneseLLM:
        def available(self):
            return True

        def chat_json(self, prompt, max_tokens=0):
            assert "素晴らしき世界" in prompt  # now there are lyrics to read
            return {"language": "Japanese", "confidence": 0.9}

    detected, origin = language.detect(
        {
            "title": "怪物",
            "artist": "秋葵AKIAOI",
            "lyrics_language": "Chinese",
            "language_origin": "llm",
            "original_lyrics": "嗚呼\n素晴らしき世界に今日も乾杯",
        },
        client=JapaneseLLM(),
    )
    assert (detected, origin) == ("Japanese", "llm")


def test_a_manual_choice_is_never_overridden(monkeypatch):
    class NeverCalled:
        def available(self):
            raise AssertionError("the LLM must not be asked when the user chose")

    detected, origin = language.detect(
        {"title": "残機", "lyrics_language": "Chinese"}, client=NeverCalled()
    )
    assert (detected, origin) == ("Chinese", "manual")


def test_detect_without_any_information_does_not_call_the_llm():
    class NeverCalled:
        def available(self):
            raise AssertionError("nothing to ask about")

    assert language.detect({"title": "", "artist": "", "original_lyrics": ""}, client=NeverCalled()) == (
        None,
        "auto",
    )
