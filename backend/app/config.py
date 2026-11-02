"""Application configuration, read from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings:
    """Runtime settings. Instantiated once at import time."""

    def __init__(self) -> None:
        self.data_dir = Path(os.environ.get("LYRICTRACE_DATA_DIR", PROJECT_DIR / "data")).resolve()
        self.audio_dir = self.data_dir / "audio"
        self.db_path = Path(os.environ.get("LYRICTRACE_DB_PATH", self.data_dir / "lyrictrace.db")).resolve()
        self.max_upload_mb = _int("LYRICTRACE_MAX_UPLOAD_MB", 200)
        self.cors_origins = [
            origin.strip()
            for origin in os.environ.get(
                "LYRICTRACE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
            ).split(",")
            if origin.strip()
        ]

        self.lrclib_base_url = os.environ.get("LYRICTRACE_LRCLIB_URL", "https://lrclib.net").rstrip("/")
        self.lrclib_timeout = _float("LYRICTRACE_LRCLIB_TIMEOUT", 15.0)
        self.lrclib_max_results = _int("LYRICTRACE_LRCLIB_MAX_RESULTS", 10)
        self.user_agent = os.environ.get(
            "LYRICTRACE_USER_AGENT", "LyricTrace/0.1.0 (https://github.com/lyrictrace)"
        )

        # Lyric sources after NetEase (the only one with a 逐字 catalogue).
        # An empty URL disables that source.
        self.lrcapi_url = os.environ.get("LYRICTRACE_LRCAPI_URL", "").rstrip("/")
        self.lrcapi_timeout = _float("LYRICTRACE_LRCAPI_TIMEOUT", 30.0)
        self.netease_url = os.environ.get("LYRICTRACE_NETEASE_URL", "https://music.163.com").rstrip("/")
        self.netease_timeout = _float("LYRICTRACE_NETEASE_TIMEOUT", 15.0)

        # Pipeline behaviour
        self.auto_search = _bool("LYRICTRACE_AUTO_SEARCH", True)
        self.auto_select_threshold = _float("LYRICTRACE_AUTO_SELECT_THRESHOLD", 0.6)
        # When the entry the lyrics came from has no 逐字 data, look for it in
        # another edition of the same song (the live cut usually has one). On by
        # default; a project can switch it off, and so can this global default.
        self.word_sync_search = _bool("LYRICTRACE_WORD_SYNC_SEARCH", True)
        self.metadata_extractor = os.environ.get("LYRICTRACE_METADATA_EXTRACTOR", "llm")
        self.llm_base_url = os.environ.get("LYRICTRACE_LLM_BASE_URL", "").rstrip("/")
        self.llm_api_key = os.environ.get("LYRICTRACE_LLM_API_KEY", "")
        self.llm_model = os.environ.get("LYRICTRACE_LLM_MODEL", "")
        self.llm_timeout = _float("LYRICTRACE_LLM_TIMEOUT", 180.0)

        # Model services (HTTP)
        self.asr_url = os.environ.get("LYRICTRACE_ASR_URL", "").rstrip("/")
        self.asr_language = os.environ.get("LYRICTRACE_ASR_LANGUAGE", "")
        self.asr_timeout = _float("LYRICTRACE_ASR_TIMEOUT", 3600.0)
        # The ASR service's own forced aligner. Off by default: per-character
        # timings come from the recogniser's own tokens (``char_align``), and on
        # singing the ASR's aligner is the weaker of the two (upstream: 80%
        # zero-width Japanese characters). A project can switch it on to keep its
        # tokens as the fallback for lines the forced aligner cannot align.
        self.asr_align = _bool("LYRICTRACE_ASR_ALIGN", False)
        self.separator_url = os.environ.get("LYRICTRACE_SEPARATOR_URL", "").rstrip("/")
        self.separator_model = os.environ.get("LYRICTRACE_SEPARATOR_MODEL", "")
        self.separator_timeout = _float("LYRICTRACE_SEPARATOR_TIMEOUT", 1800.0)
        # Optional per-character aligner service: an extra opinion on each
        # character's time, used only when a project turns it on.
        self.char_aligner_url = os.environ.get("LYRICTRACE_CHAR_ALIGNER_URL", "").rstrip("/")
        # When the ASR's window for a line does not hold up, try shifted windows
        # and let the aligner pick where the line really is. Costs a handful of
        # extra requests per doubtful line, none for clean ones.
        self.char_align_search = _bool("LYRICTRACE_CHAR_ALIGN_SEARCH", True)
        # Fuse a second per-character table (the ASR service's own aligner) into
        # the recogniser's where the two agree. Two independent measurements of
        # the same instant average out; measured on the six songs in ``/root/yrc``
        # it takes the per-song median of per-character medians from 0.100 s to
        # 0.088 s. Costs one extra ASR pass per run.
        self.char_fuse = _bool("LYRICTRACE_CHAR_FUSE", True)
        # Whether the external forced aligner takes part in the per-character
        # times at all. Off by default: the per-character table is then the
        # recogniser's own, and no aligner request is made. A forced aligner
        # places characters measurably better (0.085 s against 0.140 s on the
        # six songs in ``/root/yrc``), so a project can switch it on; this is
        # only the default.
        self.char_hfa = _bool("LYRICTRACE_CHAR_HFA", False)
        # The default for a *new* project's 「按语意合并成词」 switch. Off: the
        # finished timeline keeps one time per character, which is what was
        # measured. A project can turn it on to have the LLM say where words
        # end and merge each unit's span (``LYRICTRACE_LLM_MERGE=true`` makes
        # that the default; the per-project field still wins).
        self.llm_merge = _bool("LYRICTRACE_LLM_MERGE", False)

        # Cache limits. Everything below is a *cache*: the one-shot jobs keep
        # their upload and ASR slices after the answer is read, a project keeps
        # its work dir, and an aborted upload leaves a staging file. Nothing
        # deleted them before, so the disk grew by about one song per run
        # (``services/housekeeping`` sweeps them on a timer).
        # The one-shot jobs kept for the web UI (``/api/cache/jobs``): only the
        # newest few requests keep their upload *and* their finished result.
        # Everything older is swept — a cache of results is only useful while a
        # person can still recognise which request it came from.
        self.oneshot_cache_jobs = _int("LYRICTRACE_ONESHOT_CACHE_JOBS", 5)
        self.oneshot_max_mb = _int("LYRICTRACE_ONESHOT_MAX_MB", 2000)
        self.oneshot_max_age_hours = _int("LYRICTRACE_ONESHOT_MAX_AGE_HOURS", 24)
        # A finished one-shot job keeps ``input`` + ``job.json`` (the result the
        # caller polls). Its ``work/`` dir — the 20 s ASR slices, ~100 MB for a
        # four-minute song — is dropped as soon as the job ends: nothing reads
        # it again. Set to true to keep it for manual inspection.
        self.oneshot_keep_work = _bool("LYRICTRACE_ONESHOT_KEEP_WORK", False)
        self.work_max_mb = _int("LYRICTRACE_WORK_MAX_MB", 2000)
        self.upload_stale_minutes = _int("LYRICTRACE_UPLOAD_STALE_MINUTES", 60)
        # A project's work dir (prepared vocals + every per-line slice) is
        # dropped when a run finishes; the next run regenerates it. True keeps
        # them, and the size limit below then applies instead.
        self.keep_work_dirs = _bool("LYRICTRACE_KEEP_WORK_DIRS", False)
        # 0 disables the timer (the sweep can still be triggered by hand).
        self.housekeeping_interval_min = _int("LYRICTRACE_HOUSEKEEPING_INTERVAL_MIN", 60)

        # Provider selection
        self.asr_provider = os.environ.get("LYRICTRACE_ASR_PROVIDER", "qwen-asr-http")
        self.aligner_provider = os.environ.get("LYRICTRACE_ALIGNER_PROVIDER", "llm-aligner")
        self.separator_provider = os.environ.get("LYRICTRACE_SEPARATOR_PROVIDER", "bs-roformer")
        self.corrector_provider = os.environ.get("LYRICTRACE_CORRECTOR_PROVIDER", "unavailable")
        self.importer_provider = os.environ.get("LYRICTRACE_IMPORTER_PROVIDER", "unavailable")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
