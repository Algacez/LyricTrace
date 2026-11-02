"""Minimal OpenAI-compatible chat client for the configured LLM endpoint."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger("lyrictrace.llm")

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper around ``POST {base}/chat/completions``."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout or settings.llm_timeout

    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def chat(self, prompt: str, *, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        if not self.available():
            raise LLMError("LLM endpoint is not configured (LYRICTRACE_LLM_BASE_URL / LYRICTRACE_LLM_MODEL).")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Qwen-style servers: skip the reasoning pass, we only want the answer.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM response: {data}") from exc

    def chat_json(self, prompt: str, *, max_tokens: int = 1500) -> Any:
        """Ask for JSON and tolerate code fences / stray prose around it."""
        raw = self.chat(prompt, max_tokens=max_tokens)
        text = FENCE_RE.sub("", raw.strip()).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fall back to the first balanced JSON object/array in the reply.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise LLMError(f"LLM did not return JSON: {raw[:400]}")
