"""Thin client for any OpenAI-compatible chat endpoint (Groq, Ollama).

Structured output: we ask for a JSON object, then validate it with a Pydantic model.
If validation fails, the error is sent back once so the model can fix its output.
This works the same on every provider, unlike provider-specific schema modes.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from uae_rag.config import LLMConfig

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the LLM cannot be reached or never returns valid output."""


class JSONRejected(LLMError):
    """The provider's own JSON-mode check failed (Groq HTTP 400 json_validate_failed)."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, usage: dict[str, int] | None) -> None:
        self.calls += 1
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))


@dataclass
class RateLimiter:
    """Spaces requests evenly so a free-tier requests-per-minute limit is not hit."""

    requests_per_minute: int
    _next_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def wait(self) -> None:
        if self.requests_per_minute <= 0:
            return
        interval = 60.0 / self.requests_per_minute
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + interval
        if delay:
            time.sleep(delay)


def extract_json(text: str) -> str:
    """Return the JSON object inside a model reply (handles ```json fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]


def schema_instruction(model: type[BaseModel]) -> str:
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return "Respond with ONE JSON object only, no prose, matching this JSON schema:\n" + schema


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: bool = True,
    ) -> None:
        self.config = config
        self.model = model or config.model
        provider = config.provider_config()
        headers = {"Content-Type": "application/json"}
        key = config.api_key()
        if provider.api_key_env and not key:
            raise LLMError(
                f"{provider.api_key_env} is not set. Add it to .env "
                f"(see .env.example) or switch llm.provider to 'ollama'."
            )
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=provider.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_s,
            transport=transport,
        )
        self._limiter = RateLimiter(config.requests_per_minute if sleep else 0)
        self._sleep = sleep
        self.usage = Usage()

    def chat(self, messages: list[Message], json_mode: bool = False) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        data = self._post_with_retries(payload)
        self.usage.add(data.get("usage"))
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from exc

    def structured(self, messages: list[Message], schema: type[T]) -> T:
        """Chat and parse the reply into `schema`, with one self-repair attempt."""
        msgs = [*messages]
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + schema_instruction(schema)}
        last_error = ""
        for _ in range(2):
            try:
                reply = self.chat(msgs, json_mode=True)
            except JSONRejected as exc:
                # Provider-side JSON check failed (often an empty reply): just try again.
                last_error = str(exc)[:300]
                continue
            try:
                return schema.model_validate_json(extract_json(reply))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:500]
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": f"Invalid JSON for the schema: {last_error}. "
                        "Reply again with only the corrected JSON object.",
                    },
                ]
        raise LLMError(f"Model never returned valid {schema.__name__}: {last_error}")

    def _post_with_retries(self, payload: dict[str, object]) -> dict:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            self._limiter.wait()
            try:
                resp = self._http.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt == attempts - 1:
                    raise LLMError(f"Cannot reach LLM endpoint: {exc}") from exc
                self._backoff(attempt, None)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 400 and "json_validate_failed" in resp.text:
                raise JSONRejected(f"Provider rejected JSON output: {resp.text[:200]}")
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if not retryable or attempt == attempts - 1:
                raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
            self._backoff(attempt, resp.headers.get("retry-after"))
        raise LLMError("unreachable")  # pragma: no cover

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if not self._sleep:
            return
        try:
            delay = float(retry_after) if retry_after else 2.0 * (2**attempt)
        except ValueError:
            delay = 2.0 * (2**attempt)
        time.sleep(min(delay, 60.0))
