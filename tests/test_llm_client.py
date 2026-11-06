import json

import httpx
import pytest
from pydantic import BaseModel

from uae_rag.config import LLMConfig, ProviderConfig
from uae_rag.llm.client import LLMClient, LLMError, extract_json


class Out(BaseModel):
    value: int


def cfg(api_key_env: str = "") -> LLMConfig:
    return LLMConfig(
        provider="local",
        model="m",
        max_retries=2,
        requests_per_minute=0,
        providers={"local": ProviderConfig(base_url="http://llm.test/v1", api_key_env=api_key_env)},
    )


def reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def client_with(responses: list[httpx.Response]) -> tuple[LLMClient, list[dict]]:
    sent: list[dict] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return queue.pop(0)

    return LLMClient(cfg(), transport=httpx.MockTransport(handler), sleep=False), sent


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"value": 1}\n```') == '{"value": 1}'
    assert extract_json('Here: {"value": 2} done') == '{"value": 2}'
    with pytest.raises(ValueError):
        extract_json("no json")


def test_structured_parses_and_counts_usage():
    client, sent = client_with([reply('{"value": 7}')])
    assert client.structured([{"role": "system", "content": "s"}], Out).value == 7
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert client.usage.calls == 1 and client.usage.prompt_tokens == 3


def test_structured_self_repairs_once():
    client, sent = client_with([reply('{"value": "oops"}'), reply('{"value": 3}')])
    assert client.structured([{"role": "system", "content": "s"}], Out).value == 3
    assert "Invalid JSON" in sent[1]["messages"][-1]["content"]


def test_retries_on_429_then_succeeds():
    client, _ = client_with([httpx.Response(429, text="slow down"), reply('{"value": 1}')])
    assert client.structured([{"role": "system", "content": "s"}], Out).value == 1


def test_non_retryable_error_raises():
    client, _ = client_with([httpx.Response(401, text="bad key")])
    with pytest.raises(LLMError, match="401"):
        client.chat([{"role": "user", "content": "hi"}])


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    with pytest.raises(LLMError, match="SOME_KEY is not set"):
        LLMClient(cfg(api_key_env="SOME_KEY"))


def test_provider_json_rejection_is_retried_and_reasoning_effort_sent():
    rejected = httpx.Response(
        400, text='{"error":{"code":"json_validate_failed","failed_generation":""}}'
    )
    sent: list[dict] = []
    queue = [rejected, reply('{"value": 5}')]

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return queue.pop(0)

    config = cfg().model_copy(update={"reasoning_effort": "low"})
    client = LLMClient(config, transport=httpx.MockTransport(handler), sleep=False)
    assert client.structured([{"role": "system", "content": "s"}], Out).value == 5
    assert len(sent) == 2 and sent[0]["reasoning_effort"] == "low"
