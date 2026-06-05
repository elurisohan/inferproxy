import json
from decimal import Decimal

import httpx
import pytest
import respx
from httpx import Response

from app.providers.base import ProviderAPIError
from app.providers.openai_provider import OpenAIProvider
from app.providers.schemas import CompletionRequest, Message, Usage

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
API_KEY = "sk-test-key"


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


@pytest.fixture
def provider(client: httpx.AsyncClient) -> OpenAIProvider:
    return OpenAIProvider(api_key=API_KEY, client=client)


@pytest.fixture
def completion_request() -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role="user", content="Hello")],
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=32,
        user_id="user-1",
        tenant_id="tenant-1",
    )


@respx.mock
@pytest.mark.asyncio
async def test_complete_returns_parsed_response(
    provider: OpenAIProvider,
    completion_request: CompletionRequest,
) -> None:
    respx.post(OPENAI_URL).mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-abc",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )
    )

    response = await provider.complete(completion_request)

    assert response.id == "chatcmpl-abc"
    assert response.content == "Hi there"
    assert response.model == "gpt-4o-mini"
    assert response.provider == "openai"
    assert response.cached is False
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15
    assert response.usage.cost_usd == Decimal("0.0000045")

    request = respx.calls.last.request
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body["stream"] is False
    assert body["user"] == "user-1"


@respx.mock
@pytest.mark.asyncio
async def test_complete_retries_on_429(
    provider: OpenAIProvider,
    completion_request: CompletionRequest,
) -> None:
    route = respx.post(OPENAI_URL)
    route.side_effect = [
        Response(429, json={"error": {"message": "rate limited"}}),
        Response(
            200,
            json={
                "id": "chatcmpl-retry",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ),
    ]

    response = await provider.complete(completion_request)

    assert response.content == "OK"
    assert len(respx.calls) == 2


@respx.mock
@pytest.mark.asyncio
async def test_complete_raises_on_api_error(
    provider: OpenAIProvider,
    completion_request: CompletionRequest,
) -> None:
    respx.post(OPENAI_URL).mock(
        return_value=Response(
            401,
            json={"error": {"message": "Invalid API key"}},
        )
    )

    with pytest.raises(ProviderAPIError, match="Invalid API key"):
        await provider.complete(completion_request)


@respx.mock
@pytest.mark.asyncio
async def test_stream_yields_sse_chunks(
    provider: OpenAIProvider,
    completion_request: CompletionRequest,
) -> None:
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10},'
        '"model":"gpt-4o-mini"}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(OPENAI_URL).mock(
        return_value=Response(
            200,
            text=sse_body,
            headers={"content-type": "text/event-stream"},
        )
    )

    chunks = [chunk async for chunk in provider.stream(completion_request)]

    assert [chunk.delta for chunk in chunks[:2]] == ["Hel", "lo"]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == 10
    assert chunks[-1].usage.cost_usd == Decimal("0.0000024")

    request = respx.calls.last.request
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_estimate_cost(provider: OpenAIProvider) -> None:
    usage = Usage(
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        total_tokens=1_500_000,
        model="gpt-4o-mini",
    )

    cost = provider.estimate_cost(usage)

    assert cost == Decimal("0.45")


def test_provider_metadata(provider: OpenAIProvider) -> None:
    assert provider.name == "openai"
    assert "gpt-4o-mini" in provider.models
