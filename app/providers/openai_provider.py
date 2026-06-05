from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from decimal import Decimal
from typing import Any

import httpx

from app.providers.base import LLMProvider, ProviderAPIError
from app.providers.schemas import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
)

DEFAULT_MODELS = frozenset({"gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"})

# USD per 1M tokens: (input, output)
MODEL_PRICING: Mapping[str, tuple[Decimal, Decimal]] = {
    "gpt-4o": (Decimal("2.50"), Decimal("10.00")),
    "gpt-4o-mini": (Decimal("0.15"), Decimal("0.60")),
    "gpt-3.5-turbo": (Decimal("0.50"), Decimal("1.50")),
}

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
MILLION = Decimal("1000000")


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.openai.com/v1",
        models: frozenset[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._owned_client = client is None
        self._models = models or DEFAULT_MODELS

    @property
    def name(self) -> str:
        return "openai"

    @property
    def models(self) -> frozenset[str]:
        return self._models

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request, stream=False)
        response = await self._request_with_retry(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
        )
        data = response.json()
        return self._parse_completion(data)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        payload = self._build_payload(request, stream=True)
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise ProviderAPIError(
                    self._error_message(body, response.status_code),
                    status_code=response.status_code,
                )
            async for chunk in self._parse_sse_stream(response):
                yield chunk

    def estimate_cost(self, usage: Usage) -> Decimal:
        if usage.model is None:
            msg = "Usage.model is required for OpenAI cost estimation"
            raise ProviderAPIError(msg)
        rates = MODEL_PRICING.get(usage.model)
        if rates is None:
            msg = f"No pricing configured for model {usage.model!r}"
            raise ProviderAPIError(msg)
        input_rate, output_rate = rates
        prompt_cost = Decimal(usage.prompt_tokens) * input_rate / MILLION
        completion_cost = Decimal(usage.completion_tokens) * output_rate / MILLION
        return prompt_cost + completion_cost

    async def aclose(self) -> None:
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        request: CompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump() for message in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.user_id is not None:
            payload["user"] = request.user_id
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        client = await self._get_client()
        last_response: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            response = await client.request(
                method,
                url,
                headers=self._headers(),
                **kwargs,
            )
            if response.status_code not in RETRYABLE_STATUS_CODES:
                if response.status_code >= 400:
                    raise ProviderAPIError(
                        self._error_message(response.content, response.status_code),
                        status_code=response.status_code,
                    )
                return response
            last_response = response
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
        assert last_response is not None
        raise ProviderAPIError(
            self._error_message(last_response.content, last_response.status_code),
            status_code=last_response.status_code,
        )

    def _parse_completion(self, data: dict[str, Any]) -> CompletionResponse:
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        raw_usage = data.get("usage") or {}
        model = data.get("model", "")
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            model=model,
        )
        usage = usage.model_copy(update={"cost_usd": self.estimate_cost(usage)})
        return CompletionResponse(
            id=data["id"],
            content=content,
            usage=usage,
            model=model,
            provider=self.name,
            cached=False,
        )

    async def _parse_sse_stream(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[StreamChunk]:
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ").strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            yield self._parse_stream_chunk(data)

    def _parse_stream_chunk(self, data: dict[str, Any]) -> StreamChunk:
        usage: Usage | None = None
        raw_usage = data.get("usage")
        if raw_usage is not None:
            model = data.get("model")
            usage = Usage(
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
                model=model,
            )
            usage = usage.model_copy(update={"cost_usd": self.estimate_cost(usage)})

        choices = data.get("choices") or []
        if not choices:
            return StreamChunk(usage=usage)

        choice = choices[0]
        delta_content = choice.get("delta", {}).get("content")
        return StreamChunk(
            delta=delta_content,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    def _error_message(self, body: bytes, status_code: int) -> str:
        try:
            data = json.loads(body)
            error = data.get("error", {})
            message = error.get("message")
            if message:
                return str(message)
        except json.JSONDecodeError:
            pass
        text = body.decode("utf-8", errors="replace").strip()
        if text:
            return text
        return f"OpenAI API request failed with status {status_code}"
