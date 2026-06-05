from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

from app.providers.base import LLMProvider, ProviderError
from app.providers.registry import ProviderRegistry
from app.providers.schemas import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
)


class StubProvider(LLMProvider):
    def __init__(self, name: str, models: frozenset[str]) -> None:
        self._name = name
        self._models = models

    @property
    def name(self) -> str:
        return self._name

    @property
    def models(self) -> frozenset[str]:
        return self._models

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError
        yield StreamChunk()

    def estimate_cost(self, usage: Usage) -> Decimal:
        return Decimal(usage.total_tokens)


def test_register_and_lookup_provider() -> None:
    registry = ProviderRegistry()
    openai = StubProvider("openai", frozenset({"gpt-4o-mini"}))
    registry.register(openai)

    assert registry.get_provider("openai") is openai
    assert registry.get_provider_for_model("gpt-4o-mini") is openai
    assert registry.list_providers() == ["openai"]
    assert registry.list_models() == ["gpt-4o-mini"]


def test_register_duplicate_provider_raises() -> None:
    registry = ProviderRegistry()
    registry.register(StubProvider("openai", frozenset({"gpt-4o-mini"})))

    with pytest.raises(ProviderError, match="already registered"):
        registry.register(StubProvider("openai", frozenset({"gpt-4o"})))


def test_register_duplicate_model_raises() -> None:
    registry = ProviderRegistry()
    registry.register(StubProvider("openai", frozenset({"gpt-4o-mini"})))

    with pytest.raises(ProviderError, match="already registered to provider"):
        registry.register(StubProvider("other", frozenset({"gpt-4o-mini"})))


def test_unknown_provider_raises() -> None:
    registry = ProviderRegistry()

    with pytest.raises(ProviderError, match="not registered"):
        registry.get_provider("missing")


def test_unknown_model_raises() -> None:
    registry = ProviderRegistry()

    with pytest.raises(ProviderError, match="No provider registered"):
        registry.get_provider_for_model("unknown-model")
