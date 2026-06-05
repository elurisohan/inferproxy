from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from decimal import Decimal

from app.providers.schemas import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    Usage,
)


class ProviderError(Exception):
    """Base error for LLM provider failures."""


class ProviderAPIError(ProviderError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. openai, anthropic)."""

    @property
    @abstractmethod
    def models(self) -> frozenset[str]:
        """Models this provider can serve."""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Non-streaming chat completion."""

    @abstractmethod
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Streaming chat completion as an async iterator of chunks."""

    @abstractmethod
    def estimate_cost(self, usage: Usage) -> Decimal:
        """Estimate USD cost for the given token usage."""
