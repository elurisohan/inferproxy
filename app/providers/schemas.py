from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    user_id: str | None = None
    tenant_id: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost_usd: Decimal | None = None
    model: str | None = None


class CompletionResponse(BaseModel):
    id: str
    content: str
    usage: Usage
    model: str
    provider: str
    cached: bool = False


class StreamChunk(BaseModel):
    delta: str | None = None
    finish_reason: str | None = None
    usage: Usage | None = None
