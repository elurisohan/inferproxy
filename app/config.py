from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    database_url: str = Field(
        ...,
        description="PostgreSQL connection URL (asyncpg driver).",
    )
    redis_url: str = Field(
        ...,
        description="Redis connection URL for caching and rate limiting.",
    )
    kafka_brokers: str = Field(
        ...,
        description="Comma-separated Kafka broker addresses.",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key; required when routing to OpenAI models.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key; required when routing to Claude models.",
    )
    groq_api_key: str | None = Field(
        default=None,
        description="Groq API key; required when routing to Groq models.",
    )

    service_name: str = Field(
        default="inferproxy",
        description="Service identifier used in logs, metrics, and traces.",
    )
    environment: str = Field(
        ...,
        description="Deployment environment (e.g. local, staging, production).",
    )
    log_level: str = Field(
        ...,
        description="Logging verbosity (e.g. DEBUG, INFO, WARNING, ERROR).",
    )
    otel_exporter_endpoint: str = Field(
        ...,
        description="OpenTelemetry OTLP exporter endpoint (gRPC or HTTP).",
    )


settings = Settings()
