from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

_telemetry_initialized = False


def _resolve_log_level(level: str) -> int:
    return getattr(logging, level.upper(), logging.INFO)


def _normalize_otlp_http_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    for suffix in ("/v1/traces", "/v1/metrics", "/v1/logs"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _otlp_http_url(base: str, signal: str) -> str:
    return f"{_normalize_otlp_http_base(base)}/v1/{signal}"


def _add_trace_context(
    _logger: structlog.types.WrappedLogger,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context.is_valid:
        event_dict["trace_id"] = format(span_context.trace_id, "032x")
        event_dict["span_id"] = format(span_context.span_id, "016x")
        event_dict["correlation_id"] = event_dict["trace_id"]
    return event_dict


def _configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _add_trace_context,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=_resolve_log_level(settings.log_level),
    )


def _build_resource() -> Resource:
    return Resource.create(
        {
            "service.name": settings.service_name,
            "deployment.environment": settings.environment,
        }
    )


def _configure_tracer_provider(resource: Resource, otlp_base: str) -> None:
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=_otlp_http_url(otlp_base, "traces"))
        )
    )
    trace.set_tracer_provider(tracer_provider)


def _configure_meter_provider(resource: Resource, otlp_base: str) -> None:
    if isinstance(metrics.get_meter_provider(), MeterProvider):
        return

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=_otlp_http_url(otlp_base, "metrics")),
        export_interval_millis=5_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)


def _instrument_libraries() -> None:
    AsyncPGInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()


def setup_telemetry(app: FastAPI) -> None:
    """Configure logging, OpenTelemetry, and library instrumentation (idempotent)."""
    global _telemetry_initialized

    if _telemetry_initialized:
        return

    otlp_base = _normalize_otlp_http_base(settings.otel_exporter_endpoint)
    resource = _build_resource()

    _configure_structlog()
    _configure_tracer_provider(resource, otlp_base)
    _configure_meter_provider(resource, otlp_base)
    _instrument_libraries()
    FastAPIInstrumentor.instrument_app(app)

    _telemetry_initialized = True
