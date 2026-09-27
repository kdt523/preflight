"""OpenTelemetry tracing. Active only when OTEL_EXPORTER_OTLP_ENDPOINT is set (local Jaeger in
compose, an OTLP collector on AWS); otherwise a no-op, so tests and scripts stay quiet."""

import os

from fastapi import FastAPI


def instrument(app: FastAPI, service_name: str) -> None:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")


def tracer(name: str):
    from opentelemetry import trace

    return trace.get_tracer(name)
