"""OpenTelemetry / OpenLLMetry initialization for Langfuse export.

Call `init_telemetry()` once at process start; the function is idempotent
and safe to call from multiple entry points. Auto-instruments Anthropic
SDK calls via Traceloop (OpenLLMetry); spans + token counts ship to
Langfuse via OTLP using Basic auth derived from the Langfuse keys.

Required environment (see `.env.example`):
    OTEL_EXPORTER_OTLP_ENDPOINT   e.g. http://localhost:3000/api/public/otel
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
"""

from __future__ import annotations

import base64
import os
import threading
from typing import Final

_INIT_LOCK: Final[threading.Lock] = threading.Lock()
_INITIALIZED: bool = False


class TelemetryNotConfiguredError(RuntimeError):
    """Raised when required env vars are missing at telemetry init time."""


def init_telemetry(*, service_name: str = "auto-research") -> None:
    """Wire OpenLLMetry → Langfuse OTLP. Idempotent across calls and threads."""
    global _INITIALIZED
    with _INIT_LOCK:
        if _INITIALIZED:
            return

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        missing = [
            name
            for name, value in [
                ("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint),
                ("LANGFUSE_PUBLIC_KEY", public_key),
                ("LANGFUSE_SECRET_KEY", secret_key),
            ]
            if not value
        ]
        if missing:
            raise TelemetryNotConfiguredError(
                "Telemetry env vars missing: "
                + ", ".join(missing)
                + ". Copy .env.example to .env, start Langfuse via "
                "`docker compose up -d`, sign in at http://localhost:3000, "
                "create a project, and paste the public/secret keys."
            )

        # Compute Langfuse Basic auth.
        creds = f"{public_key}:{secret_key}".encode()
        basic = base64.b64encode(creds).decode()

        # Pass endpoint + headers explicitly. The traceloop-sdk reads
        # TRACELOOP_BASE_URL / TRACELOOP_HEADERS env vars and does NOT
        # honor the OTel standard OTEL_EXPORTER_OTLP_* vars — relying on
        # env propagation would silently misroute spans to api.traceloop.com.
        # `endpoint_is_traceloop=False` disables the SDK's hosted-mode
        # behavior (telemetry config sync, dashboards URL, etc.).
        from traceloop.sdk import Traceloop

        Traceloop.init(
            app_name=service_name,
            api_endpoint=endpoint,
            headers={"Authorization": f"Basic {basic}"},
            disable_batch=False,
            endpoint_is_traceloop=False,
        )
        _INITIALIZED = True


def is_initialized() -> bool:
    """Return True iff init_telemetry has succeeded in this process."""
    return _INITIALIZED
