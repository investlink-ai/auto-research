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

        # Encode Langfuse Basic auth into the OTLP headers. We use setdefault
        # so an explicit OTEL_EXPORTER_OTLP_HEADERS in the environment is
        # respected (covers cases where a user has additional headers).
        creds = f"{public_key}:{secret_key}".encode()
        basic = base64.b64encode(creds).decode()
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_HEADERS",
            f"Authorization=Basic {basic}",
        )

        # Import inside the function to keep import-time light for callers
        # that don't initialize telemetry (tests, CLI --help, etc.).
        from traceloop.sdk import Traceloop

        Traceloop.init(app_name=service_name, disable_batch=False)
        _INITIALIZED = True


def is_initialized() -> bool:
    """Return True iff init_telemetry has succeeded in this process."""
    return _INITIALIZED
