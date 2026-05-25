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
import sys
import threading
from typing import Final

_INIT_LOCK: Final[threading.Lock] = threading.Lock()
_INITIALIZED: bool = False
_TRY_INIT_WARNED: bool = False

# Cap for exception strings shipped in OTel span.status descriptions.
# OTLP / Langfuse / gRPC all impose practical limits, and OTel does NOT
# truncate Status.description like it does span attributes — a multi-MB
# anti-bot HTML body in a yt-dlp ExtractorError would otherwise ride
# along on every error span. 512 chars is enough to identify the failure
# class without bloating the payload.
_STATUS_DESCRIPTION_MAX_CHARS: Final = 512


def truncate_status_description(message: str) -> str:
    """Cap a span.status description so huge exception payloads (anti-bot
    HTML, raw response bodies) don't ride along on every error span.

    Returns the input unchanged if under the cap, otherwise the prefix
    plus a `...[truncated N more chars]` suffix so reviewers know data
    was elided.
    """
    if len(message) <= _STATUS_DESCRIPTION_MAX_CHARS:
        return message
    remaining = len(message) - _STATUS_DESCRIPTION_MAX_CHARS
    return f"{message[:_STATUS_DESCRIPTION_MAX_CHARS]}...[truncated {remaining} more chars]"


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


def try_init_telemetry(*, service_name: str = "auto-research") -> bool:
    """Best-effort `init_telemetry()` for CLI / interactive entry points.

    Returns True iff telemetry is initialized (either now or already).
    Any failure — missing env (`TelemetryNotConfiguredError`), unreachable
    OTLP endpoint, missing/broken `traceloop-sdk` install, malformed URL,
    etc. — is caught, a single one-line warning is printed to stderr, and
    False is returned. The CLI must remain usable without a running
    Langfuse; a misconfigured exporter must not abort the user's work.

    The warning is deduplicated per process via `_TRY_INIT_WARNED`;
    re-running a command does not spam stderr.

    Tests and the integration smoke continue to call `init_telemetry`
    directly so a misconfigured environment fails loud, not silently.
    """
    global _TRY_INIT_WARNED
    try:
        init_telemetry(service_name=service_name)
    except TelemetryNotConfiguredError as exc:
        if not _TRY_INIT_WARNED:
            print(f"warn: telemetry disabled - {exc}", file=sys.stderr)
            _TRY_INIT_WARNED = True
        return False
    except Exception as exc:
        # Broad catch is deliberate: any failure to initialize Traceloop
        # (DNS, ImportError, version drift, malformed endpoint) is a
        # config problem on the operator's side, not a reason to block
        # ingest/extract from running. Surface it on stderr once and
        # carry on with spans as no-ops.
        if not _TRY_INIT_WARNED:
            print(
                f"warn: telemetry init failed ({exc.__class__.__name__}): {exc}",
                file=sys.stderr,
            )
            _TRY_INIT_WARNED = True
        return False
    return True
