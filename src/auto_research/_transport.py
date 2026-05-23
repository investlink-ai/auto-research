"""Cross-package transport-layer primitives.

Today's only export is the set of httpx exception types that signal a
*transient* transport-layer failure worth retrying. Originally lived in
`auto_research.ingest._http` and was duplicated into
`auto_research.agents.reliability`; promoted here so both callers — and
any future HTTP-touching module (e.g., Voyage embeddings client in W2)
— share one list.

Scope is deliberately tiny. Anything source-specific (per-source
`RateLimited`/`ServerError` subclasses, `Retry-After` parsing,
`atomic_write_bytes`) stays in `ingest/_http.py` where its catch-site
ergonomics make sense.
"""

from __future__ import annotations

import httpx

# httpx-level transient errors that warrant a retry. ReadError/WriteError
# covers connection resets; ConnectError covers DNS / TCP problems;
# TimeoutException covers all of httpx's timeout subclasses;
# RemoteProtocolError covers truncated/garbled responses.
TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
)

__all__ = ["TRANSIENT_NETWORK_ERRORS"]
