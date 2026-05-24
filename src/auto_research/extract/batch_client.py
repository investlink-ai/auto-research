"""Anthropic Batch API wrapper for backfill (Issue #48).

`make_batch_client(worker, usd_cap, ...)` returns a `BatchClient` that
submits / polls / fetches results via `anthropic.messages.batches.*`.
All responses come back with `usage.service_tier="batch"`, so
`_pricing.usd_for_message` applies the documented 50% discount
automatically — the cost log is correct without per-call branching.

### When to use this vs `make_extraction_client`

Use the **batch client** when:

- Submitting **>100 documents** in one go (full backfill, mass
  re-extraction after a `prompt_version` bump).
- Latency tolerance is **minutes to hours per batch** (Anthropic
  guarantees turnaround within 24h; typically much faster).
- Cost matters: 50% off list price across the whole batch.

Use the synchronous `make_extraction_client` (`extract/client.py`) when:

- Daily incremental extraction on **a handful of new filings**.
- The caller needs the `Message` synchronously (interactive contexts,
  worker functions that don't checkpoint).
- Latency must be **seconds**.

### Why the design diverges from the sync wrapper

The sync wrapper's `@reliable_agent_node` composes `cost_cap` →
`circuit_breaker` → `retry_with_backoff` around one call that returns
one `Message`. That doesn't fit batch shape:

- **`submit()`** returns a `MessageBatch` with no usage info — there's
  no per-call cost to accumulate yet. The reliability stack on submit
  is just `circuit_breaker(retry_with_backoff(...))`.
- **`results()`** returns N `Message`s, each with usage. Cost is
  accumulated *here*, summed across the batch. The cap is checked at
  the *next* submit; in-flight batches aren't aborted because there's
  no mechanism to do so without losing partial work.
- **`poll()`** is a status read with no cost implication; not wrapped.

The cost tracker is implemented inline as a small `_CostTracker` class
rather than reusing `cost_cap`'s decorator. Two callers (sync + batch)
isn't enough to justify pulling that out yet — rule of three says wait
for a third caller before extracting.

### Error handling

Per-individual results come back as a union (`succeeded` / `errored` /
`canceled` / `expired`). `BatchResults` splits them:

- `.succeeded: dict[custom_id, Message]` — the happy path.
- `.failed: dict[custom_id, MessageBatchIndividualResponse]` — caller
  can inspect `.result.type` and `.result.error` for diagnostics.

Silently dropping failed entries (or returning `Message | None`) would
make it impossible for the caller to distinguish "errored out" from
"never submitted" from "still pending". The dataclass surface is the
honest representation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic.types import Message
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from auto_research._models import route_model
from auto_research._pricing import usd_for_message
from auto_research.agents.reliability import (
    CostCapExceeded,
    circuit_breaker,
    retry_with_backoff,
)

# --- value objects ---------------------------------------------------------


@dataclass(frozen=True)
class BatchRequest:
    """One per-document request to be submitted as part of a batch.

    `custom_id` is the caller's link back from the individual result to
    the source document (typically the accession number or doc id).
    `system_prompt` is marked cacheable on every request — the W1
    caching policy from `extract/client.py` applies to batches too.
    """

    custom_id: str
    system_prompt: str
    user_content: str


@dataclass(frozen=True)
class BatchHandle:
    """Returned by `submit()`. Opaque to callers; passed back to
    `poll()` / `results()` / `wait()` to identify the batch.
    """

    batch_id: str
    worker: str
    task: str


@dataclass(frozen=True)
class BatchResults:
    """Result of `results()` or `wait()`.

    `succeeded` carries the happy-path `Message`s keyed by `custom_id`.
    `failed` carries the raw `MessageBatchIndividualResponse` for
    everything else — `errored`, `canceled`, or `expired` — so the
    caller can branch on `result.type` for diagnostics. Splitting them
    here means workers can write `for cid, msg in results.succeeded:`
    without first filtering out failures.
    """

    succeeded: dict[str, Message]
    failed: dict[str, MessageBatchIndividualResponse] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        return not self.failed


# --- cost tracking (inline; rule-of-two — not yet pulled into reliability) -


class _CostTracker:
    """Per-`BatchClient` running USD cost with thread-safe accumulation.

    Implemented inline rather than via `reliability.cost_cap` because the
    decorator semantics (wrap a single Message-returning call) don't fit
    batch shape (cost accrues across N messages emitted by `results()`).
    See module docstring for the rule-of-two rationale.
    """

    def __init__(self, *, usd_cap: float) -> None:
        if usd_cap <= 0:
            raise ValueError("`usd_cap` must be > 0")
        self._cap = usd_cap
        self._running = 0.0
        self._lock = threading.Lock()

    def check_or_raise(self, *, where: str) -> None:
        with self._lock:
            running = self._running
        if running > self._cap:
            raise CostCapExceeded(
                f"cost_cap exceeded on {where}: "
                f"${running:.4f} > ${self._cap:.2f}"
            )

    def add_message(self, message: Message) -> None:
        delta = usd_for_message(message)
        with self._lock:
            self._running += delta

    def running_usd(self) -> float:
        with self._lock:
            return self._running


# --- BatchClient ----------------------------------------------------------


class BatchClient:
    """Per-worker batch wrapper. Construct via `make_batch_client(...)`.

    Public surface: `submit` / `poll` / `results` / `wait` /
    `running_usd`. Internal state (the SDK handle, cost tracker, wrapped
    submit callable) is set up by the factory so each worker gets
    isolated reliability + cost budgets.
    """

    def __init__(
        self,
        *,
        worker: str,
        sdk: anthropic.Anthropic,
        cost: _CostTracker,
        submit_wrapped: Any,  # callable wrapped with reliability decorators
    ) -> None:
        self._worker = worker
        self._sdk = sdk
        self._cost = cost
        self._submit_wrapped = submit_wrapped

    def submit(self, *, task: str, requests: list[BatchRequest]) -> BatchHandle:
        """Submit a batch. Routes the model, marks system as cacheable,
        runs through circuit_breaker + retry_with_backoff, returns a
        `BatchHandle` for follow-up polling.

        Raises `CostCapExceeded` if the running USD total is already
        above the cap (in-flight work isn't aborted; the cap just
        prevents *new* spending).
        """
        # route_model raises ValueError on unknown (worker, task) — surfaces
        # the typo at the boundary, before the SDK call.
        model = route_model(self._worker, task)
        sdk_requests = [_build_sdk_request(model, r) for r in requests]
        batch = self._submit_wrapped(sdk_requests)
        return BatchHandle(batch_id=batch.id, worker=self._worker, task=task)

    def poll(self, handle: BatchHandle) -> MessageBatch:
        """One-shot status read. Not retried / circuit-broken — polling
        is a cheap GET, and a transient 5xx here just means the caller
        polls again.
        """
        return self._sdk.messages.batches.retrieve(handle.batch_id)

    def results(self, handle: BatchHandle) -> BatchResults:
        """Fetch all results (must be called after `poll()` shows
        `processing_status == "ended"`). Splits succeeded vs failed by
        the `result.type` discriminator and accumulates the per-message
        USD into the cost tracker — the cap will block the *next*
        `submit()` if the running total crossed the threshold.
        """
        succeeded: dict[str, Message] = {}
        failed: dict[str, MessageBatchIndividualResponse] = {}
        for response in self._sdk.messages.batches.results(handle.batch_id):
            if response.result.type == "succeeded":
                message = response.result.message
                succeeded[response.custom_id] = message
                self._cost.add_message(message)
            else:
                failed[response.custom_id] = response
        return BatchResults(succeeded=succeeded, failed=failed)

    def wait(
        self,
        handle: BatchHandle,
        *,
        poll_interval: float = 30.0,
        timeout: float = 3600.0,
    ) -> BatchResults:
        """Block until the batch is `ended`, then return its results.

        `poll_interval=0.0` makes tests fast without changing semantics.
        On timeout, raises `TimeoutError` — the batch is still in flight
        upstream and can be queried later via `poll()` / `results()`
        with the same handle.
        """
        deadline = time.monotonic() + timeout
        while True:
            batch = self.poll(handle)
            if batch.processing_status == "ended":
                return self.results(handle)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"batch {handle.batch_id} not ended within {timeout}s "
                    f"(last status: {batch.processing_status})"
                )
            if poll_interval > 0:
                time.sleep(poll_interval)

    def running_usd(self) -> float:
        """Current accumulated USD spend through this client. Exposed for
        introspection and tests; production code should rely on the
        cost-cap enforcement rather than reading this.
        """
        return self._cost.running_usd()


def _build_sdk_request(model: str, request: BatchRequest) -> dict[str, Any]:
    """Translate a `BatchRequest` into the SDK's per-request dict shape.

    Every request gets the same caching policy as the sync client:
    `system` is a single text block marked
    `cache_control: {"type": "ephemeral"}`. The long stable prefix is
    cached across the whole batch — same economics as the sync path,
    just amortized differently.
    """
    return {
        "custom_id": request.custom_id,
        "params": {
            "model": model,
            "max_tokens": 4096,
            "system": [
                {
                    "type": "text",
                    "text": request.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": request.user_content}],
        },
    }


def make_batch_client(
    *,
    worker: str,
    usd_cap: float = 100.00,
    failures: int = 3,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    anthropic_client: anthropic.Anthropic | None = None,
) -> BatchClient:
    """Build a per-worker batch client.

    Default `usd_cap` is intentionally an order of magnitude higher than
    the sync default because batches process many documents at once;
    the per-call accounting (sum of N message costs) accumulates faster.

    Args:
        worker: feeds `route_model(worker, task)`; tags the cost-tracker
            state for diagnostics.
        usd_cap: hard USD cap enforced before each `submit()`.
        failures: consecutive-failure threshold for `@circuit_breaker`
            on submit-side failures (not per-individual record failures
            — those go into `BatchResults.failed`).
        max_retries: additional attempts for `@retry_with_backoff` on
            429 / 5xx / transient httpx errors during submit.
        initial_wait, max_wait: exponential-jitter backoff bounds.
        anthropic_client: optional injected SDK for tests; production
            callers omit it and get a real `anthropic.Anthropic()`.
    """
    sdk = anthropic_client if anthropic_client is not None else anthropic.Anthropic()
    cost = _CostTracker(usd_cap=usd_cap)

    def raw_submit(sdk_requests: list[dict[str, Any]]) -> MessageBatch:
        # Check the cap BEFORE the network call. The cap is on already-
        # accrued spend; we never preview future cost (the batch's actual
        # cost is unknown until results land).
        cost.check_or_raise(where=f"batch_client[{worker}]")
        return sdk.messages.batches.create(requests=sdk_requests)  # type: ignore[arg-type]

    # circuit_breaker outer, retry inner — same composition as the sync
    # client's `@reliable_agent_node`, minus cost_cap (handled inline).
    submit_with_retry = retry_with_backoff(
        max_retries=max_retries,
        initial_wait=initial_wait,
        max_wait=max_wait,
    )(raw_submit)
    submit_wrapped = circuit_breaker(failures=failures)(submit_with_retry)

    return BatchClient(
        worker=worker,
        sdk=sdk,
        cost=cost,
        submit_wrapped=submit_wrapped,
    )


__all__ = [
    "BatchClient",
    "BatchHandle",
    "BatchRequest",
    "BatchResults",
    "make_batch_client",
]
