"""Unit tests for `auto_research.extract.batch_client` (Issue #48).

What's pinned here:

- `make_batch_client` builds a per-worker `BatchClient` with isolated
  reliability state (separate cost tracker, separate circuit-breaker
  counters across two factories).
- `submit()` routes the model via `route_model(worker, task)`, marks
  every per-request `system` block as `cache_control: ephemeral`, and
  goes through `circuit_breaker(retry_with_backoff(...))`.
- `results()` splits the per-individual results into `.succeeded`
  (`dict[custom_id, Message]`) and `.failed` (`dict[custom_id,
  MessageBatchIndividualResponse]`) by the `result.type` discriminator
  — never silently drops failed entries.
- Cost accumulates *on results*, not on submit, because Anthropic
  returns usage metadata per-message only after the batch ends.
- Cost-cap blocks the NEXT `submit()` once the running total exceeds
  the cap; the in-flight batch isn't aborted (no mechanism to do so
  without losing the partial work).
- `wait()` polls until `processing_status == "ended"`, then fetches
  results. `poll_interval=0` makes tests fast.

The Anthropic client is injected via the same `cast(anthropic.Anthropic,
fake)` pattern as the sync client's tests — duck-typed structural fake
documented as a test-double cast.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message, TextBlock, Usage
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from auto_research.agents.reliability import CircuitOpen, CostCapExceeded
from auto_research.extract.batch_client import (
    BatchRequest,
    BatchResults,
    make_batch_client,
)

# --- fakes -----------------------------------------------------------------


def _make_message(
    *,
    model: str = "claude-haiku-4-5",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(type="text", text="ok", citations=None)],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation=None,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier="batch",  # batch path: 50% discount applied by `usd_for_message`
        ),
    )


def _make_batch(
    *,
    batch_id: str = "msgbatch_test",
    processing_status: str = "ended",
) -> MessageBatch:
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "type": "message_batch",
            "processing_status": processing_status,
            "request_counts": {
                "processing": 0,
                "succeeded": 1,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "ended_at": "2026-05-24T12:00:00Z" if processing_status == "ended" else None,
            "created_at": "2026-05-24T11:00:00Z",
            "expires_at": "2026-05-25T11:00:00Z",
            "cancel_initiated_at": None,
            "archived_at": None,
            "results_url": (
                f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results"
                if processing_status == "ended"
                else None
            ),
        }
    )


def _make_individual_response(
    *,
    custom_id: str,
    result_type: str = "succeeded",
    message: Message | None = None,
) -> MessageBatchIndividualResponse:
    if result_type == "succeeded":
        return MessageBatchIndividualResponse.model_validate(
            {
                "custom_id": custom_id,
                "result": {
                    "type": "succeeded",
                    "message": (message or _make_message()).model_dump(),
                },
            }
        )
    if result_type == "errored":
        return MessageBatchIndividualResponse.model_validate(
            {
                "custom_id": custom_id,
                "result": {
                    "type": "errored",
                    "error": {"type": "error", "error": {"type": "api_error", "message": "boom"}},
                },
            }
        )
    if result_type == "expired":
        return MessageBatchIndividualResponse.model_validate(
            {"custom_id": custom_id, "result": {"type": "expired"}}
        )
    raise ValueError(f"unknown result_type: {result_type}")


class _FakeBatches:
    def __init__(
        self,
        *,
        create_returns: MessageBatch | None = None,
        create_raises: Exception | None = None,
        retrieve_sequence: list[MessageBatch] | None = None,
        results_returns: list[MessageBatchIndividualResponse] | None = None,
    ) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self._create_returns = create_returns
        self._create_raises = create_raises
        self._retrieve_sequence = retrieve_sequence or []
        self._retrieve_idx = 0
        self._results_returns = results_returns or []

    def create(self, *, requests: Any) -> MessageBatch:
        self.create_calls.append({"requests": list(requests)})
        if self._create_raises is not None:
            raise self._create_raises
        return self._create_returns or _make_batch()

    def retrieve(self, batch_id: str) -> MessageBatch:
        self.retrieve_calls.append(batch_id)
        if not self._retrieve_sequence:
            return _make_batch()
        idx = min(self._retrieve_idx, len(self._retrieve_sequence) - 1)
        self._retrieve_idx += 1
        return self._retrieve_sequence[idx]

    def results(self, batch_id: str) -> list[MessageBatchIndividualResponse]:
        self.results_calls.append(batch_id)
        return self._results_returns


class _FakeMessages:
    def __init__(self, batches: _FakeBatches) -> None:
        self.batches = batches


class _FakeAnthropicClient:
    def __init__(self, batches: _FakeBatches | None = None) -> None:
        self.messages = _FakeMessages(batches or _FakeBatches())


def _as_sdk(fake: _FakeAnthropicClient) -> anthropic.Anthropic:
    """Cast a structural-fake into the SDK type for dependency injection."""
    return cast(anthropic.Anthropic, fake)


# --- submit: routing + cache_control + reliability composition -------------


def test_submit_routes_model_via_route_model() -> None:
    batches = _FakeBatches()
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    client.submit(
        task="dilution_event",
        requests=[
            BatchRequest(custom_id="r1", system_prompt="sys", user_content="doc1"),
        ],
    )
    sent = batches.create_calls[0]["requests"]
    assert sent[0]["params"]["model"] == "claude-haiku-4-5"


def test_submit_marks_each_request_system_as_ephemeral() -> None:
    batches = _FakeBatches()
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    client.submit(
        task="dilution_event",
        requests=[
            BatchRequest(custom_id="r1", system_prompt="prompt A", user_content="doc"),
            BatchRequest(custom_id="r2", system_prompt="prompt B", user_content="doc"),
        ],
    )
    sent = batches.create_calls[0]["requests"]
    # Every request in the batch gets the same caching policy as the sync client.
    for req in sent:
        system = req["params"]["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_submit_preserves_custom_id_ordering() -> None:
    """`custom_id` is the caller's link back from individual results to source
    docs. The wrapper must forward them verbatim — silently mangling them
    would orphan results from their docs.
    """
    batches = _FakeBatches()
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    client.submit(
        task="dilution_event",
        requests=[
            BatchRequest(custom_id="acc-001", system_prompt="sys", user_content="d"),
            BatchRequest(custom_id="acc-002", system_prompt="sys", user_content="d"),
            BatchRequest(custom_id="acc-003", system_prompt="sys", user_content="d"),
        ],
    )
    sent = batches.create_calls[0]["requests"]
    assert [r["custom_id"] for r in sent] == ["acc-001", "acc-002", "acc-003"]


def test_submit_returns_handle_with_batch_id() -> None:
    batches = _FakeBatches(create_returns=_make_batch(batch_id="msgbatch_xyz"))
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="dilution_event",
        requests=[BatchRequest(custom_id="r1", system_prompt="sys", user_content="doc")],
    )
    assert handle.batch_id == "msgbatch_xyz"
    assert handle.worker == "s_filings"
    assert handle.task == "dilution_event"


def test_submit_raises_on_unknown_task() -> None:
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(_FakeAnthropicClient()),
    )
    with pytest.raises(ValueError) as exc_info:
        client.submit(
            task="not_a_task",
            requests=[BatchRequest(custom_id="r1", system_prompt="s", user_content="d")],
        )
    assert "not_a_task" in str(exc_info.value)


# --- results: succeeded / failed split --------------------------------------


def test_results_splits_by_result_type_discriminator() -> None:
    msg_a = _make_message()
    msg_c = _make_message()
    batches = _FakeBatches(
        results_returns=[
            _make_individual_response(custom_id="a", result_type="succeeded", message=msg_a),
            _make_individual_response(custom_id="b", result_type="errored"),
            _make_individual_response(custom_id="c", result_type="succeeded", message=msg_c),
            _make_individual_response(custom_id="d", result_type="expired"),
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="dilution_event",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    results = client.results(handle)
    assert isinstance(results, BatchResults)
    # Successful entries map custom_id → Message; failed entries map
    # custom_id → the raw MessageBatchIndividualResponse so the caller
    # can inspect the error / type.
    assert set(results.succeeded.keys()) == {"a", "c"}
    assert isinstance(results.succeeded["a"], Message)
    assert set(results.failed.keys()) == {"b", "d"}
    assert results.failed["b"].result.type == "errored"
    assert results.failed["d"].result.type == "expired"


# --- cost: accumulates on results, not submit -------------------------------


def test_cost_accumulates_on_results_with_batch_discount() -> None:
    """1M-in + 1M-out Sonnet 4.6 = \$18 standard, \$9 batch. Three successful
    results add 3*\$9 = \$27 to the running total. The 50% discount is
    applied automatically by `usd_for_message` because the message has
    `service_tier="batch"` (the fake sets this in `_make_message`).
    """
    big_msg = _make_message(
        model="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    batches = _FakeBatches(
        results_returns=[
            _make_individual_response(custom_id=f"r{i}", message=big_msg) for i in range(3)
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="ten_k",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="supplier_mentions",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    client.results(handle)
    # 3 * 50% * (\$3 + \$15) = 3 * \$9 = \$27
    assert client.running_usd() == pytest.approx(27.0)


def test_cost_cap_blocks_next_submit_after_threshold_exceeded() -> None:
    """The cap is enforced at submit time: in-flight batches aren't aborted
    (no mechanism), but a follow-up submit raises once the running total
    crossed the cap.
    """
    big_msg = _make_message(
        model="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    batches = _FakeBatches(
        results_returns=[
            _make_individual_response(custom_id="r1", message=big_msg),
            _make_individual_response(custom_id="r2", message=big_msg),
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="ten_k",
        usd_cap=10.00,  # 2 * \$9 = \$18 will exceed
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="supplier_mentions",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    client.results(handle)
    # Running total now \$18 > \$10 cap. Next submit must raise.
    with pytest.raises(CostCapExceeded):
        client.submit(
            task="supplier_mentions",
            requests=[BatchRequest(custom_id="y", system_prompt="s", user_content="d")],
        )


def test_cost_cap_only_counts_succeeded_results() -> None:
    """Errored / canceled / expired entries don't carry usage metadata.
    A batch where everything errors should NOT advance the cost meter
    (otherwise the cap would block legitimate retries with no actual
    spend).
    """
    batches = _FakeBatches(
        results_returns=[
            _make_individual_response(custom_id="a", result_type="errored"),
            _make_individual_response(custom_id="b", result_type="expired"),
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="ten_k",
        usd_cap=10.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="supplier_mentions",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    client.results(handle)
    assert client.running_usd() == pytest.approx(0.0)


# --- reliability: circuit_breaker on submit failures -----------------------


def test_circuit_breaker_opens_on_consecutive_submit_failures() -> None:
    """The AC says circuit_breaker trips on submit-side failures (not on
    per-record results). Three consecutive create() failures must open
    the circuit so the next submit short-circuits with CircuitOpen.
    """
    batches = _FakeBatches(create_raises=RuntimeError("synthetic upstream failure"))
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        failures=2,
        max_retries=0,  # don't burn the failure budget on retries
        initial_wait=0.0,
        max_wait=0.0,
        anthropic_client=_as_sdk(fake),
    )
    for _ in range(2):
        with pytest.raises(RuntimeError):
            client.submit(
                task="dilution_event",
                requests=[BatchRequest(custom_id="r", system_prompt="s", user_content="d")],
            )
    with pytest.raises(CircuitOpen):
        client.submit(
            task="dilution_event",
            requests=[BatchRequest(custom_id="r", system_prompt="s", user_content="d")],
        )


# --- per-instance state isolation -----------------------------------------


def test_per_worker_state_is_isolated() -> None:
    """Two factories build two independent clients. Blowing one's cap
    shouldn't affect the other — separate workers carry separate budgets.
    """
    big_msg = _make_message(
        model="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    batches_ten_k = _FakeBatches(
        results_returns=[
            _make_individual_response(custom_id="r1", message=big_msg),
            _make_individual_response(custom_id="r2", message=big_msg),
        ],
    )
    batches_s_filings = _FakeBatches()
    ten_k = make_batch_client(
        worker="ten_k",
        usd_cap=10.00,
        anthropic_client=_as_sdk(_FakeAnthropicClient(batches_ten_k)),
    )
    s_filings = make_batch_client(
        worker="s_filings",
        usd_cap=10.00,
        anthropic_client=_as_sdk(_FakeAnthropicClient(batches_s_filings)),
    )
    handle = ten_k.submit(
        task="supplier_mentions",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    ten_k.results(handle)
    # ten_k now exceeds its cap.
    with pytest.raises(CostCapExceeded):
        ten_k.submit(
            task="supplier_mentions",
            requests=[BatchRequest(custom_id="y", system_prompt="s", user_content="d")],
        )
    # s_filings is unaffected.
    s_filings.submit(
        task="dilution_event",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    assert len(batches_s_filings.create_calls) == 1


# --- wait(): polls until ended ---------------------------------------------


def test_wait_polls_until_processing_status_ended() -> None:
    """wait() makes one retrieve() per poll until status is 'ended', then
    returns the results dict. Tests use `poll_interval=0.0` to avoid
    actual sleeping.
    """
    batches = _FakeBatches(
        retrieve_sequence=[
            _make_batch(processing_status="in_progress"),
            _make_batch(processing_status="in_progress"),
            _make_batch(processing_status="ended"),
        ],
        results_returns=[
            _make_individual_response(custom_id="a", message=_make_message()),
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="dilution_event",
        requests=[BatchRequest(custom_id="a", system_prompt="s", user_content="d")],
    )
    results = client.wait(handle, poll_interval=0.0)
    assert len(batches.retrieve_calls) == 3
    assert "a" in results.succeeded


def test_wait_raises_timeout_when_batch_never_ends() -> None:
    batches = _FakeBatches(
        retrieve_sequence=[
            _make_batch(processing_status="in_progress"),
        ],
    )
    fake = _FakeAnthropicClient(batches)
    client = make_batch_client(
        worker="s_filings",
        usd_cap=1000.00,
        anthropic_client=_as_sdk(fake),
    )
    handle = client.submit(
        task="dilution_event",
        requests=[BatchRequest(custom_id="x", system_prompt="s", user_content="d")],
    )
    # Tiny timeout + zero poll interval so the test runs fast.
    with pytest.raises(TimeoutError):
        client.wait(handle, poll_interval=0.0, timeout=0.0)
