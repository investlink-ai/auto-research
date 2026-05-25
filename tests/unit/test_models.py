"""Tiered model routing (per spec §7.3, Issue #10 AC).

`route_model(worker, task) -> str` is the single source of truth for
"which Claude model handles which extraction or agent task". It lives at
the package root (not under `extract/`) because the research agent and
the live critic also consume the routing table.

Unknown `(worker, task)` MUST raise — silently falling back to a default
model would produce confusing cost-attribution and quality results when
a typo in the call site (`"supplier_mappin"`) silently downgrades to
Haiku.
"""

from __future__ import annotations

import pytest

from auto_research._models import route_model


def test_route_model_returns_sonnet_for_cross_doc_reasoning() -> None:
    # Spec §7.3: cross-doc supplier mapping + Q&A evasiveness ⇒ Sonnet 4.6.
    assert route_model("ten_k", "supplier_mentions") == "claude-sonnet-4-6"
    assert route_model("transcript", "q_and_a_evasiveness") == "claude-sonnet-4-6"


def test_route_model_returns_haiku_for_pattern_recognition() -> None:
    # Spec §7.3: high-volume pattern recognition ⇒ Haiku 4.5.
    assert route_model("eight_k", "event_classification") == "claude-haiku-4-5"
    assert route_model("s_filings", "dilution_event") == "claude-haiku-4-5"


def test_route_model_returns_haiku_for_routine_extraction() -> None:
    # Spec §7.3: templated extraction ⇒ Haiku 4.5.
    assert route_model("ten_k", "guidance_tone") == "claude-haiku-4-5"
    # Contextual chunking: per-chunk one-line context is a templated,
    # high-volume rewrite — routes to Haiku 4.5.
    assert route_model("extract", "contextual_chunk") == "claude-haiku-4-5"


def test_route_model_returns_opus_for_hard_critique() -> None:
    # Spec §7.3: research agent / live critic default Sonnet, Opus for
    # hard critique moments.
    assert route_model("research_agent", "default") == "claude-sonnet-4-6"
    assert route_model("research_agent", "hard_critique") == "claude-opus-4-7"


def test_route_model_raises_on_unknown_task() -> None:
    # AC: "raises on unknown task." Silent default would mask a typo in
    # call sites that downgrades to a cheaper / weaker model unseen.
    with pytest.raises(ValueError) as exc_info:
        route_model("s_filings", "not_a_real_task")
    # Error message should pinpoint what wasn't routable.
    assert "s_filings" in str(exc_info.value)
    assert "not_a_real_task" in str(exc_info.value)


def test_route_model_raises_on_unknown_worker() -> None:
    with pytest.raises(ValueError) as exc_info:
        route_model("not_a_worker", "default")
    assert "not_a_worker" in str(exc_info.value)
