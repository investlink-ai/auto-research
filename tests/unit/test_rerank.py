"""Unit tests for the Qwen3-Reranker tier-selection layer.

These tests cover the explicit-config contract: tier validation, the
loud-error policy mirrors `EmbeddingAdapter`. Real-model scoring lives
in tests/live/.
"""

from __future__ import annotations

import logging

import pytest

from auto_research.extract.rerank import (
    ALLOWED_TIERS,
    RERANKER_VERSION_TAG,
    Qwen3Reranker,
    reranker_version,
)


def test_tier_allowlist_is_frozen_and_complete() -> None:
    assert isinstance(ALLOWED_TIERS, frozenset)
    assert frozenset({"dev", "deployment", "ci-cpu"}) == ALLOWED_TIERS


def test_init_logs_tier_model_device_dtype(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="auto_research.extract.rerank"):
        Qwen3Reranker(tier="ci-cpu")
    matching = [r for r in caplog.records if "reranker_init" in r.getMessage()]
    assert len(matching) == 1
    msg = matching[0].getMessage()
    assert "tier=ci-cpu" in msg
    assert "model=Qwen3-Reranker-0.6B" in msg
    assert "device=cpu" in msg
    assert "dtype=" in msg


def test_unknown_tier_raises_value_error() -> None:
    with pytest.raises(ValueError, match="tier must be"):
        Qwen3Reranker(tier="prod")  # type: ignore[arg-type]


def test_tier_to_model_mapping() -> None:
    assert Qwen3Reranker(tier="dev").model == "Qwen3-Reranker-0.6B"
    assert Qwen3Reranker(tier="deployment").model == "Qwen3-Reranker-4B"
    assert Qwen3Reranker(tier="ci-cpu").model == "Qwen3-Reranker-0.6B"


def test_tier_to_device_mapping() -> None:
    assert Qwen3Reranker(tier="dev").device == "mps"
    assert Qwen3Reranker(tier="deployment").device == "mps"
    assert Qwen3Reranker(tier="ci-cpu").device == "cpu"


def test_reranker_version_token_stable() -> None:
    r = Qwen3Reranker(tier="ci-cpu")
    assert r.reranker_version == f"ci-cpu:Qwen3-Reranker-0.6B:{RERANKER_VERSION_TAG}"
    assert reranker_version("ci-cpu", "Qwen3-Reranker-0.6B") == r.reranker_version


def test_reranker_version_distinguishes_tiers() -> None:
    # Same model but different tier (0.6B on dev/MPS vs ci-cpu/CPU) must
    # produce distinct vector-space tokens — output distributions diverge
    # by dtype and device.
    dev = Qwen3Reranker(tier="dev").reranker_version
    cpu = Qwen3Reranker(tier="ci-cpu").reranker_version
    assert dev != cpu
