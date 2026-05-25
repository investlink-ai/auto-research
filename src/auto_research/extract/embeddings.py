"""Embedding adapter for the RAG retrieval layer.

Default model is Voyage `voyage-finance-2` per ADR D1
(`docs/decisions/2026-05-24-rag-enhancements.md`); falls back to local
`bge-small-en-v1.5` when `VOYAGE_API_KEY` is absent or `force_local=True`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_VOYAGE_MODELS: frozenset[str] = frozenset({
    "voyage-finance-2",
    "voyage-3",
    "voyage-3-large",
    "voyage-3.5",
    "voyage-4",
    "voyage-4-large",
})
DEFAULT_VOYAGE_MODEL = "voyage-finance-2"

_MODEL_DIM: dict[str, int] = {
    "voyage-finance-2": 1024,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "bge-small-en-v1.5": 384,
}

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallbackDecision:
    backend: str
    model: str
    reason: str  # voyage_used | no_key | quota | explicit_override


class EmbeddingAdapter:
    def __init__(
        self,
        *,
        rag_root: Path = Path("data/rag"),
        voyage_model: str | None = None,
        force_local: bool = False,
    ) -> None:
        resolved = voyage_model or os.environ.get("VOYAGE_MODEL") or DEFAULT_VOYAGE_MODEL
        if resolved not in ALLOWED_VOYAGE_MODELS:
            raise ValueError(
                f"VOYAGE_MODEL={resolved!r} not in {sorted(ALLOWED_VOYAGE_MODELS)}"
            )
        self._rag_root = rag_root
        if force_local:
            self._decision = FallbackDecision("bge", "bge-small-en-v1.5", "explicit_override")
        elif not os.environ.get("VOYAGE_API_KEY"):
            self._decision = FallbackDecision("bge", "bge-small-en-v1.5", "no_key")
        else:
            self._decision = FallbackDecision("voyage", resolved, "voyage_used")
        _log.info(
            "embedding_adapter_init backend=%s model=%s reason=%s",
            self._decision.backend,
            self._decision.model,
            self._decision.reason,
        )

    @property
    def decision(self) -> FallbackDecision:
        return self._decision
