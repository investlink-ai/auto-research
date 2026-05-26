"""Live smoke for the Qwen3-Embedding-4B MLX backend.

Hits the real `mlx-community/Qwen3-Embedding-4B-mxfp8` weights
(~8 GB once quantized, ~3-4 GB on disk). Confirms the
deployment-grade variant of the in-process MLX path loads and emits
the published native 2560-dim vectors through the same
`EmbeddingAdapter.embed → query` round-trip exercised by unit tests
against the 0.6B variant.

What this catches that unit tests + the 0.6B Mac-only test can't:

- Upstream Qwen3-4B repo drift (mlx-community renames, weight-format
  changes, quantization parameter shifts) — surfaces here before a
  backfill run hits it on a 1000x10 corpus.
- The 4B-only memory footprint: a Mac with insufficient RAM to hold
  the 4B model + LanceDB writes will fail here, not silently in a
  long backfill.

Gated by `QWEN3_FULL=1` — pulling 4B weights is 8 GB+ and not
something a routine `make live-smoke` should incur. Set the env var
explicitly to opt in:

    QWEN3_FULL=1 make live-smoke

Apple-Silicon-only by definition (the MLX backend is); skipped
cleanly on non-arm64-Darwin hosts.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytest

from auto_research.extract.chunking import ChildChunk, ChunkMetadata
from auto_research.extract.chunking_contextual import ContextualChildChunk
from auto_research.extract.embeddings import EmbeddingAdapter


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# Gate the 8 GB 4B download on QWEN3_FULL=='1' — matches the Makefile's
# `setup-mlx` shell check exactly. The live conftest's
# `live_requires_env` mechanism gates on env-var non-emptiness, so
# `QWEN3_FULL=0` would still trip it; this explicit `!= "1"` check is
# what an operator typing `QWEN3_FULL=0` reasonably expects.
@pytest.mark.skipif(
    os.environ.get("QWEN3_FULL") != "1",
    reason="QWEN3_FULL=1 required to opt into the 8 GB 4B download",
)
@pytest.mark.skipif(
    not _is_apple_silicon(),
    reason="Qwen3-MLX backend is Apple-Silicon-only",
)
def test_qwen3_mlx_4b_real_inference_smoke(live_tmpdir: Path) -> None:
    """End-to-end smoke against the 4B variant: embed → query through
    LanceDB, asserting the native 2560-dim vector shape from the
    upstream model card."""
    adapter = EmbeddingAdapter(
        backend="qwen3-mlx",
        rag_root=live_tmpdir,
        mlx_qwen3_model="Qwen3-Embedding-4B",
    )
    assert adapter.backend == "qwen3-mlx"
    assert adapter.model == "Qwen3-Embedding-4B"

    chunks = [
        ContextualChildChunk(
            child=ChildChunk(
                text=f"NVDA AI infrastructure passage {i}",
                char_span=(0, 32),
                token_count=5,
                parent_id="doc-Q4BSM:0:32",
                section_name="Item 7",
                from_table=False,
                metadata=ChunkMetadata(
                    ticker="NVDA",
                    filing_date=__import__("datetime").date(2025, 3, 15),
                    fiscal_period="FY2025",
                    doc_type="10-K",
                    doc_id="doc-Q4BSM",
                ),
            ),
            context="",
        )
        for i in range(3)
    ]
    adapter.embed(chunks)

    import lancedb

    df = lancedb.connect(live_tmpdir).open_table("doc-Q4BSM").to_pandas()
    assert len(df) == 3
    assert all(len(v) == 2560 for v in df["vector"])

    hits = adapter.query(
        "AI infrastructure", k=3, store="per_doc", doc_id="doc-Q4BSM"
    )
    assert len(hits) == 3
    assert all(h.doc_id == "doc-Q4BSM" for h in hits)
