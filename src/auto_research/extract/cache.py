"""Content-hash idempotent cache for extraction worker outputs (Issue #11).

Defends INV-6: extraction is a pure function of the full completion config,
not just the prompt version. The cache key is

    sha256(raw_doc_bytes + prompt_version + schema_version + model_id +
           canonical_json(decoding_params))

so changing the routed model, the decoding temperature, the prompt text
(via its version), or the output schema shape (via its version) all
produce a fresh key and force a fresh LLM call. The original
`(raw_doc, prompt_version)` formulation was wrong: tiered routing
(Haiku→Sonnet swap) would silently reuse stale cache.

Storage: one JSON file per cache entry at `<root>/<worker>/<sha>.json`.
Pure-hash filenames — the full provenance metadata is inside the file so
any single `cat <sha>.json` shows what produced it. Atomic writes via
`auto_research._io`.

Not in this module:

- LRU / size-based eviction. The cache is content-addressed; growth is
  bounded by `len(raw_docs) * len(distinct_completion_configs)`. For
  the four extraction workers across ~2,700 docs that's ~12K files
  steady-state — well under any reasonable disk budget.
- Async I/O. Workers are nightly batch; the marginal latency from a
  sync read is irrelevant against the seconds-per-LLM-call regime.
- A cache invalidation API. Bumping `*_PROMPT_VERSION` or `SCHEMA_VERSION`
  is the invalidation primitive — old keys simply become unreferenced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from auto_research._io import atomic_write_text

DEFAULT_CACHE_ROOT = Path("data/cache/extract")


def _canonical_json(value: Any) -> str:
    """Stable JSON serialization for hash inputs: sorted keys, no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cache_key(
    *,
    raw_doc: bytes,
    prompt_version: str,
    schema_version: str,
    model_id: str,
    decoding_params: dict[str, Any],
) -> str:
    """Compute the sha256 cache key for a completion config.

    Returns a 64-char lowercase hex string.
    """
    h = hashlib.sha256()
    h.update(raw_doc)
    h.update(b"|")
    h.update(prompt_version.encode())
    h.update(b"|")
    h.update(schema_version.encode())
    h.update(b"|")
    h.update(model_id.encode())
    h.update(b"|")
    h.update(_canonical_json(decoding_params).encode())
    return h.hexdigest()


def read(root: Path, worker: str, key: str) -> dict[str, Any] | None:
    """Return the cached payload dict, or None on miss.

    `root` is the cache root (production: `data/cache/extract/`; tests:
    `tmp_path`). `worker` namespaces entries so a `find
    data/cache/extract/s_filings/` enumerates one worker's hits.
    """
    path = root / worker / f"{key}.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record["payload"]  # type: ignore[no-any-return]


def write(root: Path, worker: str, key: str, payload: dict[str, Any]) -> None:
    """Persist `payload` keyed by `key`. Atomic — partial writes never leave
    a half-file behind on crash."""
    path = root / worker / f"{key}.json"
    record = {"key": key, "worker": worker, "payload": payload}
    atomic_write_text(path, json.dumps(record, indent=2))


__all__ = ["DEFAULT_CACHE_ROOT", "cache_key", "read", "write"]
