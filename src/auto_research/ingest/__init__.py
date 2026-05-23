"""Ingest plane — fetches raw documents from external sources.

See `docs/specs/2026-05-22-design.md` §6.1 for the source list (EDGAR,
FMP) and §6.2 for the raw-document store layout. The append-only Parquet
ledger in `manifest` is the idempotency boundary: re-running a fetcher
with content already in the manifest is a no-op.
"""

from __future__ import annotations
