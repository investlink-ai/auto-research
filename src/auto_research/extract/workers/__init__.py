"""Per-worker extraction modules — one file per worker.

Each module exposes one entry function
`extract_<worker>(*, raw_doc, doc_id, ...) -> Output | None`. The `None`
return is the quarantine signal — the caller MUST NOT persist any part of
a `None` result. See `extract.guardrails.validate_or_quarantine`.
"""
