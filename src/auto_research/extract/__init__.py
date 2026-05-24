"""Extraction plane: per-worker LLM extraction with citation-grounded outputs.

Worker modules (`ten_k.py`, `transcript.py`, ...) land in W2; this package
provides the schemas (`schemas.py`) and the citation-grounding post-validator
+ quarantine router (`guardrails.py`) those workers compose at the
validation boundary. Both must be in place before any LLM call so INV-2
(every claim traces to verbatim source text) is mechanically enforced from
the first worker line.
"""
