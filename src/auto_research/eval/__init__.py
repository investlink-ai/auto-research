"""Extraction-quality eval harness.

Reference-based field metrics live in `metrics.py` (pure, hermetic);
LLM-judge scoring of subjective `Claim` fields lives in `geval.py`
(DeepEval). The per-worker wiring is in `registry.py`.
"""
