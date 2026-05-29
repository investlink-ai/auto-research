"""Per-worker eval wiring.

``field_metrics`` maps a schema field to how it is scored:

- ``"claim_list"``: ``list[Claim|Mention]`` -> claim_list_f1 over source_quotes
- ``"exact"``: scalar categorical/identity-ish field -> exact_match
- ``"numeric"``: float field -> Spearman across the gold set
- ``"claim_presence"``: ``Claim | None`` -> presence + quote match

Subjective Claim fields are scored by G-Eval (see geval.py) and listed in
``subjective_fields``; identity fields (cik/accession/...) are excluded from
quality scoring but listed so the coverage test passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from auto_research.extract.prompts.eight_k import EIGHT_K_PROMPT_VERSION
from auto_research.extract.prompts.s_filings_dilution import (
    S_FILINGS_DILUTION_PROMPT_VERSION,
)
from auto_research.extract.prompts.ten_k_narrative import (
    TEN_K_NARRATIVE_PROMPT_VERSION,
)
from auto_research.extract.prompts.transcript_split import TRANSCRIPT_QA_PROMPT_VERSION
from auto_research.extract.schemas import (
    EightKOutput,
    SFilingOutput,
    TenKOutput,
    TranscriptOutput,
)
from auto_research.extract.workers.eight_k import extract_eight_k
from auto_research.extract.workers.s_filings import extract_s_filing
from auto_research.extract.workers.ten_k import extract_ten_k
from auto_research.extract.workers.transcript import extract_transcript


def _gold_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / "eval" / "gold_sets"
    raise FileNotFoundError("project root not found")


@dataclass(frozen=True)
class WorkerEvalSpec:
    worker: str
    extract_fn: Callable[..., BaseModel | None]
    output_model: type[BaseModel]
    prompt_version: str
    identity_fields: tuple[str, ...]
    subjective_fields: tuple[str, ...]
    field_metrics: dict[str, str]
    default_thresholds: dict[str, float] = field(default_factory=dict)

    @property
    def gold_path(self) -> Path:
        return _gold_root() / f"{self.worker}.jsonl"


WORKER_EVALS: dict[str, WorkerEvalSpec] = {
    "ten_k": WorkerEvalSpec(
        worker="ten_k",
        extract_fn=extract_ten_k,
        output_model=TenKOutput,
        prompt_version=TEN_K_NARRATIVE_PROMPT_VERSION,
        identity_fields=("cik", "accession_number", "fiscal_period_end"),
        subjective_fields=("guidance_tone",),
        field_metrics={
            "accrual_flags": "claim_list",
            "supplier_mentions": "claim_list",
            "customer_mentions": "claim_list",
            "risk_factor_deltas": "claim_list",
            "icfr_material_weaknesses": "claim_list",
            "critical_accounting_estimate_changes": "claim_list",
            "going_concern": "claim_presence",
            "language_novelty_score": "numeric",
        },
        default_thresholds={"min_f1": 0.6, "min_geval": 0.6},
    ),
    "transcript": WorkerEvalSpec(
        worker="transcript",
        extract_fn=extract_transcript,
        output_model=TranscriptOutput,
        prompt_version=TRANSCRIPT_QA_PROMPT_VERSION,
        identity_fields=("ticker", "event_datetime"),
        subjective_fields=("prepared_remarks_tone", "q_and_a_evasiveness"),
        field_metrics={"forward_statements": "claim_list"},
        default_thresholds={"min_f1": 0.6, "min_geval": 0.6},
    ),
    "eight_k": WorkerEvalSpec(
        worker="eight_k",
        extract_fn=extract_eight_k,
        output_model=EightKOutput,
        prompt_version=EIGHT_K_PROMPT_VERSION,
        identity_fields=("cik", "accession_number"),
        subjective_fields=(),
        field_metrics={
            "event_classification": "exact",
            "milestone_mentions": "claim_list",
            "dilution_language_flags": "claim_list",
        },
        default_thresholds={"min_f1": 0.6},
    ),
    "s_filings": WorkerEvalSpec(
        worker="s_filings",
        extract_fn=extract_s_filing,
        output_model=SFilingOutput,
        prompt_version=S_FILINGS_DILUTION_PROMPT_VERSION,
        identity_fields=("cik", "accession_number"),
        subjective_fields=(),
        field_metrics={
            "form_type": "exact",
            "dilution_event": "claim_presence",
            "capital_raise_language": "claim_list",
            "use_of_proceeds": "claim_list",
        },
        default_thresholds={"min_f1": 0.6},
    ),
}
