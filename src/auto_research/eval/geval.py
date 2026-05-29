"""DeepEval G-Eval metric builders for subjective Claim fields.

DeepEval is used *only* here — structured field comparison (F1 / exact /
Spearman) stays in metrics.py because the test-case API is text-based. Each
field's rubric scores whether the extracted claim's quote + categorical
confidence is a defensible reading of the source passage, judged against
the gold rationale supplied via the test case context.

The judge is an Anthropic model, not DeepEval's default OpenAI `GPTModel`:
this is an Anthropic-based project, the eval suites already gate on
`ANTHROPIC_API_KEY`, and defaulting to OpenAI would force a *second*
(`OPENAI_API_KEY`) credential for every eval run. `AnthropicModel` reads
`ANTHROPIC_API_KEY` at construction, so `build_geval_metric` must only be
called where that key is present (the @pytest.mark.eval suites) — or with a
dummy key set for construction-only unit tests."""

from __future__ import annotations

from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import SingleTurnParams

# Sonnet-tier judge: G-Eval rubric scoring needs cross-sentence reasoning,
# matching the tier the subjective fields themselves are extracted at.
_JUDGE_MODEL = "claude-sonnet-4-6"

_RUBRICS: dict[str, str] = {
    "guidance_tone": (
        "Given the source passage (context) and the gold rationale, judge "
        "whether the extracted guidance-tone claim — its quote and its "
        "high/medium/low confidence — is a defensible reading of management's "
        "forward guidance. Penalize quotes that omit the operative guidance "
        "language or confidence that overstates hedged language."
    ),
    "prepared_remarks_tone": (
        "Judge whether the extracted prepared-remarks tone claim faithfully "
        "characterizes the scripted-remarks sentiment in the source, with "
        "confidence matching how explicit the language is."
    ),
    "q_and_a_evasiveness": (
        "Judge whether the extracted Q&A-evasiveness claim correctly reflects "
        "analyst-question dodging / non-answers in the source, with confidence "
        "matching how clearly evasive the exchange is."
    ),
}

SUBJECTIVE_GEVAL_FIELDS = tuple(_RUBRICS)


def build_geval_metric(field: str, *, threshold: float) -> GEval:
    """Build a configured (not yet evaluated) G-Eval metric for `field`."""
    if field not in _RUBRICS:
        raise KeyError(f"no G-Eval rubric for field {field!r}")
    return GEval(
        name=f"{field}_quality",
        criteria=_RUBRICS[field],
        model=AnthropicModel(model=_JUDGE_MODEL),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        threshold=threshold,
    )
