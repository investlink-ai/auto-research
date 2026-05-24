"""Eval-gated promotion of a prompt to the Langfuse `production` tag.

Pattern (Issue #11): a *script*, not a CI pipeline. For a single-machine
research project, a small gate is the right level — same discipline
("don't promote unless evals pass"), no Actions/Jenkins theatre.

Usage:

    uv run python scripts/promote_prompt.py s_filings_dilution

Reads the in-code `*_PROMPT_VERSION` constant for the named prompt, runs
the matching worker against each sample in
`eval/baselines/<prompt_name>__gold.json`, computes a bag-of-phrases F1
over the expected fields, and refuses to flip the Langfuse `production`
tag if F1 < threshold or per-doc cost > ceiling. The script does NOT
accept a candidate version on the CLI — promoting a *different* version
than the one being evaluated decouples the gate from the artifact and
defeats the purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_research.extract.prompts._registry import set_prompt_tag

# Worker implementations are late-imported inside `_resolve_worker` so this
# module stays importable by tests that pass a synthetic `worker_fn` and
# don't want to depend on the full extract stack at import time.


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    f1: float
    usd_per_doc: float
    reason: str


def _comparable_quotes(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Project both sides into a comparable set-of-tuples form.

    The tuple key is `(field, quote)` so a dilution_event and a
    use_of_proceeds phrase carrying the same text don't collide. None
    expected values are filtered (an explicit "no dilution expected"
    gold sample uses `null`, not an empty string).
    """
    expected_set: set[tuple[str, str]] = set()
    actual_set: set[tuple[str, str]] = set()

    de_expected = expected.get("dilution_event_quote")
    if isinstance(de_expected, str):
        expected_set.add(("de", de_expected))
    for phrase in expected.get("use_of_proceeds_phrases", []) or []:
        if isinstance(phrase, str):
            expected_set.add(("up", phrase))

    de = actual.get("dilution_event") or {}
    de_q = (de.get("citation") or {}).get("source_quote") if isinstance(de, dict) else None
    if isinstance(de_q, str):
        actual_set.add(("de", de_q))
    for c in actual.get("use_of_proceeds") or []:
        cit = (c.get("citation") or {}) if isinstance(c, dict) else {}
        q = cit.get("source_quote")
        if isinstance(q, str):
            actual_set.add(("up", q))

    return expected_set, actual_set


def compute_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Bag-of-phrases F1 over the comparable string fields.

    For v1 we compare `dilution_event_quote` (single) and
    `use_of_proceeds_phrases` (list). Future workers extend
    `_comparable_quotes` without changing the promotion contract.

    Empty-on-both-sides returns 1.0 (perfect agreement on an empty
    contract — a 'no novel dilution language' gold sample correctly
    extracted as empty deserves a perfect score, not zero).
    """
    expected_set, actual_set = _comparable_quotes(expected, actual)
    if not expected_set and not actual_set:
        return 1.0
    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def promote(
    *,
    prompt_name: str,
    version: str,
    gold_path: Path,
    worker_fn: Callable[[str, str], Any],
    langfuse_client: Any,
) -> PromotionResult:
    """Run the gate and (maybe) flip the production tag.

    `version` is the in-code string label (e.g. `"v1"`); the script does
    NOT accept a CLI override. Passing a label that doesn't match the
    in-code constant means the evaluated artifact and the promoted
    artifact diverge — exactly the bug class this gate exists to prevent.
    """
    gold = json.loads(gold_path.read_text())
    samples = gold.get("samples", [])
    thresholds = gold["thresholds"]
    min_f1 = float(thresholds["min_f1"])
    # `max_usd_per_doc` is intentionally Optional[float]: `null` means
    # "don't gate on cost" (the only currently-supported mode). A positive
    # value is honored only as a loud refusal — see the NOTE in the
    # eval loop below.
    raw_max_usd = thresholds.get("max_usd_per_doc")
    max_usd: float | None = float(raw_max_usd) if raw_max_usd is not None else None

    if not samples:
        return PromotionResult(
            promoted=False,
            f1=0.0,
            usd_per_doc=0.0,
            reason=f"no samples in gold set — below f1 threshold {min_f1}",
        )

    f1_scores: list[float] = []
    for sample in samples:
        actual = worker_fn(sample["raw_doc"], sample["doc_id"])
        if actual is None:
            # Worker quarantined the sample (citation grounding failed,
            # missing-quote, schema violation). Treat as zero-F1 for this
            # doc and continue the eval — don't crash the run.
            f1_scores.append(0.0)
            continue
        actual_dict = (
            actual if isinstance(actual, dict) else actual.model_dump(mode="json")
        )
        f1_scores.append(compute_f1(sample["expected"], actual_dict))
    mean_f1 = sum(f1_scores) / len(f1_scores)

    # NOTE: per-doc USD cost telemetry is not yet wired through the cache
    # layer; the gold-set's `max_usd_per_doc` field is honored only if it
    # is `null` (meaning "don't gate on cost"). Any positive value is
    # rejected loudly so an operator cannot accidentally rely on a check
    # that has no input — a previous revision of this script hardcoded
    # `usd_per_doc = 0.0`, making the ceiling check dead code that passed
    # silently regardless of actual spend. Wiring cost via cache-record
    # metadata is tracked as follow-up work.
    if max_usd is not None:
        return PromotionResult(
            promoted=False,
            f1=mean_f1,
            usd_per_doc=0.0,
            reason=(
                "cost gating not yet implemented; set "
                "thresholds.max_usd_per_doc to null to bypass"
            ),
        )

    if mean_f1 < min_f1:
        return PromotionResult(
            promoted=False,
            f1=mean_f1,
            usd_per_doc=0.0,
            reason=f"f1={mean_f1:.3f} below f1 threshold {min_f1}",
        )

    set_prompt_tag(
        name=prompt_name,
        version=version,
        tag="production",
        client=langfuse_client,
    )
    return PromotionResult(
        promoted=True,
        f1=mean_f1,
        usd_per_doc=0.0,
        reason=f"promoted: f1={mean_f1:.3f} >= {min_f1}",
    )


def _resolve_worker_and_version(prompt_name: str) -> tuple[Callable[[str, str], Any], str]:
    """Pick the worker function and read its in-code version constant.

    Returns both so `main()` can pass the SAME `version` string to the
    promote() / set_prompt_tag() flow that the worker is actually
    running — eliminating the operator-error mode where the CLI version
    and the in-code constant could diverge.
    """
    if prompt_name == "s_filings_dilution":
        from auto_research.extract.prompts.s_filings_dilution import (
            S_FILINGS_DILUTION_PROMPT_VERSION,
        )
        from auto_research.extract.workers.s_filings import extract_s_filing

        def _w(raw: str, doc_id: str) -> Any:
            return extract_s_filing(raw_doc=raw, doc_id=doc_id)

        return _w, S_FILINGS_DILUTION_PROMPT_VERSION
    raise ValueError(f"unknown prompt_name: {prompt_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_name")
    args = parser.parse_args(argv)

    from langfuse import Langfuse  # local import — keeps tests langfuse-free

    client = Langfuse()
    gold_path = Path("eval/baselines") / f"{args.prompt_name}__gold.json"
    worker_fn, version = _resolve_worker_and_version(args.prompt_name)

    result = promote(
        prompt_name=args.prompt_name,
        version=version,
        gold_path=gold_path,
        worker_fn=worker_fn,
        langfuse_client=client,
    )
    log_path = Path("eval/promotions") / f"{args.prompt_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "version": version,
                    "promoted": result.promoted,
                    "f1": result.f1,
                    "usd_per_doc": result.usd_per_doc,
                    "reason": result.reason,
                }
            )
            + "\n"
        )
    print(result.reason)
    return 0 if result.promoted else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["PromotionResult", "compute_f1", "promote"]
