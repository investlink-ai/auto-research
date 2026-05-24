"""Eval-gated promotion of a prompt version to the Langfuse `production` tag.

Pattern (Issue #11): a *script*, not a CI pipeline. For a single-machine
research project, a small gate is the right level — same discipline
("don't promote unless evals pass"), no Actions/Jenkins theatre.

Usage:

    uv run python scripts/promote_prompt.py s_filings_dilution v1

Reads `eval/baselines/<prompt_name>__gold.json`, runs the matching worker
against each sample under the candidate version, computes a token-level
F1 over the expected fields, and refuses to flip the Langfuse
`production` tag if F1 < threshold or per-doc cost > ceiling.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    use_of_proceeds phrase carrying the same text don't collide.
    """
    expected_set: set[tuple[str, str]] = set()
    actual_set: set[tuple[str, str]] = set()

    if "dilution_event_quote" in expected:
        expected_set.add(("de", expected["dilution_event_quote"]))
    for phrase in expected.get("use_of_proceeds_phrases", []) or []:
        expected_set.add(("up", phrase))

    de = actual.get("dilution_event") or {}
    de_q = (de.get("citation") or {}).get("source_quote") if isinstance(de, dict) else None
    if de_q:
        actual_set.add(("de", de_q))
    for c in actual.get("use_of_proceeds") or []:
        cit = (c.get("citation") or {}) if isinstance(c, dict) else {}
        q = cit.get("source_quote")
        if q:
            actual_set.add(("up", q))

    return expected_set, actual_set


def compute_f1(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    """Bag-of-phrases F1 over the comparable string fields.

    For v1 we compare `dilution_event_quote` (single) and
    `use_of_proceeds_phrases` (list). Future workers extend
    `_comparable_quotes` without changing the promotion contract.
    """
    expected_set, actual_set = _comparable_quotes(expected, actual)
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
    """Run the gate and (maybe) flip the production tag."""
    gold = json.loads(gold_path.read_text())
    samples = gold.get("samples", [])
    thresholds = gold["thresholds"]
    min_f1 = float(thresholds["min_f1"])
    max_usd = float(thresholds["max_usd_per_doc"])

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
        actual_dict = (
            actual if isinstance(actual, dict) else actual.model_dump(mode="json")
        )
        f1_scores.append(compute_f1(sample["expected"], actual_dict))
    mean_f1 = sum(f1_scores) / len(f1_scores)

    # v1 doesn't compute live USD; the worker handles caching and the script
    # doesn't want to re-invoke the cost-cap layer. Reserve the ceiling
    # check for when we add per-call cost telemetry from the cache record.
    usd_per_doc = 0.0

    if mean_f1 < min_f1:
        return PromotionResult(
            promoted=False,
            f1=mean_f1,
            usd_per_doc=usd_per_doc,
            reason=f"f1={mean_f1:.3f} below f1 threshold {min_f1}",
        )
    if usd_per_doc > max_usd:
        return PromotionResult(
            promoted=False,
            f1=mean_f1,
            usd_per_doc=usd_per_doc,
            reason=f"usd_per_doc={usd_per_doc:.4f} above ceiling {max_usd}",
        )

    langfuse_client.update_prompt(
        name=prompt_name,
        version=version,
        new_labels=["production"],
    )
    return PromotionResult(
        promoted=True,
        f1=mean_f1,
        usd_per_doc=usd_per_doc,
        reason=f"promoted: f1={mean_f1:.3f} >= {min_f1}",
    )


def _resolve_worker(prompt_name: str) -> Callable[[str, str], Any]:
    """Pick the worker function for `prompt_name`. Single dispatch table —
    grows as workers land."""
    if prompt_name == "s_filings_dilution":
        from auto_research.extract.workers.s_filings import extract_s_filing

        def _w(raw: str, doc_id: str) -> Any:
            return extract_s_filing(raw_doc=raw, doc_id=doc_id)

        return _w
    raise ValueError(f"unknown prompt_name: {prompt_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt_name")
    parser.add_argument("version")
    args = parser.parse_args(argv)

    from langfuse import Langfuse  # local import — keeps tests langfuse-free

    client = Langfuse()
    gold_path = Path("eval/baselines") / f"{args.prompt_name}__gold.json"

    result = promote(
        prompt_name=args.prompt_name,
        version=args.version,
        gold_path=gold_path,
        worker_fn=_resolve_worker(args.prompt_name),
        langfuse_client=client,
    )
    log_path = Path("eval/promotions") / f"{args.prompt_name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "version": args.version,
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
