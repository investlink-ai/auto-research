"""Validate filled 8-K labeling worksheets and merge them into
eval/gold_sets/eight_k.jsonl.

A worksheet is "filled" once its labels.json has a non-empty
``event_classification``. For each filled worksheet this script:

- checks ``event_classification`` is a real EventClassification value;
- checks every quote in ``milestone_mentions`` / ``dilution_language_flags``
  is a VERBATIM substring of that worksheet's source.txt (the gold raw_doc),
  the same grounding the runtime guardrail enforces;
- emits a gold line ``{doc_id, raw_doc, expected, subjective, rationale}``.

Validated lines are merged into the gold file, deduped by doc_id and sorted,
so re-running after labeling more worksheets is idempotent. Anything that
fails validation is reported and skipped (never written).

    uv run python scripts/gold_labeling/ingest_worksheets.py            # dry-run
    uv run python scripts/gold_labeling/ingest_worksheets.py --write     # merge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_research._io import project_root
from auto_research.extract.enums import EventClassification

_VALID_EVENTS = {e.value for e in EventClassification}
_QUOTE_FIELDS = ("milestone_mentions", "dilution_language_flags")


def _validate(doc_dir: Path) -> tuple[dict | None, list[str]]:
    """Return (gold_line, errors). gold_line is None when there are errors."""
    labels = json.loads((doc_dir / "labels.json").read_text())
    raw_doc = (doc_dir / "source.txt").read_text()
    errors: list[str] = []
    doc_id = labels.get("doc_id") or doc_dir.name

    event = labels.get("event_classification", "")
    if event not in _VALID_EVENTS:
        errors.append(f"event_classification {event!r} not in {sorted(_VALID_EVENTS)}")

    expected: dict[str, object] = {"event_classification": event}
    for field in _QUOTE_FIELDS:
        quotes = labels.get(field, [])
        if not isinstance(quotes, list):
            errors.append(f"{field} must be a list, got {type(quotes).__name__}")
            continue
        for q in quotes:
            if not isinstance(q, str) or q not in raw_doc:
                errors.append(f"{field}: quote not a verbatim substring of source.txt: {q!r:.80}")
        expected[field] = quotes

    if errors:
        return None, errors
    return {
        "doc_id": doc_id,
        "raw_doc": raw_doc,
        "expected": expected,
        "subjective": {},
        "rationale": labels.get("rationale", ""),
    }, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="merge into the gold file")
    args = ap.parse_args()

    root = project_root()
    ws_root = root / "eval" / "gold_sets" / "_worksheets" / "eight_k"
    gold_path = root / "eval" / "gold_sets" / "eight_k.jsonl"

    new_lines: dict[str, dict] = {}
    n_filled = n_ok = n_bad = 0
    for doc_dir in sorted(p for p in ws_root.iterdir() if p.is_dir()):
        labels = json.loads((doc_dir / "labels.json").read_text())
        if not labels.get("event_classification", "").strip():
            continue  # not yet labeled
        n_filled += 1
        line, errors = _validate(doc_dir)
        if line is None:
            n_bad += 1
            print(f"✗ {doc_dir.name}:")
            for e in errors:
                print(f"    - {e}")
        else:
            n_ok += 1
            new_lines[line["doc_id"]] = line
            ml, dl = len(line["expected"]["milestone_mentions"]), len(
                line["expected"]["dilution_language_flags"]
            )
            print(f"✓ {doc_dir.name}: {line['expected']['event_classification']} (m={ml}, d={dl})")

    print(f"\nfilled={n_filled}  valid={n_ok}  invalid={n_bad}")
    if not args.write:
        print("dry-run — re-run with --write to merge into the gold file.")
        return
    if n_bad:
        print("refusing to write while any worksheet is invalid; fix the above first.")
        return

    # Merge: keep existing lines, overlay validated worksheet lines by doc_id.
    merged: dict[str, dict] = {}
    if gold_path.exists():
        for ln in gold_path.read_text().splitlines():
            if ln.strip():
                obj = json.loads(ln)
                merged[obj["doc_id"]] = obj
    merged.update(new_lines)
    ordered = sorted(merged.values(), key=lambda o: o["doc_id"])
    gold_path.write_text("".join(json.dumps(o) + "\n" for o in ordered))
    print(f"wrote {len(ordered)} gold lines -> {gold_path} ({n_ok} from worksheets)")


if __name__ == "__main__":
    main()
