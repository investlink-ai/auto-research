# learning/

Long-form walkthroughs of non-obvious work in this repo. Written
explicitly as teaching material — the *why* behind each decision,
the categories of mistake a code review caught, and the broader
patterns to recognize in future work.

Files are date-prefixed (`YYYY-MM-DD-<topic>.md`) so chronological
order is the read order. Each one stands alone — no required
prereqs beyond `AGENTS.md` §2 (invariants) and `docs/ARCHITECTURE.md`.

Not part of the test gate. Not in `docs/` because `docs/` is for
specifications / contracts / current behavior; `learning/` is for
narrative explanation.

## Index

- [`2026-05-25-chunking-walkthrough.md`](./2026-05-25-chunking-walkthrough.md) — design walkthrough of `extract/chunking.py`: parent-document retrieval, INV-2 enforcement, the 16 code-review findings as a learning catalog, and interview-ready talking points.
