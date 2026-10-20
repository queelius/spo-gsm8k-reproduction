# Reproducing SPO on GSM8K with one A100

An audited reproduction of an LLM reinforcement-learning paper's result on GSM8K, done as a PhD
training-pipeline exercise (October 8 to 14, 2026).

**Result:** 58.19 percent test mean pass@1 at the fixed endpoint of 690 collections / 11,040
updates, versus 56.7 percent reported in the paper and 39.77 percent for the matched SFT baseline.
All final saved answers were independently regraded and the full checkpoint is preserved with
matching hashes. The final report with limitations is [docs/final-result-20261014.md](docs/final-result-20261014.md);
the dated progress notes under `docs/` are historical.

Layout: `docs/` (reports, protocol audit, evidence), `code/` (queue, endpoint, adapters and
tests used for the run), `slides/` (the advisor deck of 14 October 2026). The checkpoint is not in
git; it will be released on Hugging Face. This repository was split with history from the
`rlm` monorepo on 2026-10-20.
