# Reproducing Segment Policy Optimization (SPO) on GSM8K with one A100

An audited reproduction of the GSM8K result of **Segment Policy Optimization** (SPO), done as a
PhD training-pipeline exercise on one A100 40GB between 8 and 14 October 2026. The method,
code, data and baseline models are the original authors'; this repository adds the protocol
audit, the queue and endpoint scripts used to run and evaluate their code unattended, and the
evidence receipts. Nothing here claims novelty.

## The paper and code being reproduced

- **Paper:** Yiran Guo, Lijie Xu, Jie Liu, Dan Ye, Shuang Qiu. *Segment Policy Optimization:
  Effective Segment-Level Credit Assignment in RL for Large Language Models.* arXiv:2505.23564,
  2025. https://arxiv.org/abs/2505.23564
- **Official code:** https://github.com/AIFrameResearch/SPO, commit
  `6c5f94723c20e4de68578545b8bb98186e956797` (2025-09-19), MIT license (the code base derives from
  McGill NLP's VinePPO, https://github.com/McGill-NLP/VinePPO, also MIT). The clone was used
  unmodified; our scripts drive it from outside.
- **Models:** SFT input policy `realtreetune/rho-1b-sft-GSM8K` (Hugging Face revision
  `b28fda3216f8178c1f37c3c764196e3ca2439a30`); the authors' released SPO policy
  `gyr66/spo-chain-int5-rho1.1B-gsm8k` (revision `a806ef8ef83beca009725da026fb7bde467a9dde`) was
  evaluated as a reference.
- **Data:** GSM8K (Cobbe et al., 2021), https://github.com/openai/grade-school-math, MIT; 7,100
  train, 373 validation, 1,319 test as split by the authors' configs (no question overlap across
  splits; see the protocol audit).

```bibtex
@misc{guo2025segmentpolicyoptimizationeffective,
  title={Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models},
  author={Yiran Guo and Lijie Xu and Jie Liu and Dan Ye and Shuang Qiu},
  year={2025}, eprint={2505.23564}, archivePrefix={arXiv}, primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.23564}
}
```

## Result

| | Test mean pass@1 |
| --- | ---: |
| SFT baseline (matched, our evaluation) | 39.77 % |
| **Our SPO run, fixed endpoint (690 collections / 11,040 updates)** | **58.19 %** |
| Paper | 56.7 % |

The endpoint was fixed before training (not the best intermediate checkpoint). All final saved
answers were regraded independently of the training code; the full checkpoint is preserved with a
fourteen-file hash receipt (`docs/evidence/final11040-preservation-20261014.json`). Report and
limitations: [docs/final-result-20261014.md](docs/final-result-20261014.md). Known deviations from
the paper's stated settings (for example the effective top-p of the historical evaluation pipeline)
are listed in [docs/protocol-audit.md](docs/protocol-audit.md). A first attempt was diagnosed and
restarted on 12 October ([docs/diagnosis-and-restart-20261012.md](docs/diagnosis-and-restart-20261012.md)).

## Slides and learning materials

- **Tutorial deck** (14 slides, for everyone): [slides/tutorial.pdf](slides/tutorial.pdf).
- Advisor deck of 14 October 2026, revised 20 October: [slides/research-update.pdf](slides/research-update.pdf)
  (sources, presenter notes and evidence links in [slides/](slides/README.md)).
- **Pedagogical walkthrough** of the paper, the method and this reproduction:
  https://queelius.github.io/spo-gsm8k-reproduction/ (source in `site-docs/guide.md`).

## Reproducing this reproduction

- Hardware: one NVIDIA A100-PCIE-40GB (MIG 7g.40gb), driver 580.
- Environment: Python 3.10.19 with the package list in
  [env/requirements-spo-6c5f947.txt](env/requirements-spo-6c5f947.txt) (torch 2.1.2, vLLM
  0.4.0.post1, DeepSpeed 0.14.1, transformers 4.38.1), created with `uv` from the official
  repository's `requirements.txt` at the commit above.
- Scripts: [code/README.md](code/README.md) describes the queue (`code/run_queue.py`), the fixed
  endpoint (`code/fixed_endpoint.py`), the BOS-only adapter and launch script, and their tests.
  The exact author commands and configs we invoked are recorded as JSON in `docs/evidence/`
  (`author-reference-command.json`, `author-training-config.json`, and so on).
- The trained checkpoint (weights and tokenizer of the audited final state) is on Hugging Face:
  https://huggingface.co/queelius/spo-chain-rho1.1b-gsm8k-reproduction (with a licensing caveat:
  the upstream SFT model declares no license).

## Credit and citation

All credit for the method and implementation belongs to the SPO authors and to the VinePPO
authors whose code base they built on. If you use this reproduction, cite the paper above and
this repository (see `CITATION.cff`).
