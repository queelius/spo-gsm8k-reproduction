# Reproducing SPO on GSM8K

One PhD student, one A100 40 GB, the authors' unmodified code, and an audited answer to
one question: does the published GSM8K result of *Segment Policy Optimization* (Guo et al.,
2025) come back when the learning stage is rerun?

| | Test mean pass@1 |
| --- | ---: |
| Authors' SFT starting model (our evaluator) | 39.77 % |
| **Our SPO run, endpoint fixed before training** | **58.19 %** |
| Paper | 56.7 % |

- **[Read the walkthrough](guide.md)**: the task, RL for language models in one picture,
  the credit-assignment problem, what SPO does, what we ran, what broke and how it was
  diagnosed, deviations, how to read the evidence, lessons, and how to reproduce it.
- [Result and evidence](result.md): the final report, the protocol audit and the receipts.
- [Slides](slides.md).
- [Repository](https://github.com/queelius/spo-gsm8k-reproduction), with `CITATION.cff`.

Written for graduate students who know machine learning but are new to reinforcement
learning for language models. Nothing here is a new method; all credit for SPO belongs
to its authors.
