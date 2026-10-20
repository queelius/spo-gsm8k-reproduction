# Result and evidence

**58.192760 %** test mean pass@1 (12,281 of 21,104 sampled answers) at the fixed endpoint
of 690 collections / 11,040 optimizer updates, against 39.774450 % for the matched SFT
starting model and 56.7 % reported in the paper. The authors' released checkpoint scores
56.69 % in the same evaluator, which validates the evaluator rather than our training.

- [Final report and limitations](https://github.com/queelius/spo-gsm8k-reproduction/blob/main/docs/final-result-20261014.md)
- [Protocol audit](https://github.com/queelius/spo-gsm8k-reproduction/blob/main/docs/protocol-audit.md): what the authors' historical run used, what the public code does, and every deviation
- [Diagnosis and restart of the first attempt](https://github.com/queelius/spo-gsm8k-reproduction/blob/main/docs/diagnosis-and-restart-20261012.md)
- [Evidence receipts](https://github.com/queelius/spo-gsm8k-reproduction/tree/main/docs/evidence): one JSON file per claim, described in the walkthrough's section 10
- [Scripts](https://github.com/queelius/spo-gsm8k-reproduction/tree/main/code) and [environment lock](https://github.com/queelius/spo-gsm8k-reproduction/blob/main/env/requirements-spo-6c5f947.txt)

The trained checkpoint will be published on Hugging Face and linked here.
