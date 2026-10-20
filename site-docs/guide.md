# Reproducing SPO on GSM8K: a walkthrough

File names in code font refer to files in [the repository](https://github.com/queelius/spo-gsm8k-reproduction); evidence receipts live under `docs/evidence/`.

A guide to this repository's reproduction of
Guo, Xu, Liu, Ye and Qiu, *Segment Policy Optimization* (arXiv:2505.23564). Written for
graduate students who know ML but are new to RL for language models.

Source note. The paper PDF was not consulted when writing this guide. The method
description below comes from the official README at commit
`6c5f94723c20e4de68578545b8bb98186e956797`, from the official source at that commit, and
from the reproduction's `docs/protocol-audit.md`. Where the paper might differ, this guide
says so.

## 1. Start here

What this is. Between 8 and 14 October 2026 one PhD student ran the authors' SPO code,
unmodified, on one A100 40 GB to retrain the authors' GSM8K starting model and check
whether the published score comes back. The repository holds the protocol audit, the small
scripts that drove the authors' code, and JSON receipts for every claim.

What it is not. Not a new method, not a multi-seed study, and not a bit-for-bit replay of
the authors' historical run (their exact source revision is unknown).

Result in one paragraph (`README.md`, `docs/final-result-20261014.md`). The authors' SFT
starting model scores 39.77 % mean pass@1 on the GSM8K test set in our evaluator
(8,394/21,104). After 690 collections / 11,040 optimizer updates, fixed as the endpoint
before training, our own SPO run scores 58.19 % (12,281/21,104). The paper reports
56.7 %. The authors' released SPO checkpoint scores 56.69 % in the same evaluator, which
validates the evaluator, not our training. A first attempt collapsed to 22.19 % and
crashed; its diagnosis led to the two adapters and a restart.

## 2. The task: GSM8K

GSM8K (Cobbe et al., 2021) is a set of grade-school math word problems with worked
solutions. The authors' archive splits it 7,100 train / 373 validation / 1,319 test; the
validation split is carved from the original 7,473 training rows, and the audit found zero
question overlap between any pair of splits (`docs/evidence/dataset-provenance.json`).

First test question, as stored in the authors' Arrow file:

> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes
> muffins for her friends every day with four. She sells the remainder at the farmers'
> market daily for \$2 per fresh duck egg. How much in dollars does she make every day at
> the farmers' market?

Gold answer field:

```
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.
#### 18
```

The policy sees `[MATH_TASK] Problem:\n{question}\n\nSolution:` and must write a solution
ending in `\n#### <answer>`. Generation stops at `\n\n\nProblem:` or 1,024 new tokens.
Grading (`src/treetune/tasks/gsm8k.py`): take the text after the last `####`, strip it,
remove commas, compare case-insensitively as a string to the gold answer. No `####` means
wrong. This is exact string matching, not a numeric checker: `1600` against gold `1,600`
scores zero, because commas are stripped from the prediction only
(`docs/evidence/restart-reward-formatting247-20261013.json`).

## 3. RL for language models in one picture

Sample answers, score them with a verifiable reward, update the policy. One collection in
this recipe: draw 64 training questions, sample 8 answers each (temperature 0.6, top-p
0.9), score them, then make two passes over the 512 episodes in batches of 64. That is 16
optimizer updates per collection.

Policy. The language model $\pi_\theta(y \mid x)$, the probability of answer $y$ given
prompt $x$, factored over tokens $\pi_\theta(y_t \mid x, y_{<t})$.

Reward. A number computed from the final answer by a checker, not a learned model. Here
$r = 1$ for a correct `####` answer, $0$ for a wrong one, $-1$ for no marker or a
length-limited answer, and $-2$ for more than one marker (historical config in
`docs/evidence/author-training-config.json`; `math_episode_generator.py:53-56`).

Advantage. How much better an action was than expected: value achieved minus value
expected. Positive advantage raises the probability of the tokens that earned it. The
paper's whole question is how to compute an advantage for each token.

PPO-style clipping (Schulman et al., 2017). Let
$\rho_t = \pi_\theta(y_t \mid \cdot) / \pi_{\text{old}}(y_t \mid \cdot)$, the ratio
between the current policy and the policy that generated the sample. The per-token loss is
$-\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)$ with
$\epsilon = 0.2$. Clipping removes the incentive to move far from the sampler in one step.
A KL penalty (coefficient $10^{-4}$) to the frozen starting model is added. If a
minibatch's mean ratio exceeds 10, the code multiplies its loss by zero
(`ppo_trainer.py`, `ratio_threshold`). Remember that guard; it matters in Section 8.

## 4. The credit-assignment problem

An answer is 100 to 300 tokens. The checker returns one number at the end. Which tokens
deserve the credit or blame? Three positions:

- Trajectory level. Every token gets the same advantage. GRPO (Shao et al., 2024) does
  this: within a question's group of 8 answers, $A = (r - \bar r)/(\sigma_r + 10^{-8})$,
  copied to every response token (SPO repo,
  `math_episode_generator_with_group_advantages.py:266-280`, `424-431`). Cheap and
  critic-free, but a wrong answer with a good first half gets no internal distinction,
  and if all 8 answers agree the advantage is zero.
- Token level. PPO with a learned critic predicts a value at every token. Most
  informative in principle, but the critic is a second model trained on noisy targets;
  the official README calls its estimates "often inaccurate".
- Segment level. Cut the answer into a few segments, estimate a value at each cut by
  sampling continuations, and give each segment one advantage. This is SPO. VinePPO
  (Kazemnejad et al., 2024) is here too: Monte Carlo values at every reasoning step
  (each line), no critic. SPO's code is a fork of VinePPO's; SPO differs in where it cuts
  (by token probability, not by line) and in using fewer cuts.

The README's framing: "Trajectory-level methods such as GRPO rely solely on sparse final
rewards... Token-level methods like PPO heavily depend on the critic model... SPO ...
operating at the segment granularity".

## 5. What SPO does (chain variant, interval 5)

The README names three components: segment partition, segment advantage estimation,
policy optimization. For short chain-of-thought the instance is SPO-chain:
"probability-guided segment partition, chain-based segment advantage estimation, and
policy optimization via policy gradient with probability masks". Here is what the code at
the pinned commit does.

Partition (`math_episode_generator_with_mc_advantages.py:920-931`). After sampling an
answer, take the sampling policy's probability $p_t$ of each response token. Mark every
position with $p_t < 0.9$, where the model was unsure. Shift each mark back one token so
the cut sits just before the uncertain token. Keep every 5th mark; that is the interval.
The kept positions $c_1 < \dots < c_K$ are the cutpoints. In the fresh run's first
collection, 512 answers averaged 3.16 cutpoints (max 18)
(`docs/evidence/startup-audit-20261012.json`).

Value estimates (`_compute_mc_value`, `_create_cutpoint_query`). For each cutpoint, build
the prefix $x \oplus y_{\le c_k}$ and sample $N = 9$ continuations $z_1 \dots z_9$ from
the same policy with the training sampling settings. The value is the mean reward:

$$V(c_k) = \frac{1}{N}\sum_{j=1}^{N} r\big(x,\ y_{\le c_k} \oplus z_j\big).$$

$V(c_k)$ estimates "how likely is a correct final answer given the answer so far". No
critic; the estimate is unbiased for the current policy. Cost: in that first collection,
1,620 cutpoint states produced 14,580 continuation records on top of the 512 answers
(1,080 unique prefixes after de-duplication, so fewer actual generation calls).

Segment advantages (`_compute_token_advantages:394-440`). Walk the cutpoints in order:

$$A_k = V(c_k) - V(c_{k-1}) \quad (k \ge 2), \qquad A_{K+1} = r(x,y) - V(c_K).$$

$A_k$ is the change in success probability produced by the tokens between cut $k-1$ and
cut $k$; every token in that segment receives $A_k$. In the inspected source the request
for a value at the empty prefix is commented out, so the first segment gets $A_1 = 0$, and
an answer with no cutpoint gets zero advantage everywhere. Whether the paper defines
$A_1$ this way could not be checked without the PDF.

Probability mask (`_compute_token_advantages:437-439`, `_compute_actor_loss`,
`masked_whiten`). Tokens with $p_t > 0.9$ get advantage zero and are excluded from the
loss mask and from whitening. Confident tokens are not treated as decisions; credit lands
on the uncertain ones.

Policy update. Advantages are whitened (mean 0, std 1) over the masked tokens of the
microbatch, then the PPO-clip loss of Section 3 is applied on the masked tokens, plus the
KL term. Symbols: $x$ prompt; $y$ sampled answer; $y_{\le c}$ its first $c+1$ tokens;
$z_j$ a sampled continuation; $r$ the checker reward; $p_t$ the old policy's probability
of token $t$; $\rho_t$ the PPO ratio.

## 6. What the paper reports for GSM8K with RhoMath 1.1B SFT

The official README's Figure 3 (`figures/compare-baselines-short-CoT.png`, "GSM8K Test
Accuracy (RhoMath 1.1B)") shows: Init SFT 40.3, RestEM 42.8, RLOO 44.5, GRPO 44.6, DPO+
46.4, PPO 50.1, VinePPO 53.4, SPO-tree 56.4, SPO-chain 56.7. README caption: "SPO-chain
achieves the highest accuracy on the GSM8K test set, outperforming PPO and GRPO by 6-12
percentage points."

The protocol audit traced the 56.7 to the authors' W&B evaluation run `ytu1arac`:
`0.5672384382107657` at checkpoint `ckpt--iter_0690--epoch_2.00--step_11040`. It also
notes, from the paper's Figure 3 caption, that the baseline bars are inherited from
VinePPO rather than rerun (`docs/grpo-control-plan.md`). A later author rerun of all 1,000
iterations (`yc23440y`) logged a last test pass@1 of 57.16 % and best 57.42 % over
101.93 h; the original two training segments took 59.86 h.

## 7. What we ran

Starting model. `realtreetune/rho-1b-sft-GSM8K`, HF revision
`b28fda3216f8178c1f37c3c764196e3ca2439a30`. Reference policy for scoring the authors'
result: `gyr66/spo-chain-int5-rho1.1B-gsm8k`, revision
`a806ef8ef83beca009725da026fb7bde467a9dde`.

Data. The authors' own archive (`wandb_export_root.zip`, hashes in
`dataset-provenance.json`), 7,100/373/1,319.

Hardware and environment. One NVIDIA A100-PCIE-40GB (MIG 7g.40gb), driver 580. Python
3.10.19; torch 2.1.2+cu121; the authors' custom vLLM 0.4.0.post1 wheel; DeepSpeed 0.14.1;
transformers 4.38.1; FlashAttention 2.5.5 (`docs/evidence/environment.md`,
`environment-lock.txt`).

Recipe. The audit found the README's current defaults differ from the historical run, so
training used the config recovered from W&B run `rbtvnfqm`
(`docs/evidence/author-training-config.json`) with only paths changed; the differences
(KL clip, unfinished-response penalty, microbatch size) are tabulated in
`docs/protocol-audit.md`. Settings: 64 questions × 8 responses; 9 MC continuations per cutpoint; interval 5,
threshold 0.9; train temperature 0.6 / top-p 0.9; max 1,024 generated tokens; 2 epochs
per collection; global batch 64; AdamW, LR 1e-6, warmup ratio 0.03; KL coefficient 1e-4;
clip 0.2; $\gamma=\lambda=1$; seed 42; bf16. `num_iterations` stayed at 1,000 because it
sets the WarmupDecayLR horizon (16,000 planned updates, 480 warmup);
`early_stop_iteration: 690` stops the loop without changing the schedule. The only
resource change was vLLM memory utilization 0.3 instead of auto.

The fixed endpoint, and why it was fixed before training. The audit established that the
paper's number is checkpoint 690 of a 1,000-iteration schedule, that the authors'
validation maximum lies near 520, and that nothing records why 690 was reported
(`docs/protocol-audit.md`). Choosing our own best test checkpoint would make the
comparison meaningless. So 690 / 11,040 was declared the comparison point prospectively,
and `code/fixed_endpoint.py` refuses to evaluate unless the saved trainer, actor,
optimizer and scheduler counters read exactly 690 / 11,040 / 11,039.

Evaluation protocol. 1,319 test questions × 16 samples = 21,104 answers; temperature
0.35; effective top-p 1.0; max 1,024 new tokens; context 2,047; seed 42; stop
`\n\n\nProblem:`. Metric: `correct_frac`, the mean over questions of the fraction of the
16 samples that are correct, which equals total correct / 21,104. This is mean pass@1. It
is not `once_hit` (at least one of 16 correct) and not majority vote. The loop also evaluates the 373 validation questions (5,968 answers) and a
372-question training subset (5,952 answers, not held out) after iterations 0, 10, 20,
..., so "iteration 0" is already 16 updates in; the SFT model was evaluated separately.

Two different checks. (1) Score the authors' released checkpoint with our pipeline:
56.68594 % versus their log's 56.72384 %, a 0.038-point gap, in about seven minutes of
GPU time (`author-reference-native.json`, `author-reference-validation.json`,
`author-reference-command.json`). This checks the evaluator. (2) Train our own weights
from the SFT model and score them. Only (2) tests whether we can reproduce learning.

## 8. What happened

### 8.1 First attempt (8 to 10 October)

Three sub-attempts share one ancestry (`docs/README.md`, "What has happened so far").

- Attempt 1 (08:06 UTC, 8 Oct) ran out of GPU memory while computing actor log-probs;
  the vLLM server had reserved 26.27 GiB. Exit 1 after about 90 s
  (`training-attempt1-command.json`). Fix: `train-memory03.jsonnet` sets vLLM memory
  to 0.3. Resource only.
- Attempt 2 completed the first collection: 512 answers, 1,066 unique MC requests, 16
  finite updates (`first-training-iteration.json`); test 40.01 % after 16 updates and
  41.20 % after 176 (`first-own-evaluation.json`, `progress-176-updates.json`). Copying
  the 17.6 GB checkpoint at 160 updates filled the project's 1 TB quota; training stalled
  and the job was terminated (exit 143, `training-attempt2-interrupted.json`).
- Attempt 3 resumed from the preserved 160-update state in home storage
  (`checkpoint-160-preservation.json`): 41.24 % at 176, rising to 50.52 % at 4,336
  updates (highest reviewed: 51.18 % at 3,536). Gradient norms of 2,290 and 6,850
  appeared at updates 4,271 to 4,272 with no NaN. Evaluations between 4,336 and 7,856
  were not reviewed in the notes; the run continued unattended, deteriorated, and exited
  with an error on 10 October at 00:49:56 UTC at optimizer step 7,995
  (`terminal-audit-20261012.json`).

What broke, and how each piece was detected (`docs/diagnosis-and-restart-20261012.md`):

1. The score. The last scheduled evaluation, at 7,856 updates, reported 22.19 % test
   (4,684/21,104), 22.18 % validation, 26.81 % on the training subset. Found by reading
   the native metric files after the exit.
2. Was it a scoring bug? An independent CPU regrade of 66,048 saved answers from
   evaluations 270 and 490 found zero identity or metric mismatches. Test correct went
   10,662 to 4,684 (−28.33 points); extraction failures only 14 to 18; length
   terminations 16 to 19; 96 questions went from 16/16 correct to 0/16
   (`late-regrade-20261012.json`). The model really got worse.
3. The crash. The traceback ends at
   `deepspeed/runtime/bf16_optimizer.py:297: assert all_groups_norm > 0.`
   after four consecutive microbatches were rejected by the PPO
   ratio guard with mean ratios 95.34, 36.98, 153.45 and 10.01. The guard multiplies the
   loss by zero; four zeroed microbatches in one accumulation window give a gradient norm
   of exactly zero; DeepSpeed 0.14.1's BF16 optimizer asserts it is positive. Not a
   timeout. A CPU fixture reproduced the boundary, and a GPU probe on the exactly
   restored 7,840-update state exercised a zero-norm step (`restart-20261012.json`). The
   crash explains the exit, not the decline.
4. The input mismatch. A token-alignment audit over 2,560 saved training responses from
   five archived collections found that every vLLM generation request began with the BOS
   token (id 1) followed by the training query, while the training-side tokenization
   omitted BOS. The public `_tokenize_trajectory` has its BOS and EOS lines commented
   out (`episode_generator_with_reward_function.py:149-153`). The model sampled answers
   under one input and was assigned probabilities for them under another.
5. Does the mismatch matter? On 16 saved responses, adding BOS changed the SFT model's
   total response log-probability by −0.10 nats on average; for the late model it changed
   by +53.98 nats, positive on all 16 (`bos-breadth-20261012.json`). The notes insist this
   is a likelihood diagnostic, "not a causal explanation of the earlier failure".

Fixes, each isolated and hash-pinned (`code/README.md`):

- `bos_only_adapter.py` wraps `_tokenize_trajectory` to prepend BOS when the configured
  rule says so (auto → true for Llama tokenizers), before MC likelihoods, cutpoints,
  advantages and PPO. EOS handling is deliberately unchanged.
- `bf16_zero_norm.py` recompiles the pinned `BF16_Optimizer.step` with `> 0.` replaced by
  `>= 0.`; NaN/Inf (mapped to −1) is still rejected and the Adam step still runs.
- `fixed_endpoint.py` validates the 690/11,040 state and calls the native evaluator with
  no extra update.
- `test_bos_alignment.py`, `test_endpoint.py`, `test_bf16_zero_norm.py` exercise these
  on CPU with the real tokenizer and real optimizer class.

### 8.2 The restart (12 to 14 October)

The fresh run started 12 October 15:22:49 UTC from the original SFT weights, not from an
earlier checkpoint, with the same recipe, seed 42, a 68-hour cap, and the fixed endpoint
(`restart-20261012.json`). Its first collection was audited in full: 512 answers (273
correct, 238 wrong, 1 length-penalised), 1,620 MC states, all 512 training queries now
token-identical to the vLLM requests, 16 finite updates (`startup-audit-20261012.json`).
A separate audit confirmed the logged learning rates follow the 16,000-update schedule
(`restart-learning-rate-fidelity-20261012.json`).

Learning curve (test mean pass@1, from `slides/learning-curve.csv`; all 70 scheduled
points were independently regraded and every point is kept):

| Updates | Test % | Updates | Test % |
| ---: | ---: | ---: | ---: |
| 0 (SFT) | 39.77 | 6,096 | 56.72 |
| 16 | 40.03 | 7,056 | 56.36 |
| 496 | 44.93 | 8,016 | 56.85 |
| 976 | 47.64 | 8,336 | 55.88 |
| 2,096 | 51.05 | 9,136 | 57.19 |
| 3,056 | 53.59 | 10,096 | 57.81 |
| 4,016 | 54.20 | 10,896 | 58.32 |
| 5,136 | 55.61 | **11,040 (final)** | **58.19** |

Final audited numbers (`final690-audit-20261014.json`): test 12,281/21,104 =
58.192760 %; validation 4,324/5,968 = 72.45 %; training subset 5,390/5,952 = 90.56 %;
no extraction failures; all 21,104 test answers ended with `stop`. Gain over matched SFT:
+18.418309 points. Versus the paper: +1.492760 points. The final checkpoint is not the
best intermediate one (10,896 updates scored 58.315959 %); the endpoint stayed where it
was fixed. For context only: at 2,416 updates the fresh run scored 51.81 % where the old
run had 48.49 %, but the notes stress this is "not an isolated test of the BOS
correction", since the old run also had a resumed ancestry. Launch to final audit took
under 51 hours, including 70 three-split evaluations.

## 9. Deviations and limitations

From `docs/protocol-audit.md`, `docs/final-result-20261014.md` and `restart-20261012.json`:

1. Effective top-p. Paper and config say evaluation top-p 0.9, but the evaluation prompt
   template omits the `top_p` argument, so the guidance-library default 1.0 applies, in
   the authors' historical pipeline too. The reproduction kept 1.0 so the reference
   comparison is exact. Training templates do pass top-p 0.9.
2. BOS adapter. Prepending BOS changes the public implementation. The authors' training
   predates the public source, which already has BOS commented out; whether their code
   added BOS is unknown.
3. EOS. The public source also omits EOS from training responses. Left as is.
4. BF16 adapter. Accepting a finite zero gradient norm is a runtime deviation from the
   pinned DeepSpeed.
5. One seed (42), one task. Sixteen samples per question are not independent runs. The
   result is a "performance-level reproduction", not statistical equivalence.
6. Historical config, not historical code. Recovered from W&B; no git revision recorded.
   `max_step_for_value_estimation: 25` is inactive in the inspected code. An early worry
   that the multiple-conclusion penalty was bypassed was later corrected: the reward
   function returns −2 itself.
7. Checkpoint selection. The authors' reason for reporting 690 is unresolved; their
   validation peaked near 520. Our endpoint was fixed in advance.
8. Resource and schedule. vLLM memory 0.3; three evaluation pipelines every 10
   iterations, where the historical summaries show validation only.
9. Reward quirks preserved: exact string grading; author gold labels inconsistent with
   their own worked solutions (train row 754: 108 vs 106); a correct final answer can
   carry wrong intermediate arithmetic (`restart-outcome-reasoning222-20261013.json`).
10. No GRPO control was run (`docs/grpo-control-plan.md`); checkpoint copies are
    same-filesystem, not backups; no GPU restore of the final state was tested; the old
    run's gradient spikes have no established cause.

## 10. How to read the evidence

Everything is under `docs/evidence/`:

- Immutable inputs: `realtreetune--rho-1b-sft-GSM8K.json`,
  `gyr66--spo-chain-int5-rho1.1B-gsm8k.json` (HF revision, per-file SHA256),
  `dataset-provenance.json`, `environment.md`, `environment-requirements.txt`,
  `environment-lock.txt`.
- Author recipe: `author-training-config.json` (run `rbtvnfqm`),
  `author-training-rerun-config.json` (`yc23440y`), `author-evaluation-pipelines.json`
  (`ytu1arac`), the overlays `train-historical-paths.jsonnet`, `train-memory03.jsonnet`,
  `train-resume160-home.jsonnet`, and `training-launch.json` (argv, env, schedule
  arithmetic, resume rules).
- Reference evaluations: `base-reference-*.json` (SFT) and `author-reference-*.json`.
  `-command` is the owner's exit receipt, `-native` the seven metrics the authors' code
  wrote, `-validation` the independent regrade.
- First attempt: `training-attempt1-command.json`, `training-attempt2-interrupted.json`,
  the `*-queue.json` job definitions, `first-training-iteration.json`,
  `first-own-evaluation.json`, `resumed-176-evaluation.json`, `progress-<updates>-*.json`
  per scheduled evaluation, `checkpoint-<step>-preservation.json`,
  `terminal-audit-20261012.json`, `late-regrade-20261012.json`.
- Diagnosis and restart: `restart-20261012.json` (portable summary),
  `bos-breadth-20261012.json`, `bos-preparation-20261012.json` (hashes of the sealed
  adapters), `startup-audit-20261012.json`, `restart-review-20261012.json`,
  `restart-learning-rate-fidelity-20261012.json`.
- Fresh-run evaluations: `restart-eval<N>-<date>.json` for N = 0, 10, ..., 550, with
  `-interpretation` companions; 560 has an interpretation only;
  `evaluation-backlog570-640-20261014.json` covers 570 to 640; 650 to 680 appear in
  `docs/final-result-20261014.md` and `slides/curve-provenance.json` (receipts in the
  external run store); `final690-audit-20261014.json` and
  `final690-execution-20261014.json` are the endpoint.
- Checkpoints: `restart-full<step>-*.json` (metadata, protection, preflight, finite
  scan, inventory), `checkpoint8960-` and `checkpoint10240-protection-*.json`,
  `final11040-preservation-20261014.json`.
- Training-loop audits: `restart-training-batch<N>` (all 512 rewards and token bindings
  of one collection), `restart-training-windows<step>` (logged update scalars),
  `restart-parameter-update<step>` (saved weights really changed), `restart-pacing<N>`,
  `restart-storage-headroom`, `restart-recovery-*`.

The independent regrade. A CPU-only script (in the external run store; its command is in
`late-regrade-20261012.json`) reloads the authors' dataset, checks that test indices
0..1318 are all present, matches each saved inference tree's prompt to its original
question, asserts 16 children per tree, re-extracts every answer with the native
`GSM8K.evaluate_predictions(use_original_format=True)`, and compares all seven metrics to
the `log.json` the training loop wrote. A `restart-eval` file shows this per split:
`correct`, `finish_reasons`, `independently_regraded_metrics`, `native_metrics`,
`metric_deltas` (all zero when it passes), `extraction_failures`, and a
`numeric_format_only_potential_rescues` diagnostic that never changes the score.

The checkpoint hash receipt (`final11040-preservation-20261014.json`). The 690 checkpoint
was copied to a preserved location; each of the 14 files (17,603,255,667 bytes: optimizer
states, actor model states, HF weights, tokenizer, trainer pickles) was SHA256-hashed at
source and destination; sizes and mtimes were checked for stability before and after;
`all_hashes_match: true`. The counters inside (iteration 690, step 11,040, scheduler
11,039, skipped 0) were read separately. The receipt states its own limit: a
same-filesystem copy, not a backup, with no tensor-finiteness or GPU-restore test.

## 11. Lessons for running RL experiments

- Fix the comparison point before you train, or you will pick the best test checkpoint
  without noticing.
- Score a known model first. Seven minutes on the authors' checkpoint separated
  evaluator bugs from training bugs for the rest of the week.
- Report the gain from a matched baseline under one scoring rule. 39.77 → 58.19 is the
  result; 58.19 alone is not.
- Check that the tokens you reward are the tokens you scored. Generation engine and
  trainer must agree on the exact input, BOS included; audit it every few collections.
- Regrade saved answers independently of the training code. CPU-only, and it ruled out
  a scoring bug within minutes.
- Read the safeguards. A guard that "skips" a batch produced a zero gradient and crashed
  the optimizer.
- Keep full optimizer state, not just weights, outside the trainer's cleanup path. Check
  quotas, including file counts, before copying 17.6 GB.
- Preserve failures; the collapsed run's artifacts made the diagnosis possible.
- Do not shorten a schedule to fit an allocation; `num_iterations` also set the
  learning-rate horizon.
- Write the limits next to the number. Every receipt here carries a `limits` field.

## 12. Reproduce it yourself

Copy commands from the receipts named below rather than retyping them.

1. Environment. Build Python 3.10 from `env/requirements-spo-6c5f947.txt` with `uv`
   (versions and lock in `docs/evidence/environment.md`, `environment-lock.txt`). Clone
   `https://github.com/AIFrameResearch/SPO` at `6c5f94723c20e4de68578545b8bb98186e956797`
   and leave it unmodified.
2. Assets. Download the dataset archive from the URL in
   `docs/evidence/dataset-provenance.json`, verify its hashes, extract only the ten GSM8K
   members. Snapshot the two HF models at the revisions in
   `realtreetune--rho-1b-sft-GSM8K.json` and `gyr66--spo-chain-int5-rho1.1B-gsm8k.json`
   and verify file hashes.
3. Reference evaluations. Run the `argv` in `docs/evidence/author-reference-command.json`
   (note `--no_local_rank`; `base-reference-command.json` records why). Regrade as in
   `author-reference-validation.json`. The recorded values were 0.5668593631539045 and
   0.3977445034116755.
4. Training, historical path. `docs/evidence/training-launch.json` records argv, env and
   the overlay `train-historical-paths.jsonnet`; `historical-training-queue.json` is the
   job `code/run_queue.py` ran. This reproduces the first attempt, BOS mismatch included.
5. Training, corrected path. `docs/evidence/restart-20261012.json` (`fresh.owner.argv`)
   records the launch:
   `python launch_bos_only.py --configs fresh_bos_only.jsonnet run_iteration_loop`
   with `APP_DIRECTORY` a fresh absolute directory,
   `APP_EXPERIMENT_NAME=native`, `APP_SEED=42`, `APP_MINIMIZE_STORED_FILES=False`; the
   launcher refuses anything else. Run the CPU tests first (command in `code/README.md`).
   Update the absolute paths in the three `code/` files; the source-hash pins must match.
6. Watch it. `code/review_monitor.py` reads native `log.json` metrics as they appear;
   verify the first collection as in `startup-audit-20261012.json`.
7. Regrade and preserve. Regrade each scheduled evaluation (Section 10; command pattern
   in `late-regrade-20261012.json`). After the endpoint hook writes
   `endpoint-evaluated.json`, copy and hash the 690 checkpoint as in
   `final11040-preservation-20261014.json`.
8. Plot. `slides/learning-curve.csv` and `slides/curve-provenance.json` map every point
   to its receipt and SHA256.

## 13. Glossary

- Advantage: value achieved minus value expected; the per-token learning signal.
- BOS / EOS: beginning- and end-of-sequence tokens a tokenizer may add.
- Collection (iteration): 64 questions × 8 answers plus MC sampling, then 16 updates.
- Cutpoint: a response position chosen for an MC value estimate.
- GRPO: Group Relative Policy Optimization; group-normalised trajectory-level advantages.
- KL penalty: term keeping the policy near the frozen reference model.
- Mean pass@1 (`correct_frac`): mean over questions of the fraction of 16 samples that
  are correct.
- Monte Carlo (MC) value: mean reward of sampled continuations from a prefix.
- Once-hit: fraction of questions with at least one correct sample; not the paper metric.
- PPO ratio: new-policy over sampling-policy probability of a token.
- Probability mask: exclude tokens with old-policy probability above 0.9.
- Segment: the tokens between two cutpoints.
- SFT: supervised fine-tuning; the starting model was SFT'd on GSM8K.
- Update: one optimizer step on 64 episodes.

## 14. References

- Guo, Y., Xu, L., Liu, J., Ye, D., Qiu, S. (2025). Segment Policy Optimization:
  Effective Segment-Level Credit Assignment in RL for Large Language Models.
  arXiv:2505.23564. Code: https://github.com/AIFrameResearch/SPO.
- Kazemnejad, A., Aghajohari, M., Portelance, E., Sordoni, A., Reddy, S., Courville, A.,
  Le Roux, N. (2024). VinePPO: Unlocking RL Potential for LLM Reasoning Through Refined
  Credit Assignment. arXiv:2410.01679. Code: https://github.com/McGill-NLP/VinePPO.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M.,
  Tworek, J., Hilton, J., Nakano, R., Hesse, C., Schulman, J. (2021). Training Verifiers
  to Solve Math Word Problems. arXiv:2110.14168.
  https://github.com/openai/grade-school-math.
- Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K.,
  Wu, Y., Guo, D. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in
  Open Language Models. arXiv:2402.03300.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy
  Optimization Algorithms. arXiv:1707.06347.
- Towell, A. (2026). Reproducing Segment Policy Optimization (SPO) on GSM8K with one
  A100. https://github.com/queelius/spo-gsm8k-reproduction (`CITATION.cff`).

## Questions not answerable from the sources

1. Does the paper define the first segment's advantage as zero, or use a value at the
   empty prefix? The code's query-state request is commented out; no PDF was available.
2. Did the authors' historical code prepend BOS during training? The public source does
   not; their W&B config records no git revision.
3. Why did the authors report checkpoint 690 rather than the validation maximum near 520?
4. What caused the first run's decline between 4,336 and 7,856 updates? The BOS
   mismatch is a supported hypothesis, not a demonstrated cause; no ablation was run.
5. How large is seed-to-seed variation? One seed was run.
6. Which threshold, interval and $N$ does the paper's Appendix H state, as opposed to the
   recovered config (0.9, 5, 9)? Not checkable without the PDF.
7. The diagnosis document says "seven focused CPU tests" passed while
   `restart-20261012.json` counts 4 + 3 + 3 published-copy tests; the sources do not
   reconcile these counts.
