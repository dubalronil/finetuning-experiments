# Experiments

## Experiment 001 — Initial LoRA baseline

### Question

Can LoRA teach Qwen3-0.6B-Base to perform the shipment extraction task zero-shot?

### Training data

- 240 examples
- Templates T1–T6
- 40 examples per template

### Configuration

- Model: Qwen/Qwen3-0.6B-Base
- LoRA targets: q_proj, v_proj
- Rank: 8
- Alpha: 16
- Dropout: 0.0
- Learning rate: 2e-4
- Batch size: 8
- Epochs: 3

### Results

| Epoch | Train Loss | Val Loss | JSON Valid | Exact |
| ----- | ---------: | -------: | ---------: | ----: |
| 1     |     0.1130 |   0.0713 |      38.3% | 38.3% |
| 2     |     0.0004 |   0.0677 |      40.0% | 40.0% |
| 3     |     0.0002 |   0.0687 |      40.0% | 40.0% |

### Diagnosis

- T7: 0/30 correct
- T8: 24/30 correct
- All 30 T7 missing-field failures omitted `carrier`.
- T7 was the only validation pattern with a bare sentence-initial carrier.

### Takeaway

The model learned the training patterns but did not generalize reliably to a carrier appearing at the beginning of the sentence.

## Experiment 002 — Add carrier-first training examples

### Question

Does adding training examples with a bare sentence-initial carrier improve generalization to T7?

### Change from Experiment 001

Added 40 T11 examples:

`{carrier} collected order #{order_id} on {ship_date} for {customer}.`

The original 240 Experiment 001 training examples remain byte-for-byte unchanged.

### Training data

- 280 examples
- T1–T6: 40 each
- T11: 40
- Validation and test sets unchanged

### Configuration

Same as Experiment 001.

### Results

| Epoch | Train Loss | Val Loss | JSON Valid | Exact |
| ----- | ---------: | -------: | ---------: | ----: |
| 1     |     0.0991 |   0.0083 |      93.3% | 93.3% |
| 2     |     0.0004 |   0.0081 |      98.3% | 93.3% |
| 3     |     0.0002 |   0.0072 |      98.3% | 93.3% |

### Diagnosis

- T7: 30/30 correct
- T8: 26/30 correct
- All 4 remaining failures were on T8.
- 3 failures had wrong field values.
- 1 failure used the wrong schema.
- The original T7 failure mode — omitting `carrier` — disappeared completely.
- OnTrac was not generally difficult: T7 with OnTrac scored 6/6.
- T8 with OnTrac scored 2/5.
- This suggests the remaining failure is an interaction between OnTrac and T8's ambiguous `to {carrier} ... to {customer}` structure, rather than insufficient exposure to OnTrac itself.

### Takeaway

Adding one new training pattern increased validation exact accuracy from 40.0% to 93.3%.

T7 improved from 0/30 to 30/30, strongly supporting the hypothesis that the original failure was largely driven by insufficient training-data diversity rather than insufficient LoRA capacity.

The remaining errors are now a different failure mode: on some T8 examples, the model confuses the customer and carrier fields.

### Next

Investigate the remaining T8 failures before changing model capacity or hyperparameters.

## Experiment 003 — Role disambiguation

### Question

Can training examples with both carrier/customer `to` orderings eliminate the remaining T8 role confusion?

### Change from Experiment 002

Added 40 examples:

- T12: 20 examples
- T13: 20 examples
- Opposite carrier/customer ordering around `to`
- 4 examples per carrier per template

The original 280 Experiment 002 rows remain byte-for-byte unchanged.

### Training data

- 320 examples
- T1–T6: 40 each
- T11: 40
- T12: 20
- T13: 20
- Validation and test sets unchanged

### Configuration

Same as Experiment 001.

### Results

| Epoch | Train Loss | Val Loss | JSON Valid |  Exact |
| ----- | ---------: | -------: | ---------: | -----: |
| 1     |     0.0948 |   0.0015 |     100.0% | 100.0% |
| 2     |     0.0002 |   0.0003 |     100.0% | 100.0% |
| 3     |     0.0001 |   0.0003 |     100.0% | 100.0% |

### Diagnosis

- T7: 30/30 correct
- T8: 30/30 correct
- JSON validity: 100%
- All four Experiment 002 T8 failures disappeared.
- The previous customer/carrier swap failure mode was no longer observed on validation.

### Role Generalization Probe

Before relying on the saturated validation set, Experiment 002 was evaluated on a separate 100-example probe with two novel sentence structures containing both customer and carrier roles.

| Adapter        |            Exact |    P1 |    P2 | Customer/Carrier Swaps |
| -------------- | ---------------: | ----: | ----: | ---------------------: |
| Exp002 epoch 2 | 100/100 (100.0%) | 50/50 | 50/50 |                      0 |

Experiment 002 generalized perfectly to both unseen probe structures.

This result ruled out the broader hypothesis that Experiment 002 had simply learned a universal `to → customer` shortcut. Its remaining weakness was narrower and appeared to be specific to the T8 structure or wording.

### T8 Diagnostic

A separate 150-example T8 diagnostic set was used to check whether the original T8 weakness was repeatable.

| Adapter        |           Exact | Customer/Carrier Swaps |
| -------------- | --------------: | ---------------------: |
| Exp002 epoch 2 | 138/150 (92.0%) |                      8 |
| Exp003 epoch 2 | 149/150 (99.3%) |                      0 |

On the same 150 examples, Experiment 003 fixed 11 Experiment 002 failures, introduced 0 new failures, and left 1 example wrong in both models.

The customer/carrier swap failure mode disappeared completely.

### Takeaway

Experiment 003 substantially improved performance on the specific T8 structure: exact accuracy increased from 92.0% to 99.3% on a larger diagnostic set, while customer/carrier swaps fell from 8 to 0.

This supports the conclusion that the targeted T12/T13 training examples corrected a real, repeatable T8-specific weakness. It does not by itself prove broader role generalization, since Experiment 002 already scored 100/100 on the separate novel-frame role probe.

### Test Evaluation

The selected Experiment 003 epoch-2 adapter was evaluated zero-shot on the fixed 300-example test set.

| Metric                 |           Result |
| ---------------------- | ---------------: |
| JSON validity          |           100.0% |
| Exact record accuracy  | 300/300 (100.0%) |
| Order ID accuracy      |           100.0% |
| Customer accuracy      |           100.0% |
| Ship date accuracy     |           100.0% |
| Carrier accuracy       |           100.0% |
| Customer/carrier swaps |                0 |

By template:

- T9: 150/150
- T10: 150/150

No test failures were observed.

### Final Takeaway

Across three controlled experiments, the main gains observed came from improving training-data coverage while LoRA capacity and hyperparameters were kept fixed.

Experiment 001 exposed a carrier-position generalization failure. Experiment 002 fixed that failure by adding a targeted carrier-first pattern, but revealed a narrower T8 role-confusion weakness. Experiment 003 added examples that varied customer/carrier role ordering and reduced T8 diagnostic errors from 12/150 to 1/150 while eliminating all customer/carrier swaps.

The selected Experiment 003 adapter then achieved 300/300 exact accuracy on the fixed zero-shot test set.

## Experiment 004 — LoRA rank sweep

### Question

How much LoRA capacity does this task actually need?

### Change from Experiment 003

The Experiment 003 training data, `q_proj`/`v_proj` targets, learning rate 2e-4, batch size 8, 3 epochs, prompt format, loss masking, decoding and evaluation logic were retained. Experiment 004 also used the now-explicit training seed of 0 for every run. Within the Experiment 004 sweep, only LoRA rank and its corresponding alpha varied.

Alpha was set to twice the rank in every run, so the adapter scale (alpha / rank) stayed at 2.0 throughout. Rank therefore changed capacity alone, rather than changing capacity and the magnitude of the adapter's contribution together.

The Experiment 003 rank-8 adapter was not reused. It was trained before the explicit training seed was added, so reusing it would have left one run of five starting from a different initialisation. All five ranks were trained fresh under identical conditions.

### Training data

- 320 examples (unchanged from Experiment 003)
- T1–T6: 40 each
- T11: 40
- T12: 20
- T13: 20
- Validation, test, diagnostic and probe sets unchanged

### Configuration

All Experiment 004 runs used the same data and training configuration, including training seed 0; only rank and its corresponding alpha varied across the sweep.

| Rank | Alpha | Alpha / Rank | Trainable params | % of total params |
| ---: | ----: | -----------: | ---------------: | ----------------: |
|    1 |     2 |            2 |          143,360 |            0.024% |
|    2 |     4 |            2 |          286,720 |            0.048% |
|    4 |     8 |            2 |          573,440 |            0.096% |
|    8 |    16 |            2 |        1,146,880 |            0.192% |
|   16 |    32 |            2 |        2,293,760 |            0.383% |

Trainable parameters scale exactly linearly with rank, at 143,360 per unit of rank.

### Checkpoint selection protocol

Fixed before any Experiment 004 checkpoint was evaluated on the diagnostic, probe or test sets:

1. highest validation exact accuracy
2. if tied, lowest validation loss
3. if still tied, the earlier epoch

Validation was used only to choose each rank's epoch. The ranks were then compared on the 150-example T8 diagnostic and the 100-example role probe, with validation reported separately rather than pooled back into the comparison. The 300-example test set was not used to select the rank.

### Results

Selected checkpoint per rank:

| Rank | Selected epoch | Val Exact | Val Loss |
| ---: | -------------: | --------: | -------: |
|    1 |              1 |     86.7% |   0.0148 |
|    2 |              2 |     90.0% |   0.0042 |
|    4 |              3 |    100.0% |   0.0004 |
|    8 |              2 |    100.0% |   0.0014 |
|   16 |              3 |    100.0% |   0.0001 |

### Development evaluation

| Rank |   T8 diagnostic |      Role probe |  Pooled | Customer/Carrier Swaps |
| ---: | --------------: | --------------: | ------: | ---------------------: |
|    1 | 138/150 (92.0%) | 100/100 (100.0%) | 238/250 |                      0 |
|    2 | 148/150 (98.7%) |  98/100 (98.0%) | 246/250 |                      0 |
|    4 | 150/150 (100.0%) | 100/100 (100.0%) | 250/250 |                      0 |
|    8 | 150/150 (100.0%) |  98/100 (98.0%) | 248/250 |                      0 |
|   16 | 149/150 (99.3%) | 100/100 (100.0%) | 249/250 |                      0 |

### Diagnosis

- Validation separated the low ranks cleanly: rank 1 reached 52/60 and rank 2 reached 54/60, while ranks 4, 8 and 16 all reached 60/60. A capacity floor is visible below rank 4.
- Every rank scored zero customer/carrier swaps on both development sets. The role distinction learned in Experiment 003 held at every capacity tested, including rank 1 with 143,360 trainable parameters.
- The remaining low-rank failures manifested mostly as customer-value reproduction errors — the name copied or generated incorrectly — plus a small number of schema errors, rather than as customer/carrier role confusion.
- Ranks 4, 8 and 16 differ by one to two examples across the 250 development examples, including the small non-monotonicity where rank 8 scored 98/100 on the probe while ranks 4 and 16 scored 100/100. Differences that small, on evaluations this close to saturated, do not separate those three ranks from one another.
- Rank 2 performed strongly on the development evaluations, at 148/150 on the diagnostic and 98/100 on the probe. Its validation accuracy was 90.0% (54/60), below the 100.0% reached by ranks 4, 8 and 16. The decision prioritised reliable validation performance alongside the development evaluations, so rank 2 was not selected.

### Rank selection

Rank 4 was selected. It was the smallest rank to reach 100.0% validation exact accuracy, 150/150 on the T8 diagnostic and 100/100 on the role probe.

Ranks 8 and 16 showed no demonstrated improvement over rank 4 on the development evaluations, while rank 4 uses half the trainable parameters of rank 8 and one quarter those of rank 16.

### Test Evaluation

The selected rank-4 epoch-3 adapter was then evaluated zero-shot on the fixed 300-example test set. This was the only Experiment 004 checkpoint run on the test set.

| Metric                 |          Result |
| ---------------------- | --------------: |
| JSON validity          |          100.0% |
| Exact record accuracy  | 293/300 (97.7%) |
| Order ID accuracy      |          100.0% |
| Customer accuracy      |           97.7% |
| Ship date accuracy     |          100.0% |
| Carrier accuracy       |          100.0% |
| Customer/carrier swaps |               0 |

By template:

- T9: 143/150 (95.3%)
- T10: 150/150 (100.0%)

All seven failures were wrong customer values on T9, and all seven were the same first name: `Ingrid` rendered as `Indrg`, as in `Ingrid Osei` to `Indrg Osei`. The schema, the field assignments, the surname and the other three fields were correct in every case.

The failure is narrower than "names are unreliable". `Ingrid` appears in 47 test rows and 0 of the 320 training rows. It is reproduced correctly in all 25 of its T10 rows and in 15 of its 22 T9 rows, so the errors are confined to one out-of-training first name in one template — the one where the customer name sits at the end of the sentence — and even there it fails only about a third of the time. No other name was ever misspelled.

### Takeaway

Rank 4 was the smallest rank that saturated the development evaluations, making it the best quality-per-parameter choice among those tested. It showed no meaningful development-performance disadvantage relative to ranks 8 and 16 while using 573,440 trainable parameters — 0.096% of the model's total parameters.

That conclusion is about relative efficiency on the development sets, and it should not be extended further. Rank 4 did not solve the task perfectly: the test set exposed a residual weakness in copying one out-of-training first name under one template, at 293/300 rather than a clean sweep. The development sets had too little headroom to have predicted it, and their name pools do not contain `Ingrid`.

Ranks 8 and 16 were deliberately not evaluated on the test set after this result was seen. The test set played no part in selecting the rank, and running the alternatives only after an unsatisfying number would compromise it as an independent final confirmation for Experiment 004, turning it into part of the rank-selection process.

Experiment 003's rank-8 adapter did score 300/300 on the same test set. That was a separate training run, made before the training seed was added, and it is not evidence that rank 8 is inherently better than rank 4. Comparing the two would require re-running both under identical conditions and evaluating on a set that had not already been consumed.

The honest summary is narrower than the headline: rank 4 showed no measurable disadvantage against the larger ranks on the development evaluations, and a residual weakness on a single unseen first name remains that none of these evaluations was sensitive enough to catch.

## Reproducibility

Dataset generation is fully deterministic: `scripts/make_dataset.py` regenerates every
file under `data/` byte-for-byte from fixed seeds, and each experiment's training set is
the previous one with rows appended rather than resampled. Evaluation is deterministic
too — greedy decoding, `do_sample=False`, and outputs verified identical across batch
sizes.

Training was not fully seeded when Experiments 001–003 were run. LoRA's `A` matrices are
drawn from the global PyTorch RNG, which was left unseeded, so those three runs cannot be
reproduced exactly. `scripts/train_lora.py` now sets `torch.manual_seed(TRAIN_SEED)` with
`TRAIN_SEED = 0` before the adapters are attached, making future runs reproducible.

The results recorded above are preserved exactly as measured, before that seed was added.
Re-running Experiments 001–003 with the current code will produce a different adapter
initialisation and therefore slightly different loss values. The conclusions do not depend
on those exact numbers — each rests on a large accuracy gap (40.0% → 93.3% → 100.0% on
validation, and 92.0% → 99.3% on the 150-example T8 diagnostic) rather than on a small
difference that initialisation noise could explain.

### Base model few-shot baseline

An earlier version of `scripts/eval_baseline.py` ignored the `--shots` value and always
used four exemplars, so the runs originally labelled 1-shot and 2-shot were in fact 4-shot
runs. The bug is fixed and the affected results were re-measured on the same 300-example
test set:

| Prompt  | JSON validity |            Exact | T9      | T10     |
| ------- | ------------: | ---------------: | ------- | ------- |
| 0-shot  |          0.0% |     0/300 (0.0%) | 0/150   | 0/150   |
| 1-shot  |         50.0% | 150/300 (50.0%)  | 150/150 | 0/150   |
| 2-shot  |        100.0% | 300/300 (100.0%) | 150/150 | 150/150 |
| 4-shot  |        100.0% | 300/300 (100.0%) | 150/150 | 150/150 |

The 1-shot split is clean rather than random: with a single T1 exemplar the model handles
T9 perfectly but fails every T10, where the sentence opens with "Tracking for …" and the
model emits a `tracking_number` key instead of the required schema. One example is not
enough to pin the output format against lexical priming from the input; two are.

This does not change any experiment's conclusion — every fine-tuned result is measured
zero-shot — but it sharpens the motivation for fine-tuning: the base model needs at least
two in-context examples to produce the schema reliably, which fine-tuning removes the need
for entirely.
