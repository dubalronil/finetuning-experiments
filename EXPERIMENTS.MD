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
