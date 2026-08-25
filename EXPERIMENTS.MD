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

### Takeaway

Validation exact accuracy increased from 93.3% to 100.0%. The four remaining T8 failures from Experiment 002 disappeared after adding examples that prevented the model from relying on a fixed `to`-position shortcut.

This shows that targeted training-data diversity can correct specific learned shortcuts without increasing LoRA capacity.

### Next

Evaluate on new held-out sentence structures to determine whether the customer/carrier role distinction generalized beyond T12/T13 and the existing validation templates.
