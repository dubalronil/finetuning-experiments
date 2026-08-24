# finetuning-experiments

## Goal

Learn fine-tuning from first principles through controlled experiments with small open-weight language models.

## Experiment 001 — Base Model Baseline

Question:
How well does the untouched base model perform on a structured extraction task,
before any fine-tuning?

Model:
Qwen/Qwen3-0.6B-Base

### Task

Given one sentence describing a shipment, emit a single-line JSON object with
four fields: `order_id`, `customer`, `ship_date`, `carrier`.

Input:

    Order #4417 for Priya Raman shipped on 2024-03-08 via FedEx.

Output:

    {"order_id": "4417", "customer": "Priya Raman", "ship_date": "2024-03-08", "carrier": "FedEx"}

Every value appears in the sentence, except `order_id` which drops the leading
`#`. The content is easy on purpose, so that most failure is _format_ failure.

### Dataset

Small and synthetic, generated from sentence templates filled with made-up
names, order numbers, dates, and a fixed set of five carriers.

| Split | Items |
| ----- | ----- |
| train | 240   |
| val   | 60    |
| test  | 300   |

**Test uses sentence templates that never appear in train or val**, so the score
measures generalization rather than memorization. Train is unused in this
experiment but defined now for later fine-tuning.

Generation is deterministic: one fixed seed, and regenerating produces identical
files.

### Evaluation

Two prompt regimes, both greedy decoding:

- **4-shot** — four fixed worked examples before the eval sentence. Primary.
- **Zero-shot** — no examples.

Three metrics:

1. **JSON validity** — did it produce a well-formed object with the four
   expected fields?
2. **Exact record accuracy** — were all four fields correct?
3. **Per-field accuracy** — which fields does it get right or wrong?

Validity and accuracy are kept separate on purpose. A later fine-tune might fix
the format without improving the content, and that difference is worth seeing.

### Goal

Establish the baseline score that every future fine-tuned model must beat.
