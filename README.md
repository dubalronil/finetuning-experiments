# finetuning-experiments

A hands-on project for learning how supervised fine-tuning and LoRA work through controlled experiments on a small open-weight language model.

LoRA fine-tunes a model by keeping its original weights frozen and training a small set of extra parameters that adjust its behavior for a specific task.

The project fine-tunes `Qwen/Qwen3-0.6B-Base` to convert shipment sentences into structured JSON. Rather than relying on high-level fine-tuning frameworks, the training pipeline is intentionally small and readable so the full process stays visible: dataset design, prompt/completion masking, LoRA, training, evaluation, and failure analysis.

Each experiment changes one major variable at a time and uses model failures to decide what to test next.

## Task

Given a shipment sentence:

```text
Order #4417 for Priya Raman shipped on 2024-03-08 via FedEx.
```

produce:

```json
{"order_id": "4417", "customer": "Priya Raman", "ship_date": "2024-03-08", "carrier": "FedEx"}
```

The task is intentionally simple so that failures in formatting, field assignment, and generalization are easy to inspect.

## Compute

I used two laptops for different parts of the project:

- **MacBook Air 15-inch (M2, 2023, 8 GB unified memory)** — main development machine for writing code, designing experiments, generating datasets, reviewing results, and managing Git.
- **ASUS ROG Zephyrus M16 (RTX 3060 Laptop GPU, 6 GB VRAM, 16 GB RAM)** — CUDA worker for the heavier fine-tuning and evaluation runs.

I initially tested training on the Mac using PyTorch's MPS backend, but the 8 GB shared memory and large vocabulary projection made training slow and memory-constrained.

The Zephyrus's NVIDIA GPU could run the same code with CUDA much faster, so I kept the Mac as the main development environment and used the Zephyrus as a small GPU worker.

The scripts automatically select CUDA, MPS, or CPU depending on the available hardware.

## Results

Across the first three experiments, using the same Qwen3-0.6B base model and LoRA configuration throughout, I improved zero-shot extraction performance by changing only the training data.

| Experiment | Change                         | Zero-shot Validation Exact |
| ---------- | ------------------------------ | -------------------------: |
| Exp001     | Initial 240-example dataset    |                      40.0% |
| Exp002     | + carrier-first examples       |                      93.3% |
| Exp003     | + role-disambiguation examples |                     100.0% |

The selected Experiment 003 adapter achieved:

- **300/300 (100%) exact accuracy** on the fixed zero-shot test set
- **100% JSON validity**
- **100% accuracy on every extracted field**
- **0 customer/carrier swaps**

A separate 150-example zero-shot T8 diagnostic improved from **92.0% with Experiment 002 to 99.3% with Experiment 003**, while customer/carrier swaps fell from 8 to 0.

Experiment 004 then swept LoRA rank across 1, 2, 4, 8 and 16 on the same 320-example training set. Rank 4 was the smallest rank to reach 100% validation exact accuracy along with a perfect T8 diagnostic and role probe, using **573,440 trainable parameters — about 0.096% of total parameters**. Its final zero-shot test result was **293/300 (97.7%)** with 100% JSON validity.

## Documentation

- [How it works](HOW_IT_WORKS.md) — conceptual walkthrough of the fine-tuning pipeline
- [Experiments](EXPERIMENTS.MD) — experiment history, results, failure analysis, and takeaways

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The Qwen base model is downloaded from Hugging Face automatically the first time it is needed.

Generate the datasets:

```bash
python scripts/make_dataset.py
```

## Train an adapter

Train the selected Experiment 004 rank-4 adapter:

```bash
python scripts/train_lora.py --exp exp003 --rank 4 --ckpt-dir checkpoints/exp004_r4
```

This creates local LoRA checkpoints such as:

```text
checkpoints/exp004_r4/epoch1.pt
checkpoints/exp004_r4/epoch2.pt
checkpoints/exp004_r4/epoch3.pt
```

The `checkpoints/` directory is gitignored, so trained adapters are not included in the GitHub repository.

Anyone cloning the repository must train an adapter locally before using it.

## Evaluate an adapter

Evaluate the selected rank-4 epoch-3 adapter on the test set:

```bash
python scripts/eval_adapter.py \
  --exp exp003 \
  --adapter checkpoints/exp004_r4/epoch3.pt \
  --split test
```

Evaluation is zero-shot and uses greedy deterministic decoding.

## Try the fine-tuned model

After training an adapter, run the interactive prediction script:

```bash
python scripts/predict.py --adapter checkpoints/exp004_r4/epoch3.pt
```

Then enter your own shipment sentence:

```text
sentence> FedEx picked up order #9991 for Alex Johnson on 2026-08-25.
```

Example model output:

```text
' {"order_id": "9991", "customer": "Alex Johnson", "ship_date": "2026-08-25", "carrier": "FedEx"}'
```

`predict.py` loads the same Qwen base model and LoRA adapter used during evaluation and reuses the same zero-shot prompt, greedy decoding, and parsing path.

Type `quit` or `exit`, or press Ctrl-C/Ctrl-D, to stop.

## What I Learned

The biggest lesson from the first three experiments was how strongly fine-tuning behavior depends on training-data coverage.

Instead of immediately changing model size, LoRA rank, or hyperparameters, I kept the training configuration fixed, inspected specific zero-shot failures, formed hypotheses about why they occurred, added targeted training examples, and evaluated again.

This made it possible to connect individual data changes to specific improvements rather than treating fine-tuning as a black box.

The rank sweep added a second lesson: more LoRA capacity was not automatically better. Ranks 8 and 16 showed no measurable advantage over rank 4 on the development evaluations while using two to four times as many trainable parameters.
