# How It Works

This project is a small, readable fine-tuning pipeline built to understand what happens between raw training examples and a trained LoRA adapter.

The overall flow is:

    Synthetic shipment examples
              ↓
        Prompt formatting
              ↓
    Qwen3-0.6B-Base tokenizer
              ↓
       Frozen Qwen model
              +
       Trainable LoRA weights
              ↓
        Next-token predictions
              ↓
    Loss on target JSON + EOS
              ↓
         Backpropagation
              ↓
      Update LoRA weights
              ↓
        Save LoRA adapter
              ↓
       Zero-shot evaluation
              ↓
       Inspect failures
              ↓
    Design the next experiment

## 1. Data

Each example contains a shipment sentence and the JSON the model should produce.

Example input:

```text
Order #4417 for Priya Raman shipped on 2024-03-08 via FedEx.
```

Target:

```json
{"order_id": "4417", "customer": "Priya Raman", "ship_date": "2024-03-08", "carrier": "FedEx"}
```

The data is synthetic so sentence structure and field placement can be controlled precisely.

This makes it possible to create specific training patterns, hold out other patterns, and test whether the model generalizes.

## 2. Prompt formatting

Training examples are converted into a prompt and completion:

```text
Sentence: Order #4417 for Priya Raman shipped on 2024-03-08 via FedEx.
JSON: {"order_id": "4417", "customer": "Priya Raman", "ship_date": "2024-03-08", "carrier": "FedEx"}
```

The tokenizer converts the text into token IDs that Qwen can process.

## 3. Supervised fine-tuning

During training, the model receives both the shipment sentence and the correct answer.

However, the loss is calculated only on the target JSON tokens and the final EOS token.

Conceptually:

    Sentence: ...      JSON: ...      <EOS>
    [ignored loss]     [learn from these tokens]

The prompt tokens are masked from the loss.

This teaches the model:

    shipment sentence → desired JSON output

rather than teaching it to reproduce the prompt itself.

## 4. LoRA

The original Qwen3-0.6B parameters stay frozen.

Instead of updating all ~597 million model parameters, small trainable LoRA matrices are attached to parts of the model's attention layers.

How large those matrices are is set by the adapter's rank. At rank 8 with alpha 16, the configuration used in Experiments 001–003:

```text
Base model parameters
~597M

Trainable LoRA parameters
1,146,880

Trainable fraction
~0.192%
```

Rank is configurable. The trainable parameter count scales linearly with it, at 143,360 parameters per unit of rank.

LoRA is attached to:

```text
q_proj
v_proj
```

During training, gradients update only the LoRA parameters.

The original Qwen weights remain unchanged.

Conceptually:

    Frozen Qwen weights
            +
    Small trainable LoRA weights
            ↓
       model behavior

## 5. Training

A simplified training step looks like:

    batch
      ↓
    forward pass
      ↓
    next-token predictions
      ↓
    compare target-token predictions
    with the correct target tokens
      ↓
    loss
      ↓
    loss.backward()
      ↓
    gradients
      ↓
    optimizer.step()
      ↓
    updated LoRA weights

This repeats across the training dataset for multiple epochs.

The important pieces are:

```text
forward pass
→ calculate loss
→ backpropagate gradients
→ optimizer updates LoRA parameters
```

Only the adapter changes.

The base Qwen model stays frozen.

## 6. Checkpoints

At the end of each epoch, the learned LoRA parameters are saved as a checkpoint, together with the rank and alpha they were trained with so the adapter can be rebuilt exactly.

A checkpoint does not contain another full copy of Qwen.

For example:

```text
checkpoints/exp004_r4/epoch3.pt
```

At inference time:

    Qwen3-0.6B-Base
            +
    Experiment 004 rank-4 epoch-3 LoRA checkpoint
            ↓
       fine-tuned behavior

This means multiple experiments can share the same frozen Qwen base model while using different learned adapters.

Checkpoints are generated locally on whichever machine performs training.

In my workflow, the main training runs are performed on the Zephyrus using CUDA.

The directory:

```text
checkpoints/
```

is gitignored, so checkpoints are not committed to GitHub.

## 7. Zero-shot evaluation

Fine-tuned adapters are evaluated zero-shot.

The model receives:

```text
Sentence: <shipment sentence>
JSON:
```

There are no worked examples in the prompt.

Generation is:

```text
greedy
deterministic
do_sample=False
```

The generated output is checked for:

- JSON validity
- exact record accuracy
- per-field accuracy
- wrong values
- missing fields
- wrong schemas
- customer/carrier swaps

Results can also be broken down by sentence template to reveal failures that overall accuracy might hide.

## 8. Interactive prediction

The same inference path can also be used interactively.

Running:

```bash
python scripts/predict.py --adapter checkpoints/exp004_r4/epoch3.pt
```

loads:

    Qwen3-0.6B-Base
            +
    selected LoRA checkpoint

and lets me type new shipment sentences manually.

For example:

```text
sentence> FedEx picked up order #9991 for Alex Johnson on 2026-08-25.
```

The model then generates the JSON response using the same prompt construction and greedy decoding used by the evaluator.

This makes it possible to inspect the trained model directly instead of only looking at aggregate benchmark scores.

## 9. Experiment loop

The project follows the same process for each experiment:

    Train
      ↓
    Evaluate
      ↓
    Inspect failures
      ↓
    Form a hypothesis
      ↓
    Change one major variable
      ↓
    Train again

For example:

    model misses carriers
    at sentence start
              ↓
    hypothesis:
    insufficient carrier-first examples
              ↓
    add a carrier-first
    training pattern
              ↓
    retrain
              ↓
    test whether the specific
    failure disappears

The goal is not just to maximize an accuracy number.

The goal is to understand why the model behaves differently after each controlled change.

## Code Map

```text
scripts/make_dataset.py
    Generates controlled synthetic datasets.

scripts/train_lora.py
    Attaches LoRA modules and trains the adapter.

scripts/eval_baseline.py
    Evaluates the untouched base model.

scripts/eval_adapter.py
    Loads a LoRA checkpoint and evaluates it.

scripts/predict.py
    Loads a LoRA checkpoint and lets me enter my own
    shipment sentences interactively.

scripts/play.py
    Interactive free-form experimentation with the base model.

data/
    Generated training, validation, test, probe, and
    diagnostic datasets.

checkpoints/
    Locally generated LoRA adapter checkpoints.
    Gitignored and not stored on GitHub.

results/
    Per-example evaluation outputs.

EXPERIMENTS.MD
    Experiment questions, results, diagnoses, and takeaways.
```
