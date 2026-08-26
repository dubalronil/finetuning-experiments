"""Minimal LoRA fine-tune. Trains on any experiment's data via --exp.

Goal: teach the base model to do the JSON-extraction task ZERO-SHOT, i.e. without
needing a worked example in the prompt. On the 300-example test set the base model
scores 0% zero-shot, 50% with one exemplar and 100% with two, so the capability is
there - it just needs the output format baked in rather than demonstrated.

Everything is hand-rolled (no peft / TRL / Trainer) so each mechanical step is
visible: adapter maths, loss masking, forward, backward, optimizer step.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the evaluation code so training and eval can never drift apart:
# the same build_prompt() makes both the training text and the eval prompt.
from eval_baseline import (DEVICE, FIELDS, add_exp_arg, build_prompt, generate, load,
                           load_model, parse, use_experiment)

# --- Hyperparameters (see EXPERIMENTS.MD) ----------------------------------
RANK, ALPHA, DROPOUT = 8, 16, 0.0   # rank 8 -> 1.15M trainable params (0.19% of model)
LR, BATCH_SIZE, EPOCHS = 2e-4, 8, 3 # LoRA tolerates a much higher LR than full fine-tuning
WARMUP_STEPS = 10                   # ramp LR from 0 to avoid a destructive first step
TARGETS = ("q_proj", "v_proj")      # the original LoRA paper's minimal choice
TRAIN_SEED = 0                      # seeds LoRA init; batch order is seeded separately
CKPT_ROOT = Path("checkpoints")   # actual dir is CKPT_ROOT/<exp>_lora, chosen in main()


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a small trainable "detour" beside it.

    Normal fine-tuning would update W directly - 600M numbers. Instead we leave W
    frozen and learn two skinny matrices whose product is added to W's output:

        h = Wx + (alpha/r) * B(Ax)

    A is (r x in) and B is (out x r), so instead of (out x in) parameters we only
    train r*(in + out). With r=8 that is ~40k numbers per layer instead of ~3M.

    The key trick: B is initialised to ZERO, so B(Ax) = 0 on the first forward pass
    and the wrapped model is bit-identical to the base model before training starts.
    A fine-tune therefore cannot begin worse than the baseline.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base                                   # the original weights; stays frozen
        # LoRA params are float32 even though the base model is bfloat16. The adapters
        # receive tiny gradients, and bf16 has too few mantissa bits to accumulate them
        # reliably. This is standard mixed-precision practice.
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.normal_(self.A, std=0.02)                  # B stays zero -> no-op at step 0
        # alpha/r keeps the update's magnitude roughly constant if you change rank later,
        # so you don't have to re-tune the learning rate every time.
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)                                 # frozen path (bfloat16)
        # Adapter path in float32, then cast back so the two can be added.
        delta = F.linear(F.linear(self.dropout(x).float(), self.A), self.B) * self.scale
        return out + delta.to(out.dtype)


def attach_lora(model):
    """Freeze the whole model, then replace the target projections with LoRA versions.

    Order matters: freeze FIRST, then attach. The LoRA parameters are created after
    the freeze, so they keep requires_grad=True and end up as the only trainable
    tensors in the model.

    Prints and returns the trainable parameter count: a silent mismatch here
    (adapters not attached, or base weights left unfrozen) is one of the easiest bugs
    to miss, and the printed line is the cheapest way to catch it.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    # Qwen3's decoder blocks live at model.model.layers; each has a .self_attn
    # holding q_proj / k_proj / v_proj / o_proj. We swap two of them in place.
    for layer in model.model.layers:
        for name in TARGETS:
            base = getattr(layer.self_attn, name)
            setattr(layer.self_attn, name, LoRALinear(base, RANK, ALPHA, DROPOUT).to(base.weight.device))

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for _, p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable {n_train:,} / {n_total:,} = {n_train/n_total:.3%}  ({len(trainable)} tensors)")
    return n_train


def encode(tokenizer, row):
    """Turn one dataset row into (input_ids, labels).

    The training text is the eval prompt plus the answer the model should have given:

        Sentence: Order #4958 for Rosa Ferreira shipped on 2024-03-19 via USPS.
        JSON: {"order_id": "4958", ...}<|endoftext|>
        \\____________ prompt, masked ____________/\\___ completion, learned ___/

    Two details that are easy to get wrong:
      * the completion starts with a SPACE, because the prompt ends at "JSON:" with
        none. Train on the wrong side of that space and the model learns to predict a
        token that never appears at eval time.
      * the EOS token is what teaches the model to STOP. Without it the fine-tuned
        model would keep rambling after the closing brace, exactly like the base model.

    Labels are a copy of input_ids with prompt positions replaced by -100, which is
    PyTorch's ignore_index: those positions contribute zero loss and zero gradient.
    We mask the prompt because we don't want to teach the model to write shipment
    sentences - only to answer them.
    """
    prompt = build_prompt(row["sentence"], [])            # [] = no exemplars = zero-shot
    completion = " " + json.dumps(row["target"])
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ids = tokenizer(prompt + completion, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    # BPE can merge across the prompt/completion boundary, which would mean the tokens
    # we train on don't line up with the tokens produced at eval time. Fail loudly.
    assert ids[: len(p_ids)] == p_ids, "tokenizer boundary shifted between prompt and full text"
    return ids, [-100] * len(p_ids) + ids[len(p_ids) :]


def collate(batch, pad_id):
    """Stack variable-length examples into rectangular tensors.

    Sequences in a batch differ in length, so short ones are padded to the longest.
    Padding is neutralised twice over:
      * labels get -100    -> pad positions contribute nothing to the loss
      * attention_mask 0   -> the model doesn't attend to them

    Right-padding is fine here. (Left-padding is only required for GENERATION, where
    the last token must be a real one; see load_model() in eval_baseline.py.)
    """
    width = max(len(ids) for ids, _ in batch)
    input_ids = torch.tensor([ids + [pad_id] * (width - len(ids)) for ids, _ in batch])
    labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch])
    return input_ids, labels, attention_mask


def loss_on(model, input_ids, labels, attention_mask):
    """Standard causal-LM cross-entropy.

    Position i predicts token i+1, so hidden[:, :-1] is compared against
    labels[:, 1:] - that one-position offset is the "shift". (HF does this internally
    if you pass labels=; it's written out here so the mechanic is visible.)

    Only positions whose label is not -100 contribute anything, so those are selected
    FIRST and just they go through lm_head. The vocabulary projection is 151,936 wide,
    so running it at every position would build a ~187MB logits tensor (doubled again
    by .float()) and then throw over half of it away. Gathering first is mathematically
    identical - verified to match the naive full-logits version exactly - but cheaper.
    """
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    h, tgt = hidden[:, :-1, :], labels[:, 1:]
    keep = tgt != -100                                # positions that carry loss
    return F.cross_entropy(model.lm_head(h[keep]).float(), tgt[keep])


@torch.no_grad()
def val_loss(model, batches, device):
    """Average loss over the validation set, weighted by token count.

    Weighting by the number of loss-bearing tokens (rather than averaging the
    per-batch means) keeps a short final batch from counting as much as a full one.
    """
    model.eval()
    total, n = 0.0, 0
    for input_ids, labels, attention_mask in batches:
        loss = loss_on(model, input_ids.to(device), labels.to(device), attention_mask.to(device))
        ntok = (labels[:, 1:] != -100).sum().item()
        total, n = total + loss.item() * ntok, n + ntok
    return total / n


@torch.no_grad()
def val_accuracy(model, tokenizer, rows, device):
    """Zero-shot exact-record accuracy, using the same logic as the baseline evaluator.

    This is the metric that actually matters. Validation LOSS can look healthy while
    the model still emits malformed JSON, because loss is computed with teacher
    forcing (the model sees the correct prefix at every position). Generation has no
    such safety net - one wrong token derails the rest of the sequence.

    use_cache is toggled on for generation (the KV cache is a big speed win when
    producing tokens one at a time) and back off for training, where it's unused.
    """
    model.eval()
    model.config.use_cache = True
    prompts = [build_prompt(r["sentence"], []) for r in rows]
    raws = generate(tokenizer, model, prompts, batch_size=8)
    model.config.use_cache = False
    preds = [parse(r) for r in raws]
    valid = sum(p is not None for p in preds) / len(rows)
    exact = sum(
        p is not None and all(str(p[k]).strip() == r["target"][k] for k in FIELDS)
        for p, r in zip(preds, rows)
    ) / len(rows)
    return valid, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-steps", type=int, help="stop early; for sanity checks")
    ap.add_argument("--skip-eval", action="store_true", help="skip generation-based accuracy")
    add_exp_arg(ap)
    args = ap.parse_args()

    use_experiment(args.exp)
    # Seed BEFORE attach_lora: LoRA's A matrices are drawn from the global torch RNG,
    # and they are the only unseeded randomness left (the batch order has its own
    # generator, and dropout is 0). Without this a rerun trains a different adapter.
    # Experiments 001-003 predate this line - see the note in EXPERIMENTS.MD.
    torch.manual_seed(TRAIN_SEED)
    # Checkpoints are namespaced by experiment, so a later run cannot overwrite an
    # earlier experiment's adapters.
    ckpt_dir = CKPT_ROOT / f"{args.exp}_lora"
    device = DEVICE
    tokenizer, model = load_model()
    model.config.use_cache = False    # the KV cache is for generation; unused in training
    attach_lora(model)

    # Tokenise everything up front - 300 short examples, so this is instant and keeps
    # the training loop free of data-prep noise.
    train_rows, val_rows = load("train"), load("val")
    train_data = [encode(tokenizer, r) for r in train_rows]
    val_batches = [
        collate(v, tokenizer.pad_token_id)
        for v in [[encode(tokenizer, r) for r in val_rows[i : i + BATCH_SIZE]]
                  for i in range(0, len(val_rows), BATCH_SIZE)]
    ]

    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
    total_steps = steps_per_epoch * args.epochs
    # Only the LoRA parameters are handed to the optimizer. Everything else is frozen,
    # so AdamW's momentum/variance state is tiny (2 extra copies of 1.15M numbers).
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR)
    # Linear warmup for WARMUP_STEPS, then linear decay to zero. Warmup matters at
    # 2e-4: a full-size step into randomly-initialised A can wreck the adapter early.
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: s / max(1, WARMUP_STEPS) if s < WARMUP_STEPS
        else max(0.0, (total_steps - s) / max(1, total_steps - WARMUP_STEPS)),
    )
    print(f"  {len(train_data)} train / {len(val_rows)} val | {steps_per_epoch} steps/epoch "
          f"| {total_steps} total steps | lr {LR}")

    generator = torch.Generator().manual_seed(0)   # fixed seed -> same shuffle every run
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Reshuffle each epoch so the model doesn't learn anything from batch ordering.
        order = torch.randperm(len(train_data), generator=generator).tolist()
        running = []
        for i in range(0, len(order), BATCH_SIZE):
            batch = [train_data[j] for j in order[i : i + BATCH_SIZE]]
            input_ids, labels, attention_mask = collate(batch, tokenizer.pad_token_id)

            # --- the five steps of a training iteration ---
            loss = loss_on(model, input_ids.to(device), labels.to(device), attention_mask.to(device))
            loss.backward()                 # compute gradients; only A and B receive any
            optimizer.step()                # nudge the LoRA params downhill
            schedule.step()                 # advance the learning-rate schedule
            optimizer.zero_grad()           # PyTorch ACCUMULATES grads; clear before next batch

            running.append(loss.item())
            step += 1
            print(f"\r  epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}", end="", flush=True)
            if args.max_steps and step >= args.max_steps:
                print(f"\n  stopped early at {step} steps. mean loss "
                      f"first 3 {sum(running[:3])/3:.4f} -> last 3 {sum(running[-3:])/3:.4f}")
                return

        # End of epoch: report both loss and the metric we actually care about.
        tr, vl = sum(running) / len(running), val_loss(model, val_batches, device)
        line = f"\r  epoch {epoch}  train_loss {tr:.4f}  val_loss {vl:.4f}"
        if not args.skip_eval:
            valid, exact = val_accuracy(model, tokenizer, val_rows, device)
            line += f"  val_json_valid {valid:.1%}  val_exact {exact:.1%}"
        print(line)

        # Save ONLY the adapter (~4.5MB), not the 600M-parameter base model. The base
        # weights never changed, so the adapter plus the model id is the full record.
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        torch.save(adapter, ckpt_dir / f"epoch{epoch}.pt")


if __name__ == "__main__":
    main()
