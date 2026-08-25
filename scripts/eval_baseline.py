"""Baseline evaluator for Experiment 001.

Measures how well a model turns a shipment sentence into a fixed four-field JSON
record. Supports two prompt regimes:

  * zero-shot - just the question. A base model has no idea what schema you want.
  * few-shot  - N solved examples first, so the model can infer the pattern.

Decoding is greedy (no randomness), so a run is exactly reproducible.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
FIELDS = ["order_id", "customer", "ship_date", "carrier"]
DATA = Path("data/exp001")
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 96          # a correct answer is ~43 tokens; this leaves fair headroom


def pick_device():
    """CUDA if present, else Apple MPS, else CPU.

    Lets the identical code run on a CUDA worker and on the Mac without edits.
    Note DTYPE stays bfloat16 everywhere; on CPU that runs but is slow, so CPU is a
    last-resort fallback rather than a supported path.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()


def load(split):
    """Read one JSONL split into a list of dicts (sentence / target / template_id)."""
    return [json.loads(line) for line in open(DATA / f"{split}.jsonl")]


def block(sentence, target=None):
    """Format one prompt block.

    With a target it's a worked example (the model sees the answer);
    without one it's the question the model must complete.
    """
    line = f"Sentence: {sentence}\nJSON:"
    return line if target is None else f"{line} {json.dumps(target)}"


def build_prompt(sentence, shots):
    """Join N worked examples and the question into a single prompt string.

    Blocks are separated by a blank line. The prompt deliberately ends with "JSON:"
    and NO trailing space: a leading space is part of the next token in BPE, so
    adding one here would change the model's first prediction. The space instead
    belongs to what the model generates.

    Passing shots=[] gives the zero-shot prompt. Training uses this same function,
    so the text a fine-tune learns on can never drift from the text it's tested on.
    """
    return "\n\n".join([block(r["sentence"], r["target"]) for r in shots] + [block(sentence)])


def parse(text):
    """Extract a prediction from the model's raw output, strictly.

    Rules: take the first line only, and it must be ENTIRELY a JSON object carrying
    exactly the four expected keys. No salvaging JSON out of surrounding prose, no
    tolerating extra or missing fields.

    Strictness is deliberate. Lenient parsing would quietly paper over the format
    failures this experiment exists to measure. Returns None when invalid.
    """
    try:
        obj = json.loads(text.split("\n")[0].strip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and set(obj) == set(FIELDS) else None


def load_model():
    """Load tokenizer and model onto the GPU (MPS) in bfloat16.

    padding_side="left" is REQUIRED for batched generation. Sequences in a batch have
    different lengths, so short ones get padded. With right-padding a short prompt
    would end in <pad>, and the model would be asked to continue from padding rather
    than from "JSON:" - producing garbage silently, with no error raised.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    model.eval()                 # disable train-time behaviour such as dropout
    return tokenizer, model


def generate(tokenizer, model, prompts, batch_size=1, progress=False):
    """Greedily continue each prompt and return the generated text only.

    batch_size affects speed alone - outputs were verified byte-identical across
    batch sizes 1, 2, 4 and 8.

    do_sample=False makes generation deterministic: at every position the single
    highest-probability token is taken, rather than a weighted random draw. Sampling
    would add run-to-run variance that could swamp the differences we're measuring.

    stop_strings=["\\n"] halts a sequence at its first newline, since a correct answer
    is one line. Without it every sequence would run the full 96 tokens.
    """
    outputs = []
    for i in range(0, len(prompts), batch_size):
        inputs = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():                      # no gradients needed -> less memory
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,          # greedy -> deterministic
                stop_strings=["\n"],
                tokenizer=tokenizer,
            )
        # generate() returns prompt + continuation. Slice the prompt off so we keep
        # only what the model actually produced. Left-padding means every row in the
        # batch shares the same prompt width, so one offset works for all of them.
        prompt_len = inputs.input_ids.shape[1]   # same for all rows after padding
        outputs += [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in out]
        if progress:
            print(f"\r  {len(outputs)}/{len(prompts)}", end="", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=4, choices=[0, 1, 2, 4])
    ap.add_argument("--limit", type=int, help="evaluate only the first N test examples")
    ap.add_argument("--batch-size", type=int, default=1, help="speed only; results are unchanged")
    args = ap.parse_args()

    test = load("test")[: args.limit]
    # Exemplars are the first N rows of the TRAIN split - fixed for every test item, so
    # no item gets luckier examples than another, and never drawn from test.
    shots = load("train")[:4] if args.shots else []

    tokenizer, model = load_model()
    prompts = [build_prompt(row["sentence"], shots) for row in test]
    raws = generate(tokenizer, model, prompts, args.batch_size, progress=True)

    # Score every prediction and keep the raw text, so failures can be inspected later.
    records = []
    for row, raw in zip(test, raws):
        pred = parse(raw)
        # str() so an int 4417 is compared as "4417" rather than crashing.
        fields = {k: pred is not None and str(pred[k]).strip() == row["target"][k] for k in FIELDS}
        records.append({
            "sentence": row["sentence"],
            "template_id": row["template_id"],
            "target": row["target"],
            "raw": raw,
            "parsed": pred,
            "valid": pred is not None,
            "fields": fields,
            "exact": all(fields.values()),      # correct only if ALL four fields match
        })

    out_path = Path("results") / f"exp001_base_{args.shots}shot.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Validity and accuracy are reported separately on purpose: a fine-tune can fix the
    # output FORMAT without improving the CONTENT, and one number would hide that.
    n = len(records)
    print(f"\r\n{MODEL} | {args.shots}-shot | {n} examples | greedy | {DTYPE} | {DEVICE} | batch {args.batch_size}")
    print(f"  JSON validity        {sum(r['valid'] for r in records) / n:6.1%}")
    print(f"  Exact record         {sum(r['exact'] for r in records) / n:6.1%}")
    print("  Per-field accuracy")
    for k in FIELDS:
        print(f"    {k:<12s}       {sum(r['fields'][k] for r in records) / n:6.1%}")
    print(f"  raw predictions -> {out_path}")


if __name__ == "__main__":
    main()
