"""Evaluate a saved LoRA adapter, using the baseline evaluator's exact prompt and decoding.

Diagnostic counterpart to eval_baseline.py. Everything that could shift a score -
build_prompt(), greedy generation, parse(), the scoring rules - is imported rather
than reimplemented, so an adapter's number is directly comparable to the baseline's.
"""

import argparse
import collections
import json
from pathlib import Path

import torch

from eval_baseline import (DEVICE, FIELDS, add_exp_arg, build_prompt, generate, load,
                           load_model, parse, use_experiment)
from train_lora import attach_lora


def load_adapter(model, path):
    """Attach LoRA modules to the base model, then copy in the trained A/B weights.

    attach_lora() creates the adapters with B=0 (a no-op). load_state_dict fills in
    the trained values. strict=False is required because the checkpoint holds ONLY
    the adapter tensors - every base weight shows up as "missing", which is expected.
    """
    attach_lora(model)
    state = torch.load(path, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"checkpoint has keys the model doesn't: {unexpected[:3]}"

    # Guard against silently evaluating an untrained adapter: B starts at exactly zero,
    # so if every B is still zero, nothing was actually loaded.
    bs = [p for n, p in model.named_parameters() if n.endswith(".B")]
    assert any(p.abs().sum().item() > 0 for p in bs), "all B matrices are zero - adapter not loaded"
    print(f"  loaded {len(state)} adapter tensors from {path}")
    return model


def classify(raw):
    """Bucket one failure. Mirrors parse()'s rules, but explains WHY it failed."""
    first = raw.split("\n")[0].strip()
    if not first:
        return "empty first line (multiline output)"
    try:
        obj = json.loads(first)
    except json.JSONDecodeError:
        # Valid JSON spread over several lines still fails parse(), which reads line 1.
        try:
            json.loads(raw.strip())
            return "multiline / extra text"
        except json.JSONDecodeError:
            return "unparseable / other"
    if not isinstance(obj, dict):
        return "not a JSON object"
    keys = set(obj)
    if keys < set(FIELDS):
        return "missing fields"
    if keys != set(FIELDS):
        return "wrong schema"
    return "wrong values"          # right shape, wrong content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to a saved epochN.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "probe"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", help="output filename tag; defaults to the adapter's stem. "
                                  "Adapters from different experiments share stems "
                                  "(epoch2.pt), so pass this when probing more than one.")
    add_exp_arg(ap)
    args = ap.parse_args()

    use_experiment(args.exp)
    rows = load(args.split)[: args.limit]
    tokenizer, model = load_model()
    load_adapter(model, args.adapter)

    # [] = no exemplars: the exact zero-shot prompt the baseline used.
    prompts = [build_prompt(r["sentence"], []) for r in rows]
    raws = generate(tokenizer, model, prompts, args.batch_size, progress=True)

    records = []
    for row, raw in zip(rows, raws):
        pred = parse(raw)
        fields = {k: pred is not None and str(pred[k]).strip() == row["target"][k] for k in FIELDS}
        records.append({
            "sentence": row["sentence"],
            "template_id": row["template_id"],
            "target": row["target"],
            "raw": raw,
            "parsed": pred,
            "valid": pred is not None,
            "fields": fields,
            "exact": all(fields.values()),
            "failure": None if all(fields.values()) else classify(raw),
        })

    tag = args.tag or Path(args.adapter).stem
    out_path = Path("results") / f"{args.exp}_lora_{tag}_{args.split}_0shot.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n = len(records)
    print(f"\r\nadapter {args.adapter} | {args.split} | {n} examples | zero-shot | greedy | {DEVICE}")
    print(f"  JSON validity        {sum(r['valid'] for r in records) / n:6.1%}")
    print(f"  Exact record         {sum(r['exact'] for r in records) / n:6.1%}")
    print("  Per-field accuracy")
    for k in FIELDS:
        print(f"    {k:<12s}       {sum(r['fields'][k] for r in records) / n:6.1%}")

    # The failure mode T12/T13 were added to fix: the two "to"-marked values traded.
    # Counted over every record, not just failures, so it stays comparable across runs.
    swaps = [r for r in records if r["parsed"]
             and str(r["parsed"].get("customer", "")).strip() == r["target"]["carrier"]
             and str(r["parsed"].get("carrier", "")).strip() == r["target"]["customer"]]
    print(f"  customer/carrier swap  {len(swaps) / n:6.1%}  ({len(swaps)}/{n})")

    print("\n  by template")
    for tid in sorted({r["template_id"] for r in records}):
        rows = [r for r in records if r["template_id"] == tid]
        sw = sum(1 for r in rows if r in swaps)
        ex = sum(r["exact"] for r in rows)
        print(f"    {tid:<4s} exact {ex:3d}/{len(rows):<3d} {ex / len(rows):6.1%}   swaps {sw}")

    fails = [r for r in records if not r["exact"]]
    print(f"\n  failures: {len(fails)}/{n}")
    for kind, c in collections.Counter(r["failure"] for r in fails).most_common():
        print(f"    {c:3d}  {kind}")
    print(f"  by template: {dict(collections.Counter(r['template_id'] for r in fails))}")

    print("\n  --- representative successes ---")
    for r in [r for r in records if r["exact"]][:2]:
        print(f"    {r['sentence'][:64]}...\n      {r['raw'][:110]!r}")
    print("\n  --- representative failures (one per mode) ---")
    seen = set()
    for r in fails:
        if r["failure"] in seen:
            continue
        seen.add(r["failure"])
        print(f"    [{r['failure']}] {r['sentence'][:60]}...")
        print(f"      target: {json.dumps(r['target'])}")
        print(f"      raw   : {r['raw'][:200]!r}")
    print(f"\n  per-example records -> {out_path}")


if __name__ == "__main__":
    main()
