"""Interactive inference for a trained LoRA adapter.

A REPL for looking at single predictions by hand - the thing you want when a metric
moved and you would like to see WHY, rather than reading another aggregate number.

Nothing here is a new inference path. Model loading, the adapter attach, the prompt,
greedy decoding and the strict parse are all imported from the evaluators, so what you
see typing at this prompt is exactly what eval_adapter.py would have scored.
"""

import argparse

from eval_baseline import DEVICE, MODEL, build_prompt, generate, load_model, parse
from eval_adapter import load_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to a saved epochN.pt")
    args = ap.parse_args()

    tokenizer, model = load_model()
    load_adapter(model, args.adapter)
    print(f"\n{MODEL} + {args.adapter} | zero-shot | greedy | {DEVICE}")
    print("Enter a shipment sentence. Ctrl-D or 'quit' to exit.\n")

    while True:
        try:
            sentence = input("sentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not sentence:
            continue
        if sentence in ("quit", "exit"):
            return

        # [] = the same zero-shot prompt the baseline and adapter evaluators use.
        raw = generate(tokenizer, model, [build_prompt(sentence, [])])[0]
        print(f"  {raw!r}")
        # parse() is strict: one line, valid JSON, exactly the four fields. Say so when
        # it fails, since a plausible-looking output can still score as wrong.
        if parse(raw) is None:
            print("  ^ does not parse under the evaluator's rules (would score as invalid)")
        print()


if __name__ == "__main__":
    main()
