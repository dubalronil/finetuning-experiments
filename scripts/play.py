"""Type a prompt, see what Qwen3-0.6B-Base continues with. Nothing else."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B-Base"
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
model.eval()

print(f"{MODEL} loaded on {DEVICE}. Blank line or Ctrl-C to quit.\n")

while True:
    prompt = input("prompt> ")
    if not prompt.strip():
        break
    # Let a typed \n become a real newline, so multi-line prompts can be entered.
    prompt = prompt.replace("\\n", "\n")

    # Text -> token ids. Note: no chat template, no special tokens. Raw continuation.
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    # print(f"[{inputs.input_ids.shape[1]} tokens: {tokenizer.convert_ids_to_tokens(inputs.input_ids[0])}]")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,      # set False for greedy decoding
            temperature=0.8,
            top_p=0.9,
        )

    # Slice off the prompt so we only print what the model added.
    continuation = tokenizer.decode(output[0][inputs.input_ids.shape[1]:])
    print(f"\n{continuation}\n")
