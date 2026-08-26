"""
Generate text from a trained miniGPT checkpoint.

Usage:
    python generate.py --prompt "ROMEO:" --tokens 200
    python generate.py --prompt "To be or" --temperature 0.8 --top_k 40
"""

import argparse
import torch
from model import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, default="checkpoint.pt")
    parser.add_argument("--prompt",      type=str, default="\n")
    parser.add_argument("--tokens",      type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k",       type=int, default=50)
    parser.add_argument("--samples",     type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    stoi = ckpt["vocab"]["stoi"]
    itos = ckpt["vocab"]["itos"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded model: {model.num_params():,} params\n")
    print("=" * 60)

    for i in range(args.samples):
        # Encode prompt
        prompt_tokens = torch.tensor(
            [stoi[c] for c in args.prompt if c in stoi],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        # Generate
        with torch.no_grad():
            out = model.generate(
                prompt_tokens,
                max_new_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        # Decode and print
        text = "".join([itos[int(t)] for t in out[0]])
        print(text)
        if args.samples > 1:
            print("=" * 60)


if __name__ == "__main__":
    main()
