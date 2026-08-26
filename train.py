"""
Training script for miniGPT.

Usage:
    python train.py                          # uses built-in Shakespeare sample
    python train.py --data path/to/text.txt  # your own corpus
    python train.py --preset small           # bigger model
"""

import os
import sys
import math
import argparse
import urllib.request

import torch
from torch.utils.data import DataLoader, Subset

from config import GPTConfig, PRESETS
from dataset import CharDataset, load_text, train_val_split
from model import GPT


# ── Data ──────────────────────────────────────────────────────────────────────

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

def get_data(path=None):
    if path and os.path.exists(path):
        print(f"Loading data from {path}")
        return load_text(path)

    # Fall back to Shakespeare
    local = "shakespeare.txt"
    if not os.path.exists(local):
        print("Downloading Shakespeare dataset (~1MB)...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, local)
        print("Done.")
    return load_text(local)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    model.eval()
    results = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        results[split] = sum(losses) / len(losses)
    model.train()
    return results


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step, config):
    """Cosine decay with linear warmup."""
    warmup = 100
    if step < warmup:
        return config.learning_rate * step / warmup
    progress = (step - warmup) / (config.max_iters - warmup)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=str, default=None)
    parser.add_argument("--preset",  type=str, default="tiny", choices=PRESETS.keys())
    parser.add_argument("--out",     type=str, default="checkpoint.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config
    config = PRESETS[args.preset]

    # Data
    text = get_data(args.data)
    dataset = CharDataset(text, config.block_size)
    config.vocab_size = dataset.vocab_size

    train_idx, val_idx = train_val_split(dataset)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size, shuffle=False, drop_last=True
    )

    # Model
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=0.1
    )

    # Training loop
    print(f"\nTraining for {config.max_iters} steps...\n")
    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    for step in range(config.max_iters):
        # Refresh iterator if exhausted
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Learning rate schedule
        lr = get_lr(step, config)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward + backward
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluation
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            metrics = estimate_loss(model, train_loader, val_loader,
                                    config.eval_iters, device)
            print(f"step {step:4d} | lr {lr:.2e} | "
                  f"train loss {metrics['train']:.4f} | "
                  f"val loss {metrics['val']:.4f}")

            if metrics["val"] < best_val_loss:
                best_val_loss = metrics["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config,
                    "vocab": {"stoi": dataset.stoi, "itos": dataset.itos},
                }, args.out)
                print(f"  → checkpoint saved to {args.out}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
