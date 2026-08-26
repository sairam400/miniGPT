"""
Character-level tokenizer and dataset.

Reads raw text, builds a character vocabulary,
and returns (input, target) pairs for training.
"""

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """
    Character-level dataset.

    Takes a string of text and produces (x, y) pairs where:
      x = sequence of token indices of length block_size
      y = x shifted right by 1 (the 'next token' targets)
    """

    def __init__(self, text: str, block_size: int):
        # Build vocabulary from unique characters in text
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.block_size = block_size

        print(f"Dataset: {len(text):,} characters, {self.vocab_size} unique chars")

        # Mappings between chars and integers
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode the full text as a flat tensor of ints
        self.data = torch.tensor(
            [self.stoi[c] for c in text], dtype=torch.long
        )

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = chunk[:-1]   # input tokens
        y = chunk[1:]    # target tokens (shifted by 1)
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, tokens) -> str:
        return "".join([self.itos[int(t)] for t in tokens])


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_val_split(dataset: CharDataset, val_frac: float = 0.1):
    """Split dataset indices into train and validation sets."""
    n = len(dataset)
    split = int(n * (1 - val_frac))
    train_indices = list(range(split))
    val_indices = list(range(split, n))
    return train_indices, val_indices
