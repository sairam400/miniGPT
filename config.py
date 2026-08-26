"""
Configuration for miniGPT.
Presets for different scales — start with 'tiny' on CPU.
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 65        # set by dataset
    block_size: int = 128       # context length (tokens)
    n_layer: int = 4            # number of transformer blocks
    n_head: int = 4             # number of attention heads
    n_embd: int = 128           # embedding dimension
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 300
    eval_iters: int = 50


# Presets
PRESETS = {
    "tiny": GPTConfig(
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        batch_size=32,
        max_iters=3000,
    ),
    "small": GPTConfig(
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=192,
        batch_size=64,
        max_iters=5000,
    ),
    # For when you get GPU access
    "medium": GPTConfig(
        block_size=512,
        n_layer=8,
        n_head=8,
        n_embd=512,
        batch_size=128,
        max_iters=10000,
    ),
}
