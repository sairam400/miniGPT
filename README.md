# miniGPT

A GPT implementation built from scratch in PyTorch. Every component is
hand-written — no HuggingFace, no magic. The architecture mirrors GPT-2;
only the scale is different.

## What's inside

```
mingpt/
├── model.py      # The full GPT: attention, transformer blocks, generation
├── config.py     # Model presets (tiny → medium)
├── dataset.py    # Character-level tokenizer and dataset
├── train.py      # Training loop with eval, LR schedule, checkpointing
└── generate.py   # Load checkpoint and sample text
```

## Quick start

```bash
pip install torch

# Train on Shakespeare (downloads automatically, ~1MB)
python train.py --preset tiny

# Generate text
python generate.py --prompt "ROMEO:" --tokens 300 --temperature 0.9
```

## Model presets

| Preset | Layers | Heads | Dim  | Params   | Hardware |
|--------|--------|-------|------|----------|----------|
| tiny   | 4      | 4     | 128  | ~500K    | CPU ✓    |
| small  | 6      | 6     | 192  | ~2M      | CPU (slow) |
| medium | 8      | 8     | 512  | ~20M     | GPU      |

## Architecture

```
Input tokens
     ↓
Token Embedding + Positional Embedding
     ↓
Dropout
     ↓
[TransformerBlock] × n_layer
   ├─ LayerNorm
   ├─ CausalSelfAttention   ← the key piece
   ├─ residual connection
   ├─ LayerNorm
   ├─ FeedForward (4× expand → GELU → project back)
   └─ residual connection
     ↓
LayerNorm
     ↓
Linear → logits (vocab_size)
     ↓
Cross-entropy loss (training) / softmax + sample (generation)
```

## Training your own corpus

```bash
python train.py --data path/to/your/text.txt --preset tiny
```

Any plain text file works — code, books, lyrics, logs.

## What to try next

- Swap learned positional embeddings for RoPE (rotary position encoding)
- Implement KV-cache for faster generation
- Add BPE tokenization instead of character-level
- Fine-tune on a domain-specific corpus
- Benchmark your model against nanoGPT on the same dataset

## Why build this

The best way to understand transformers is to write every line yourself.
Once you've done this, reading any LLM paper becomes straightforward —
you know exactly what each component does and why it's there.
