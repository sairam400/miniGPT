"""
miniGPT — a clean GPT implementation from scratch.
Architecture mirrors GPT-2 but scaled down for CPU training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal (autoregressive) mask.
    Each token can only attend to previous tokens — not future ones.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "Embedding dim must be divisible by number of heads"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Single matrix for Q, K, V projections (efficient)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask — lower triangular: token i can attend to tokens 0..i
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V in one shot, then split
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape into (B, n_head, T, head_dim) for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, n_head, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v                                     # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads

        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Expands to 4x embedding dim, applies GELU, projects back.
    This is where most of the model's 'memory' lives.
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block: LayerNorm → Attention → residual,
    then LayerNorm → FFN → residual.
    Pre-norm formulation (more stable than original paper).
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention sublayer + residual
        x = x + self.ffn(self.ln2(x))    # FFN sublayer + residual
        return x


class GPT(nn.Module):
    """
    The full GPT model.

    Components:
    - Token embedding table (vocab_size → n_embd)
    - Positional embedding table (block_size → n_embd)
    - Stack of TransformerBlocks
    - Final LayerNorm
    - Linear head projecting to vocab logits
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "tok_emb": nn.Embedding(config.vocab_size, config.n_embd),
            "pos_emb": nn.Embedding(config.block_size, config.n_embd),
            "drop":    nn.Dropout(config.dropout),
            "blocks":  nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            "ln_f":    nn.LayerNorm(config.n_embd),
        })

        # Language model head — projects embeddings to vocab logits
        # Weight tying: share weights between token embedding and lm_head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer["tok_emb"].weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        print(f"miniGPT initialized: {self.num_params():,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Token + positional embeddings
        tok = self.transformer["tok_emb"](idx)                         # (B, T, n_embd)
        pos = self.transformer["pos_emb"](torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.transformer["drop"](tok + pos)

        # Pass through transformer blocks
        for block in self.transformer["blocks"]:
            x = block(x)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)                                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate tokens given a context.

        Args:
            idx: (B, T) tensor of starting token indices
            max_new_tokens: how many tokens to generate
            temperature: >1 = more random, <1 = more focused
            top_k: if set, only sample from top k most likely tokens
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last token's logits

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx
