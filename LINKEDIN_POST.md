I built a GPT-style language model from scratch with PyTorch.

Instead of relying on Hugging Face or a prebuilt transformer, I implemented the core pieces directly: causal multi-head self-attention, transformer blocks, positional embeddings, residual connections, a feed-forward network, training and evaluation loops, checkpointing, and autoregressive text generation.

The project includes three model sizes, starting with a CPU-friendly configuration of roughly 500K parameters. It uses character-level tokenization and can train on Shakespeare or any plain-text corpus.

Building each layer made concepts such as causal masking, weight tying, learning-rate warmup, and next-token prediction much more concrete. My next experiments will be rotary positional embeddings, a KV cache for faster generation, and BPE tokenization.

Repository: https://github.com/sairam400/miniGPT

#PyTorch #MachineLearning #DeepLearning #GenerativeAI #Transformers #Python #BuildInPublic
