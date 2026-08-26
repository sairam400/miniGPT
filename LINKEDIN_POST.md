I've been trying to understand transformers beyond just calling an API, so I built a small GPT-style language model from scratch in PyTorch.

It's intentionally simple: character-level tokenization, causal self-attention, positional embeddings, transformer blocks, and an autoregressive generation loop. The smallest configuration has about 500K parameters and can be trained on CPU using the Tiny Shakespeare dataset.

The most useful part was implementing the attention mask and training loop myself. Seeing how the tensor shapes move through the model made the architecture much less abstract.

There's still plenty to improve. I'd like to add a KV cache, experiment with RoPE, and replace the character tokenizer with BPE next.

Code: https://github.com/sairam400/miniGPT

#PyTorch #MachineLearning #DeepLearning #Transformers #Python
