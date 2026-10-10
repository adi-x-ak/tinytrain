# Design and evidence

TinyTrain implements byte-level next-token training on CPU. The model uses
pre-normalized residual blocks, explicit causal attention, GELU MLPs, learned
positions, and tied input/output embeddings. There is no dropout or learning-rate
schedule. These choices keep the training state and numerical comparisons small.

## One batch, one update

Each worker generates the same global batch from an independently owned sampler
RNG, then takes its contiguous rank slice. This intentionally duplicates sampling
work: a small local correctness experiment benefits from an unambiguous ordering.
A large training system would shard its data pipeline instead.

For `W` workers and `A` equal microbatches per worker, each microbatch mean loss is
divided by `A` before backpropagation. Manual synchronization sums all gradients
and divides by `W`. Clipping happens after that reduction. The resulting gradient
is the mean over the original global batch, including its token dimension.

The manual reducer concatenates all gradients into one buffer and performs one
Gloo all-reduce per update. DDP skips synchronization until the final microbatch.
The verification script runs both implementations in **real processes** and
compares them against single-process updates with identical initial weights,
sampled bytes, optimizer settings, and accumulation. Float64 isolates algorithmic
errors from common float32 rounding differences. Equality is within a tolerance,
not promised to be bitwise across different reduction orders.

## Recovery

A checkpoint contains model weights, AdamW moments, update count, model/training
configuration, dtype, corpus hash, sampler state, and PyTorch CPU RNG state. It is
written to a temporary file and renamed after `torch.save` completes. Validation
uses a separate sampler and never consumes training batches.

The resume test compares six uninterrupted updates against three updates,
save/load into a fresh trainer, and three more updates. Model and optimizer
tensors match exactly in the tested float32 CPU environment. Changed corpus,
configuration, or dtype is rejected. Cross-version/hardware bitwise reproduction
is not guaranteed. Recovery between updates is supported; partial microbatch
recovery is outside this implementation.

## Training study

The corpus is the [tiny Shakespeare text distributed with char-rnn](https://github.com/karpathy/char-rnn/tree/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare).
The fetch script pins that revision and verifies SHA-256. Shakespeare's source
works are public domain; the repository's code license is separate from corpus
provenance. No corpus or trained checkpoint is embedded in Git history.

The first 90% of bytes train the model; the remaining 10% provide validation.
Random windows stay inside their respective partition. Every evaluation reuses
eight seeded batches of 16 windows of 64 bytes. That estimates loss on sampled
held-out windows; it is not an exhaustive corpus score. Each seed has its own
fixed validation windows. Training runs for 600 updates with AdamW (learning rate
0.001, betas 0.9/0.95, weight decay 0.01) and gradient norm clipping at 1.0.

The report stores all observations for three seeds. Elapsed time starts after
initialization and initial validation; it includes subsequent evaluation and
previous checkpoint writes. It is an observed training-loop duration, not an
isolated kernel benchmark. CPU timings depend on the machine and concurrent
workload. Two-worker tests establish update correctness, not distributed speedup.

This tiny model is an engineering demonstration. The study makes no benchmark
record, language-model quality, or frontier-scale training claim.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product attention.
- [PyTorch scaled-dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): independent output/gradient reference used in tests.
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html): reference synchronization path.
- [nanoGPT](https://github.com/karpathy/nanoGPT): prior work on readable small Transformer training systems. TinyTrain focuses on controlled update and recovery comparisons.
