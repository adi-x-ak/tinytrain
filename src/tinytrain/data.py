"""Byte tokens and a reproducible random-window sampler."""

import hashlib

import torch
from torch import Tensor


class BatchStream:
    def __init__(self, text: bytes, context_length: int, seed: int) -> None:
        if len(text) <= context_length:
            raise ValueError("Corpus must contain more bytes than the context length")
        self.fingerprint = hashlib.sha256(text).hexdigest()
        self.tokens = torch.tensor(list(text), dtype=torch.long)
        self.context_length = context_length
        self.generator = torch.Generator().manual_seed(seed)

    def next(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        starts = torch.randint(
            len(self.tokens) - self.context_length, (batch_size,), generator=self.generator
        )
        indices = starts[:, None] + torch.arange(self.context_length + 1)
        windows = self.tokens[indices]
        return windows[:, :-1], windows[:, 1:]
