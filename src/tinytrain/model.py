"""A pre-normalized, decoder-only Transformer with tied token embeddings."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context_length, self.width, self.heads, self.layers) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Width must be divisible by the number of heads")


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        mask = torch.ones(config.context_length, config.context_length, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        q, k, v = [x.view(batch, length, self.heads, -1).transpose(1, 2) for x in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        attended = scores.softmax(dim=-1) @ v
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.mlp_norm = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.shape[1]
        if not 0 < length <= self.config.context_length:
            raise ValueError("Input length must be within the configured context window")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return F.linear(self.norm(hidden), self.tokens.weight)
