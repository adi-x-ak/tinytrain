"""Explicit CPU gradient averaging; torchrun owns process startup."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


@contextmanager
def process_group() -> Iterator[tuple[int, int]]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        yield 0, 1
        return
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        yield dist.get_rank(), dist.get_world_size()
    finally:
        dist.destroy_process_group()


def average_gradients(model: nn.Module, world_size: int) -> None:
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat)
    flat.div_(world_size)
    offset = 0
    for gradient in gradients:
        size = gradient.numel()
        gradient.copy_(flat[offset : offset + size].view_as(gradient))
        offset += size
