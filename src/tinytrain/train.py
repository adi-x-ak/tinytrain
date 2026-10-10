"""One global batch, one optimizer update, with explicit checkpoint state."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from tinytrain.data import BatchStream
from tinytrain.distributed import average_gradients
from tinytrain.model import ModelConfig, Transformer


@dataclass(frozen=True)
class TrainConfig:
    global_batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.global_batch_size, self.accumulation_steps) <= 0:
            raise ValueError("Batch size and accumulation steps must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")


class Trainer:
    def __init__(
        self,
        model_config: ModelConfig,
        config: TrainConfig,
        text: bytes,
        *,
        rank: int = 0,
        world_size: int = 1,
        sync: str = "manual",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if sync not in ("manual", "ddp"):
            raise ValueError("Synchronization must be manual or ddp")
        if world_size <= 0:
            raise ValueError("World size must be positive")
        if config.global_batch_size % (world_size * config.accumulation_steps):
            raise ValueError("Global batch must divide evenly across ranks and microbatches")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid rank")
        self.config, self.rank, self.world_size, self.sync = config, rank, world_size, sync
        torch.manual_seed(config.seed)
        self.model = Transformer(model_config).to(dtype=dtype)
        self.forward_model = (
            DistributedDataParallel(self.model) if sync == "ddp" and world_size > 1 else self.model
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
        )
        self.stream = BatchStream(text, model_config.context_length, config.seed + 1)
        self.steps = 0

    def step(self) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        inputs, targets = self.stream.next(self.config.global_batch_size)
        local_size = self.config.global_batch_size // self.world_size
        start = self.rank * local_size
        inputs, targets = inputs[start : start + local_size], targets[start : start + local_size]
        chunks = zip(
            inputs.chunk(self.config.accumulation_steps),
            targets.chunk(self.config.accumulation_steps),
            strict=True,
        )
        loss_value = torch.zeros((), dtype=self.model.tokens.weight.dtype)
        for index, (x, y) in enumerate(chunks):
            should_sync = index == self.config.accumulation_steps - 1
            context = (
                self.forward_model.no_sync()
                if isinstance(self.forward_model, DistributedDataParallel) and not should_sync
                else nullcontext()
            )
            with context:
                logits = self.forward_model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                (loss / self.config.accumulation_steps).backward()
                loss_value += loss.detach() / self.config.accumulation_steps
        if self.world_size > 1:
            if self.sync == "manual":
                average_gradients(self.model, self.world_size)
            dist.all_reduce(loss_value)
            loss_value /= self.world_size
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.steps += 1
        return loss_value.item()

    @torch.no_grad()
    def evaluate(self, text: bytes, batches: int = 8) -> float:
        if batches <= 0:
            raise ValueError("Evaluation batches must be positive")
        self.model.eval()
        stream = BatchStream(text, self.model.config.context_length, self.config.seed + 2)
        losses = []
        for _ in range(batches):
            x, y = stream.next(self.config.global_batch_size)
            logits = self.model(x)
            losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
        return sum(losses) / len(losses)

    def save(self, path: Path) -> None:
        if self.rank:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "dtype": str(self.model.tokens.weight.dtype),
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "corpus_sha256": self.stream.fingerprint,
            "sampler_rng": self.stream.generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["version"] != 1:
            raise ValueError("Unsupported checkpoint version")
        expected = {
            "dtype": str(self.model.tokens.weight.dtype),
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "corpus_sha256": self.stream.fingerprint,
        }
        for key, value in expected.items():
            if state[key] != value:
                raise ValueError(f"Checkpoint {key} does not match this run")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = state["steps"]
        self.stream.generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])


def parameter_error(left: dict[str, Tensor], right: dict[str, Tensor]) -> float:
    return max((left[name] - right[name]).abs().max().item() for name in left)
