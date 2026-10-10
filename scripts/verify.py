"""Compare real Gloo workers and DDP against the same full-batch updates."""

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from tinytrain.distributed import process_group
from tinytrain.model import ModelConfig
from tinytrain.train import TrainConfig, Trainer, parameter_error

TEXT = b"The same global batch must produce the same optimizer update.\n" * 30
MODEL = ModelConfig(width=16, context_length=8, layers=1)
CONFIG = TrainConfig(global_batch_size=8, accumulation_steps=2)


def run(path: Path, sync: str) -> None:
    torch.set_num_threads(1)
    with process_group() as (rank, world_size):
        trainer = Trainer(
            MODEL, CONFIG, TEXT, rank=rank, world_size=world_size, sync=sync, dtype=torch.float64
        )
        for _ in range(5):
            trainer.step()
        trainer.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--sync", choices=("manual", "ddp"), default="manual")
    parser.add_argument("--output", type=Path, default=Path("results/verification.json"))
    args = parser.parse_args()
    if args.worker_output:
        run(args.worker_output, args.sync)
        return
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reference = root / "reference.pt"
        run(reference, "manual")
        expected = torch.load(reference, weights_only=True)
        errors = {}
        optimizer_errors = {}
        for sync in ("manual", "ddp"):
            result = root / f"{sync}.pt"
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--master-addr=127.0.0.1",
                    f"--master-port={port}",
                    "--nproc-per-node=2",
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    str(result),
                    "--sync",
                    sync,
                ],
                check=True,
                timeout=120,
            )
            actual = torch.load(result, weights_only=True)
            errors[sync] = parameter_error(expected["model"], actual["model"])
            optimizer_errors[sync] = max(
                (value - actual["optimizer"]["state"][index][name]).abs().max().item()
                for index, state in expected["optimizer"]["state"].items()
                for name, value in state.items()
            )
            if not all(
                math.isfinite(error) and error <= 1e-10
                for error in (errors[sync], optimizer_errors[sync])
            ):
                raise AssertionError(
                    f"{sync} diverged: weights={errors[sync]}, optimizer={optimizer_errors[sync]}"
                )
        report = {
            "torch_version": torch.__version__,
            "dtype": "float64",
            "world_size": 2,
            "steps": 5,
            "global_batch_size": CONFIG.global_batch_size,
            "accumulation_steps": CONFIG.accumulation_steps,
            "tolerance": 1e-10,
            "max_parameter_error": errors,
            "max_optimizer_state_error": optimizer_errors,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
