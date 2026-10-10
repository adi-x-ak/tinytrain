"""Run three seeded CPU training experiments and publish their measurements."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import torch

from tinytrain.model import ModelConfig, Transformer

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/shakespeare.txt"))
    parser.add_argument("--runs", type=Path, default=Path("runs/study"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    experiments = []
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for seed in (17, 42, 91):
        output = args.runs / str(seed)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tinytrain.cli",
                "train",
                "--data",
                str(args.data),
                "--output",
                str(output),
                "--steps",
                "600",
                "--eval-every",
                "50",
                "--seed",
                str(seed),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            timeout=180,
        )
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        experiments.append({"seed": seed, "metrics": records})
        axis.plot(
            [0] + [record["step"] for record in records],
            [records[0]["initial_validation_loss"]]
            + [record["validation_loss"] for record in records],
            label=f"Seed {seed}",
        )
        print(f"Seed {seed}: validation loss {records[-1]['validation_loss']:.4f}", flush=True)
    content = args.data.read_bytes()
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": 1,
        "corpus_sha256": hashlib.sha256(content).hexdigest(),
        "split_byte": int(len(content) * 0.9),
        "parameters": sum(p.numel() for p in Transformer(ModelConfig()).parameters()),
        "steps": 600,
        "global_batch_size": 16,
        "context_length": 64,
        "validation_batches": 8,
        "experiments": experiments,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    axis.set(xlabel="Optimizer updates", ylabel="Validation loss (nats / byte)")
    axis.set_title("TinyTrain · 120k-parameter Transformer · CPU")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.savefig(args.results / "loss.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
