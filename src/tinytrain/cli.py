"""Training and sampling entry points."""

import argparse
import json
import time
from pathlib import Path

import torch

from tinytrain.distributed import process_group
from tinytrain.model import ModelConfig, Transformer
from tinytrain.train import TrainConfig, Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="A small Transformer with verifiable training.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Train byte-level next-token prediction on a text file"
    )
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("runs/default"))
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--context", type=int, default=64)
    train.add_argument("--batch", type=int, default=16, help="Global batch, across all workers")
    train.add_argument("--accumulate", type=int, default=1)
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--sync", choices=("manual", "ddp"), default="manual")
    train.add_argument("--resume", type=Path)
    train.add_argument("--eval-every", type=int, default=25)
    train.add_argument("--threads", type=int, default=1)
    sample = commands.add_parser("sample", help="Sample from a local checkpoint")
    sample.add_argument("checkpoint", type=Path)
    sample.add_argument("--prompt", default="ROMEO:\n")
    sample.add_argument("--tokens", type=int, default=200)
    sample.add_argument("--temperature", type=float, default=0.8)
    sample.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "sample":
        if not args.prompt or args.tokens < 0 or args.temperature <= 0:
            parser.error("Sampling requires a prompt, nonnegative tokens, and positive temperature")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        torch.set_num_threads(1)
        model = Transformer(ModelConfig(**state["model_config"])).eval()
        model.load_state_dict(state["model"])
        generator = torch.Generator().manual_seed(args.seed)
        tokens = torch.tensor([list(args.prompt.encode())])
        with torch.no_grad():
            for _ in range(args.tokens):
                logits = model(tokens[:, -model.config.context_length :])[:, -1]
                next_token = torch.multinomial(
                    (logits / args.temperature).softmax(-1), 1, generator=generator
                )
                tokens = torch.cat((tokens, next_token), dim=1)
        print(bytes(tokens[0].tolist()).decode(errors="replace"))
        return
    if min(args.steps, args.eval_every, args.threads) <= 0:
        parser.error("Steps, evaluation interval, and threads must be positive")
    torch.set_num_threads(args.threads)
    raw = args.data.read_bytes()
    split = int(len(raw) * 0.9)
    training, validation = raw[:split], raw[split:]
    if min(len(training), len(validation)) <= args.context:
        parser.error("Both the training and validation splits must exceed the context length")
    metrics = args.output / "metrics.jsonl"
    if not args.resume and metrics.exists():
        parser.error("Output already has metrics; choose a new directory or resume its checkpoint")
    config = ModelConfig(
        context_length=args.context, width=args.width, heads=args.heads, layers=args.layers
    )
    settings = TrainConfig(args.batch, args.accumulate, args.lr, seed=args.seed)
    with process_group() as (rank, world_size):
        trainer = Trainer(
            config, settings, training, rank=rank, world_size=world_size, sync=args.sync
        )
        if args.resume:
            trainer.load(args.resume)
        if trainer.steps >= args.steps:
            parser.error("Requested total steps must exceed the checkpoint step")
        if not rank:
            args.output.mkdir(parents=True, exist_ok=True)
        initial_loss = trainer.evaluate(validation)
        started = time.perf_counter()
        while trainer.steps < args.steps:
            loss = trainer.step()
            if trainer.steps % args.eval_every == 0 or trainer.steps == args.steps:
                validation_loss = trainer.evaluate(validation)
                if not rank:
                    record = {
                        "step": trainer.steps,
                        "train_loss": loss,
                        "validation_loss": validation_loss,
                        "initial_validation_loss": initial_loss,
                        "elapsed_seconds": time.perf_counter() - started,
                        "world_size": world_size,
                        "sync": args.sync,
                    }
                    print(json.dumps(record), flush=True)
                    with metrics.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                trainer.save(args.output / "checkpoint.pt")


if __name__ == "__main__":
    main()
