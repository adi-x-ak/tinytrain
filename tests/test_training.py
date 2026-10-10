from dataclasses import replace

import pytest
import torch

from tinytrain.model import ModelConfig
from tinytrain.train import TrainConfig, Trainer, parameter_error

MODEL = ModelConfig(width=16, layers=1, context_length=8)
SETTINGS = TrainConfig(global_batch_size=8)
TEXT = b"Training should preserve the same global batch across workers.\n" * 20


def test_accumulated_updates_match_full_batch():
    full = Trainer(MODEL, SETTINGS, TEXT, dtype=torch.float64)
    accumulated = Trainer(MODEL, replace(SETTINGS, accumulation_steps=4), TEXT, dtype=torch.float64)
    for _ in range(4):
        assert full.step() == pytest.approx(accumulated.step(), abs=1e-12, rel=0)
    assert parameter_error(full.model.state_dict(), accumulated.model.state_dict()) < 1e-10


def test_resume_restores_optimizer_and_next_data_batch(tmp_path):
    continuous = Trainer(MODEL, SETTINGS, TEXT)
    interrupted = Trainer(MODEL, SETTINGS, TEXT)
    for _ in range(6):
        continuous.step()
    for _ in range(3):
        interrupted.step()
    path = tmp_path / "checkpoint.pt"
    interrupted.save(path)
    resumed = Trainer(MODEL, SETTINGS, TEXT)
    resumed.load(path)
    for _ in range(3):
        resumed.step()
    assert resumed.steps == continuous.steps
    assert parameter_error(continuous.model.state_dict(), resumed.model.state_dict()) == 0
    expected = continuous.optimizer.state_dict()["state"]
    actual = resumed.optimizer.state_dict()["state"]
    for index, values in expected.items():
        for name, value in values.items():
            torch.testing.assert_close(value, actual[index][name], rtol=0, atol=0)
    torch.testing.assert_close(continuous.stream.next(8)[0], resumed.stream.next(8)[0])


def test_checkpoint_rejects_different_corpus(tmp_path):
    trainer = Trainer(MODEL, SETTINGS, TEXT)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    other = Trainer(MODEL, SETTINGS, TEXT + b"changed")
    with pytest.raises(ValueError, match="corpus_sha256"):
        other.load(path)


def test_checkpoint_rejects_dtype_change(tmp_path):
    trainer = Trainer(MODEL, SETTINGS, TEXT, dtype=torch.float64)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    other = Trainer(MODEL, SETTINGS, TEXT)
    with pytest.raises(ValueError, match="dtype"):
        other.load(path)


def test_resume_restores_cpu_rng(tmp_path):
    trainer = Trainer(MODEL, SETTINGS, TEXT)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    expected = torch.rand(5)
    resumed = Trainer(MODEL, SETTINGS, TEXT)
    resumed.load(path)
    torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)


def test_evaluation_does_not_advance_training_sampler():
    trainer = Trainer(MODEL, SETTINGS, TEXT)
    before = trainer.stream.generator.get_state().clone()
    trainer.evaluate(TEXT, batches=2)
    torch.testing.assert_close(before, trainer.stream.generator.get_state(), rtol=0, atol=0)


def test_global_batch_must_divide_across_microbatches():
    with pytest.raises(ValueError, match="divide evenly"):
        Trainer(MODEL, replace(SETTINGS, accumulation_steps=3), TEXT)


def test_parameter_comparison_cannot_hide_nonfinite_values():
    good = {"first": torch.zeros(1), "second": torch.zeros(1)}
    bad = {"first": torch.zeros(1), "second": torch.tensor([float("nan")])}
    assert not parameter_error(good, bad) <= 1e-10
