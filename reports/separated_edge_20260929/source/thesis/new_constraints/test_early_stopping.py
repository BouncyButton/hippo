"""Stopping semantics, durable resume, and best-model export coverage."""

import csv
import json
import sys
from contextlib import ExitStack

import pytest
import torch
from torch.utils.data import DataLoader

from thesis.new_constraints.early_stopping import EarlyStopping
from thesis.new_constraints import train_swinunetr_constraints as trainer


def test_patience_uses_significant_reference_and_respects_minimum():
    policy = EarlyStopping(patience=2, min_delta=0.0005, min_epochs=5)
    for epoch, score in enumerate([0.7, 0.8, 0.8003, 0.8004], 1):
        policy.update(epoch, score)
        assert not policy.should_stop
    policy.update(5, 0.8002)
    assert policy.should_stop
    assert policy.reference_epoch == 2


def test_small_improvements_accumulate_against_reference():
    policy = EarlyStopping(patience=2, min_delta=0.0005, min_epochs=1)
    for epoch, score in enumerate([0.8, 0.8003, 0.8006], 1):
        policy.update(epoch, score)
    assert not policy.should_stop
    assert policy.reference_epoch == 3


def test_disabled_policy_never_stops_and_invalid_data_is_rejected():
    policy = EarlyStopping()
    for epoch in range(1, 51):
        policy.update(epoch, 0.8)
    assert not policy.should_stop
    with pytest.raises(ValueError, match="consecutive"):
        policy.update(52, 0.8)
    with pytest.raises(ValueError, match="finite"):
        policy.update(51, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        EarlyStopping(min_delta=float("nan"))


@pytest.mark.parametrize("interrupt", [False, True])
@pytest.mark.parametrize("drop_rate", [0.0, 0.1])
def test_training_restores_best_and_resume_preserves_stopping(monkeypatch, tmp_path, interrupt, drop_rate):
    pkl = tmp_path / "data.pkl"
    pkl.write_bytes(b"synthetic data for real provenance checks")
    splits = tmp_path / "splits.json"
    splits.write_text("[]")
    output = tmp_path / "run"
    records = [{"image": torch.ones(1, 4, 4, 4),
                "label": (torch.arange(64).reshape(1, 4, 4, 4) % 3).long()}]
    models = []

    def build_model(spatial_size, num_classes, device, *, drop_rate=0.0, activation_checkpointing=True):
        model = torch.nn.Sequential(
            torch.nn.Dropout(drop_rate), torch.nn.Conv3d(1, 3, 1)
        ).to(device)
        models.append(model)
        return model

    def build_data(args, generator, **kwargs):
        return (DataLoader(records, batch_size=1, shuffle=True, generator=generator),
                DataLoader(records, batch_size=1), 3, 1, 1)

    scores = [0.7, 0.8, 0.8003, 0.8002]
    calls = 0
    interrupted = False

    def evaluate(*args, **kwargs):
        nonlocal calls, interrupted
        if interrupt and calls == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        score = scores[calls] if calls < 4 else scores[2]
        calls += 1
        return {"val_dice_hard": score, "val_dice_soft": score}

    monkeypatch.setattr(trainer, "build_swinunetr", build_model)
    monkeypatch.setattr(trainer, "build_data", build_data)
    monkeypatch.setattr(trainer, "evaluate_validation_metrics", evaluate)
    monkeypatch.setattr(trainer, "collect_source_provenance", lambda: {"sha256": "f" * 64, "files": {}})
    monkeypatch.setattr(trainer, "validate_source_provenance_unchanged", lambda expected: None)
    argv = ["train", "--pkl", str(pkl), "--splits-json", str(splits),
            "--output-dir", str(output), "--constraint-set", "none", "--device", "cpu",
            "--epochs", "8", "--early-stopping-patience", "2",
            "--early-stopping-min-epochs", "4", "--learning-rate", "0.01",
            "--drop-rate", str(drop_rate)]
    if drop_rate:
        argv += ["--plain-tensors", "--no-activation-checkpointing"]
    monkeypatch.setattr(sys, "argv", argv)
    if interrupt:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            with ExitStack() as stack:
                trainer._main(stack)
        assert torch.load(output / "checkpoint_latest.pt", weights_only=True)["epoch"] == 2
        monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    with ExitStack() as stack:
        trainer._main(stack)
    final = json.loads((output / "final_metrics.json").read_text())
    best = torch.load(output / "checkpoint_best.pt", weights_only=True)
    latest = torch.load(output / "checkpoint_latest.pt", weights_only=True)
    exported = torch.load(output / "MSD_fold0/model.pt", weights_only=True)
    assert best["epoch"] == final["epoch"] == 3
    assert latest["epoch"] == final["early_stopping"]["stopped_epoch"] == 4
    assert final["checkpoint"] == "best"
    assert best["run"]["drop_rate"] == latest["run"]["drop_rate"] == drop_rate
    assert best["run"]["plain_tensors"] == bool(drop_rate)
    assert best["run"]["activation_checkpointing"] == (not bool(drop_rate))
    assert final["early_stopping"]["triggered"] is True
    assert all(torch.equal(exported[k], best["model"][k]) for k in exported)
    assert any(not torch.equal(exported[k], latest["model"][k]) for k in exported)
    assert len(list(csv.DictReader((output / "metrics.csv").open()))) == 4
    completed_calls = calls
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    with ExitStack() as stack:
        trainer._main(stack)
    assert calls == completed_calls + 1  # Only final evaluation, no new epoch.
    assert json.loads((output / "final_metrics.json").read_text()) == final
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--early-stopping-patience", "3"])
    with pytest.raises(ValueError, match="Resume configuration"):
        with ExitStack() as stack:
            trainer._main(stack)
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--drop-rate", "0.2"])
    with pytest.raises(ValueError, match="Resume configuration"):
        with ExitStack() as stack:
            trainer._main(stack)
