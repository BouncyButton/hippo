"""Tests for validation-only interactive prediction selection."""

from pathlib import Path

import pytest

from utils.view_model_predictions import (
    _checkpoint_selection,
    _prompt_validation_case,
    _validation_case_names,
    parse_args,
)


def test_cli_is_validation_only_and_requires_seed() -> None:
    args = parse_args(["--seed", "0"])
    assert args.split == "val"
    assert args.seed == 0
    with pytest.raises(SystemExit):
        parse_args(["--seed", "0", "--split", "train"])
    with pytest.raises(SystemExit):
        parse_args([])


def test_checkpoint_seed_and_fold_are_enforced() -> None:
    checkpoint = {
        "model": {"weight": object()},
        "run": {"dataset": "MSD", "seed": 1, "fold": 3},
    }
    state, fold = _checkpoint_selection(checkpoint, requested_seed=1)
    assert state is checkpoint["model"]
    assert fold == 3
    with pytest.raises(ValueError, match="records seed 1"):
        _checkpoint_selection(checkpoint, requested_seed=0)


def test_prompt_rejects_nonvalidation_id_then_accepts() -> None:
    answers = iter(["002", "17"])
    selected = _prompt_validation_case(
        ["hippocampus_001", "hippocampus_017"],
        seed=0,
        fold=0,
        input_fn=lambda _prompt: next(answers),
    )
    assert selected == "hippocampus_017"


def test_patient_017_is_in_fold_zero_validation() -> None:
    splits = Path(__file__).resolve().parents[1] / "datasets/Dataset101_MSD/splits_final.json"
    validation = _validation_case_names(splits, 0)
    assert "hippocampus_017" in validation
