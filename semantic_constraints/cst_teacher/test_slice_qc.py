from __future__ import annotations

import csv
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

from .score_slice_qc import main as score_main
from .score_slice_qc_ensemble import main as score_ensemble_main
from .run_slice_qc_experiment import run_fold
from .extract_slice_qc_features import extract_arrays
from .model import CSTDescriptorTeacher
from .slice_qc import (
    SliceQCNet,
    fit_temporal,
    load_patient_features,
    load_ridge,
    quality_loss,
    ridge_fit,
    ridge_predict,
    save_ridge,
    temporal_predict,
)


def feature_file(path, *, target=True, names=None):
    names = np.asarray(names if names is not None else [f"patient_{index}" for index in range(8)])
    rng = np.random.default_rng(7)
    count, slices = len(names), 6
    uncertainty = rng.normal(size=(count * slices, 4)).astype(np.float32)
    relationship = rng.normal(size=(count * slices, 9)).astype(np.float32)
    combined = np.concatenate((uncertainty, rng.normal(size=(count * slices, 8))), axis=1).astype(np.float32)
    payload = {
        "case_names": names,
        "slice_groups": np.repeat(names, slices),
        "slice_uncertainty": uncertainty,
        "slice_relationship_residuals": relationship,
        "slice_combined": combined,
    }
    if target:
        payload["slice_target"] = np.clip(0.4 + 0.2 * uncertainty[:, 0], 0, 1).astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def test_load_features_preserves_patient_slice_alignment_and_target_optional(tmp_path):
    path = feature_file(tmp_path / "features.npz")
    data = load_patient_features(path, require_target=True)
    assert data.count == 8 and data.slices == 6
    assert data.features["uncertainty"].shape == (8, 6, 4)
    assert data.features["portable"].shape == (8, 6, 13)
    assert data.features["combined"].shape == (8, 6, 12)
    assert data.target.shape == (8, 6)
    unlabeled = load_patient_features(feature_file(tmp_path / "unlabeled.npz", target=False), require_target=False)
    assert unlabeled.target is None
    with pytest.raises(ValueError, match="slice_target"):
        load_patient_features(tmp_path / "unlabeled.npz", require_target=True)


def test_bad_slice_group_order_rejected(tmp_path):
    path = feature_file(tmp_path / "features.npz")
    with np.load(path, allow_pickle=False) as old:
        arrays = {key: old[key] for key in old.files}
    arrays["slice_groups"][0] = "another_patient"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="patient-major"):
        load_patient_features(path, require_target=True)


def test_ridge_artifact_scores_unlabeled_new_patients(tmp_path, monkeypatch):
    training = load_patient_features(feature_file(tmp_path / "train.npz"), require_target=True)
    model = ridge_fit(training.features["portable"], training.target, alpha=1.0)
    prediction = ridge_predict(model, training.features["portable"])
    assert prediction.shape == (8, 6) and np.isfinite(prediction).all()
    artifact = tmp_path / "ridge_portable.npz"
    save_ridge(artifact, model, kind="portable", metadata={"training_patients": training.names.tolist()})
    kind, loaded, metadata = load_ridge(artifact)
    assert kind == "portable" and metadata["training_patients"] == training.names.tolist()
    np.testing.assert_allclose(ridge_predict(loaded, training.features[kind]), prediction)
    new_features = feature_file(tmp_path / "new.npz", target=False, names=["new_0", "new_1"])
    output = tmp_path / "scores.csv"
    monkeypatch.setattr(sys, "argv", ["score_slice_qc", "--model", str(artifact), "--features", str(new_features), "--output", str(output)])
    score_main()
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 and {row["case_name"] for row in rows} == {"new_0", "new_1"}
    monkeypatch.setattr(sys, "argv", ["score_slice_qc", "--model", str(artifact), "--features", str(training.source), "--output", str(output)])
    with pytest.raises(ValueError, match="model-training patients"):
        score_main()


def test_temporal_head_trains_and_scores_without_labels(tmp_path):
    data = load_patient_features(feature_file(tmp_path / "features.npz"), require_target=True)
    x, y = data.features["combined"], data.target
    artifact, epoch = fit_temporal(
        x,
        y,
        selection_train=np.arange(6),
        selection_val=np.arange(6, 8),
        seed=11,
        max_epochs=5,
        patience=2,
        batch_size=4,
    )
    assert 1 <= epoch <= 5
    prediction = temporal_predict(artifact, x)
    assert prediction.shape == y.shape and np.isfinite(prediction).all()
    assert np.all((prediction >= 0) & (prediction <= 1))
    checkpoint = tmp_path / "temporal.pt"
    torch.save(artifact, checkpoint)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    np.testing.assert_allclose(temporal_predict(reloaded, x), prediction)
    assert torch.isfinite(quality_loss(torch.randn(2, 6), torch.rand(2, 6)))
    with pytest.raises(ValueError, match="feature dimension"):
        temporal_predict(artifact, x[..., :3])
    with pytest.raises(ValueError, match="expected"):
        SliceQCNet(12)(torch.randn(8, 12))


def test_patient_grouped_experiment_produces_out_of_fold_predictions(tmp_path):
    data = load_patient_features(feature_file(tmp_path / "features.npz"), require_target=True)
    args = Namespace(outer_folds=2, seed=17, max_epochs=2, patience=2, device="cpu")
    report = run_fold(data, fold=0, args=args)
    assert len(report["outer_splits"]) == 2
    assert set(report["metrics"]) == {
        "entropy", "ridge_uncertainty", "ridge_portable", "ridge_combined",
        "temporal_portable", "temporal_combined",
    }
    for split in report["outer_splits"]:
        assert not (set(split["train_patients"]) & set(split["test_patients"]))
    assert np.asarray(report["predictions"]["temporal_combined"]).shape == (8, 6)


def test_matched_ensemble_scores_only_identically_ordered_new_patients(tmp_path, monkeypatch):
    training = load_patient_features(feature_file(tmp_path / "train.npz"), require_target=True)
    model = ridge_fit(training.features["portable"], training.target, alpha=1.0)
    first_model, second_model = tmp_path / "first.npz", tmp_path / "second.npz"
    for path in (first_model, second_model):
        save_ridge(path, model, kind="portable", metadata={"training_patients": training.names.tolist()})
    first_features = feature_file(tmp_path / "first_features.npz", target=False, names=["new_0", "new_1"])
    second_features = feature_file(tmp_path / "second_features.npz", target=False, names=["new_0", "new_1"])
    output = tmp_path / "ensemble.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["score_slice_qc_ensemble", "--models", str(first_model), str(second_model),
         "--features", str(first_features), str(second_features), "--output", str(output)],
    )
    score_ensemble_main()
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12 and all(row["ensemble_members"] == "2" for row in rows)
    misordered = feature_file(tmp_path / "misordered.npz", target=False, names=["new_1", "new_0"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["score_slice_qc_ensemble", "--models", str(first_model), str(second_model),
         "--features", str(first_features), str(misordered), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="identical patient and slice order"):
        score_ensemble_main()


def test_label_free_feature_extractor_produces_scorable_slice_arrays(tmp_path):
    teacher = CSTDescriptorTeacher(channels=(8, 16), heads=2).eval()
    image = np.random.default_rng(1).normal(size=(1, 16, 16, 16)).astype(np.float32)
    logits = np.random.default_rng(2).normal(size=(3, 16, 16, 16)).astype(np.float32)
    probabilities = torch.from_numpy(logits).softmax(dim=0).numpy()
    arrays = extract_arrays(
        teacher, image, probabilities, set_size=6, slab_depth=3, inplane_size=8
    )
    assert arrays["slice_uncertainty"].shape == (6, 4)
    assert arrays["slice_relationship_residuals"].shape == (6, 9)
    assert arrays["slice_combined"].shape == (6, 29)
    assert np.isfinite(arrays["slice_combined"]).all()
    path = tmp_path / "label_free.npz"
    np.savez_compressed(path, case_names=np.asarray(["new_patient"]), slice_groups=np.asarray(["new_patient"] * 6), **arrays)
    data = load_patient_features(path, require_target=False)
    assert data.count == 1 and data.target is None
    with pytest.raises(ValueError, match="sum to one"):
        extract_arrays(teacher, image, probabilities * 0.5, set_size=6, slab_depth=3, inplane_size=8)
