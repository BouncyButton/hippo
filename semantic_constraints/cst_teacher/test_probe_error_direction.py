"""Synthetic leakage and alignment tests for the directional probe."""

import csv

import numpy as np
import pytest

from .probe_error_direction import crossfit_direction, directional_metrics, load_qc_scores


def test_crossfit_direction_patient_disjoint():
    names = np.asarray([f"patient_{index}" for index in range(10)])
    labels = np.tile(np.array([1, 2, 1, 2], dtype=np.int8), (10, 1))
    features = np.zeros((10, 4, 2), dtype=np.float32)
    features[:, :, 0] = np.where(labels == 1, 1, -1)
    probability, splits = crossfit_direction(features, labels, names, fold=0, seed=7)
    assert probability.shape == labels.shape
    assert np.all(probability[labels == 1] > 0.5)
    assert np.all(probability[labels == 2] < 0.5)
    assert all(not (set(split["train_patients"]) & set(split["test_patients"])) for split in splits)


def test_direction_metrics_and_incomplete_qc(tmp_path):
    label = np.array([[1, 2, 1, 2]], dtype=np.int8)
    probability = np.array([[0.9, 0.1, 0.7, 0.3]], dtype=np.float32)
    metrics = directional_metrics(label, probability, np.ones_like(label, dtype=bool))
    assert metrics["accuracy"] == 1
    assert metrics["confident_directional_slices"] == 2
    path = tmp_path / "oof.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("fold", "case_name", "slice_index", "ridge_combined"))
        writer.writerow((0, "patient", 0, 0.9))
    with pytest.raises(ValueError, match="incomplete"):
        load_qc_scores(path, np.asarray(["patient"]), 2, 0)
