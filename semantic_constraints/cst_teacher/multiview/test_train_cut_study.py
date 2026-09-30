"""Small correctness tests for cut baselines and relabeling diagnostics."""

import numpy as np

from .train_cut_study import foreground_dice, relabel_by_cut


def test_relabel_by_cut_preserves_union():
    original = np.zeros((3, 8, 3), dtype=np.int8)
    original[:, 2:6, :] = 1
    corrected = relabel_by_cut(original, 3)
    assert np.array_equal(original > 0, corrected > 0)
    assert np.all(corrected[:, 2:4, :] == 2)
    assert np.all(corrected[:, 4:6, :] == 1)


def test_foreground_dice_identity():
    label = np.zeros((3, 8, 3), dtype=np.int8)
    label[:, 2:4, :] = 2
    label[:, 4:6, :] = 1
    assert foreground_dice(label, label) == 1.0
