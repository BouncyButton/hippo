"""Small geometry checks for the MSD cut-plane audit."""

import numpy as np

from .audit_ap_geometry import analyze_label


def test_exact_planar_cut():
    label = np.zeros((4, 8, 4), dtype=np.int8)
    label[:, 1:4, :] = 2
    label[:, 4:7, :] = 1
    result = analyze_label(label)
    assert result["cut_y"] == 3
    assert result["wrong_side_voxels"] == 0
    assert result["anterior_is_high_y"]
    assert result["contact_y_positions"] == [3]


def test_irregular_boundary_is_not_exact_plane():
    label = np.zeros((4, 8, 4), dtype=np.int8)
    label[:, 1:4, :] = 2
    label[:, 4:7, :] = 1
    label[0, 3, 0] = 1
    assert analyze_label(label)["wrong_side_voxels"] == 1
