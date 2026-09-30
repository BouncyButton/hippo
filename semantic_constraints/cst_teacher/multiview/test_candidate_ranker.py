"""Synthetic tri-planar candidate construction tests."""

import numpy as np
import torch

from .candidate_ranker import CSTCandidateRanker, build_candidate_batch, candidate_cuts_around, make_candidate_mask_slices


def test_candidate_views_preserve_union_and_move_interface():
    union = torch.ones(8, 8, 8, dtype=torch.bool)
    spec = {"sagittal": 4, "coronal": 4, "axial": 4}
    low = make_candidate_mask_slices(union, 2, spec, output_size=8)
    high = make_candidate_mask_slices(union, 5, spec, output_size=8)
    assert low.shape == high.shape == (12, 2, 8, 8)
    assert torch.all(low.sum(dim=1) == 1)
    assert torch.all(high.sum(dim=1) == 1)
    assert not torch.equal(low, high)


def test_candidate_batch_and_model():
    spec = {"coronal": 4}
    batch = {
        "case_name": ["a", "b"],
        "union": torch.ones(2, 8, 8, 8, dtype=torch.bool),
        "image_slabs": torch.randn(2, 4, 3, 8, 8),
        "metadata": torch.zeros(2, 4, 4),
    }
    cuts = candidate_cuts_around([3, 4], 8, (-1, 0, 1))
    elements, metadata = build_candidate_batch(batch, cuts, spec, output_size=8)
    assert elements.shape == (6, 4, 5, 8, 8)
    assert metadata.shape == (6, 4, 5)
    model = CSTCandidateRanker(channels=(4, 8), heads=2)
    assert model(elements, metadata).shape == (6,)
    assert np.array_equal(cuts[0], [2, 3, 4])
