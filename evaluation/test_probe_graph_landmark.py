"""Landmark extraction invariance, quotient branching, and target isolation."""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components, shortest_path

from evaluation.voxel_graph_anatomy import adjacency, voxel_graph
from evaluation.probe_graph_landmark import (
    extract_features, quotient_profiles, target_cut, design, fixed_predictions,
    prior_values, fit, predict, RULES,
)


def fields(mask):
    points, edges, _ = voxel_graph(mask)
    matrix = adjacency(len(points), edges)
    _, component = connected_components(matrix, directed=False)
    degree = np.diff(matrix.indptr)
    largest = np.flatnonzero(component == np.bincount(component).argmax())
    distances = shortest_path(matrix, unweighted=True, directed=False)
    pair = np.unravel_index(distances[np.ix_(largest, largest)].argmax(), (len(largest), len(largest)))
    start, end = largest[list(pair)]
    if points[start, 1] > points[end, 1]:
        start, end = end, start
    y = points[largest, 1]
    u = (y-y.min()) / (y.max()-y.min())
    edt = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1, 1:-1][tuple(points.T)]
    return dict(coordinates_ijk=points, degree=degree,
        depth_hops=distances[:, degree < 6].min(1),
        digital_local_thickness_diameter=2*edt,
        betweenness_32_sources=np.zeros(len(points)),
        eccentricity_within_component=np.where(np.isfinite(distances), distances, 0).max(1),
        lcc_node_ids=largest, harmonic_lcc=u, fiedler01_lcc=u,
        distance_to_endpoint_hops=distances[[start, end]].T)


def test_translation_and_semantic_labels_do_not_change_features():
    mask = np.ones((3, 14, 3), bool)
    whole = fields(mask)
    base = extract_features(whole, mask.shape)
    offset = np.array([2, 4, 3])
    shifted = dict(whole, coordinates_ijk=whole["coordinates_ijk"] + offset)
    other = extract_features(shifted, tuple(np.array(mask.shape) + 2*offset))
    for key in ("position", "shape", "graph", "rules"):
        np.testing.assert_allclose(base[key], other[key], atol=1e-12)
    np.testing.assert_array_equal(base["candidates"]+offset[1], other["candidates"])
    # Extra semantic data cannot influence this extractor's explicitly selected fields.
    first = extract_features(dict(whole, node_labels=np.ones(len(whole["degree"]))), mask.shape)
    second = extract_features(dict(whole, node_labels=np.full(len(whole["degree"]), 2)), mask.shape)
    for key in ("position", "shape", "graph", "rules"):
        np.testing.assert_array_equal(first[key], second[key])


def test_disconnected_intrinsic_fields_are_finite_and_explicit():
    mask = np.zeros((6, 14, 5), bool)
    mask[:3, :, :3] = True
    mask[5, 7, 4] = True
    result = extract_features(fields(mask), mask.shape)
    assert np.isfinite(result["graph"]).all()
    i = list(result["graph_names"]).index("intrinsic_valid_fraction_value")
    assert result["graph"][:, i].min() < 1


def test_quotient_counts_branch_and_ignores_invalid_bins():
    edges = np.array([[0, 1], [1, 2], [1, 3], [2, 4], [3, 4]])
    bins = np.array([0, 1, 2, 2, 3])
    components, branch = quotient_profiles(edges, bins, 4)
    np.testing.assert_array_equal(components, [1, 1, 2, 1])
    assert branch[1] == 1
    bins[0] = -1
    components, branch = quotient_profiles(edges, bins, 4)
    assert components[0] == branch[0] == 0


def test_target_matches_exhaustive_coronal_fit():
    rng = np.random.default_rng(47)
    for _ in range(10):
        labels = rng.integers(0, 3, size=(3, 12, 3))
        expected = min(range(1, 12), key=lambda cut: np.count_nonzero(
            (labels > 0) & (labels != np.where(np.arange(12)[None, :, None] >= cut, 1, 2))))
        assert target_cut(labels) == expected


def test_shuffle_preserves_shape_and_graph_distribution():
    x = extract_features(fields(np.ones((3, 14, 3), bool)), (3, 14, 3))
    x.update(name="synthetic", target=7)
    first, shuffled = design(x, "combined"), design(x, "shuffled_graph")
    prefix = x["position"].shape[1] + x["shape"].shape[1]
    np.testing.assert_array_equal(first[:, :prefix], shuffled[:, :prefix])
    np.testing.assert_array_equal(np.sort(first[:, prefix:], axis=0), np.sort(shuffled[:, prefix:], axis=0))
    assert not np.array_equal(first[:, prefix:], shuffled[:, prefix:])
    predictions = fixed_predictions(x, prior_values([x]))
    assert all(cut in x["candidates"] for cut in predictions.values())
    assert len(predictions) == len(RULES) + 2


def test_readout_prediction_never_uses_test_target():
    x = extract_features(fields(np.ones((3, 14, 3), bool)), (3, 14, 3))
    cases = [dict(x, name=f"case{i}", target=7) for i in range(8)]
    model = fit(cases, "position")
    first = predict(dict(cases[0], target=3), model, "position")
    second = predict(dict(cases[0], target=10), model, "position")
    assert first == second
