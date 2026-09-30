import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import laplacian

from evaluation.audit_graph_property_inventory import (
    cross_sections, digital_local_thickness, region_properties, spectral_fields,
)
from evaluation.voxel_graph_anatomy import adjacency, voxel_graph
from evaluation.summarize_graph_property_inventory import best_threshold


def test_spectral_fields_match_exact_heat_kernel():
    points, edges, _ = voxel_graph(np.ones((4, 2, 1)))
    g = adjacency(len(points), edges)
    values, vectors, hks, times, diagnostics = spectral_fields(g, points)
    np.testing.assert_allclose(values, np.linalg.eigvalsh(laplacian(g).toarray()), atol=1e-12)
    for i, t in enumerate(times):
        np.testing.assert_allclose(hks[:, i], np.diag(expm(-t * laplacian(g).toarray())), atol=1e-10)
    assert diagnostics['residual_max'] < 1e-10


def test_truncated_heat_kernel_respects_tail_bound():
    points, edges, _ = voxel_graph(np.ones((4, 4, 2)))
    g = adjacency(len(points), edges)
    _, _, hks, times, diagnostics = spectral_fields(g, points, modes=12)
    for i, t in enumerate(times):
        error = np.diag(expm(-t * laplacian(g).toarray())) - hks[:, i]
        assert error.min() >= -1e-9
        assert error.max() <= diagnostics['hks_absolute_tail_bound_max'][i] + 1e-9


def test_exact_distance_and_disconnected_semantics(tmp_path):
    mask = np.ones((5, 1, 1), bool)
    stats = region_properties(mask, tmp_path / 'line.npz')
    assert stats['diameter_largest_component'] == 4
    assert stats['leaves'] == 2
    assert stats['surface_area_voxel_faces'] == 22
    assert stats['diameter_path_boundary_fraction'] == 1
    with np.load(tmp_path / 'line.npz') as z:
        np.testing.assert_array_equal(z['eccentricity_within_component'], [4, 3, 2, 3, 4])
    mask[2] = False
    stats = region_properties(mask, tmp_path / 'disconnected.npz')
    assert not stats['global_diameter_finite']
    assert stats['lambda2_full_graph'] == 0
    assert stats['diameter_max_over_components'] == 1


def test_graph_depth_is_zero_at_boundary(tmp_path):
    mask = np.ones((5, 5, 5), bool)
    region_properties(mask, tmp_path / 'cube.npz')
    with np.load(tmp_path / 'cube.npz') as z:
        i = np.flatnonzero((z['coordinates_ijk'] == [2, 2, 2]).all(1))[0]
        assert z['depth_hops'][i] == 2
        assert z['edt_background_center_radius'][i] == 3


def test_local_thickness_is_covering_ball_not_center_radius():
    points = np.column_stack((np.arange(5), np.zeros(5), np.zeros(5)))
    thickness = digital_local_thickness(points, np.array([1., 2., 3., 2., 1.]))
    np.testing.assert_array_equal(thickness, np.full(5, 6.))


def test_slab_component_graph_is_chain_for_prism():
    rows, graph = cross_sections(np.ones((3, 5, 2), bool))
    assert graph == dict(nodes=5, edges=4, components=1, branch_nodes=0, branch_y=[], leaves=2, cycle_rank=0)
    assert all(r['area'] == 6 and r['perimeter'] == 10 for r in rows)


def test_coordinate_threshold_does_not_split_ties():
    result = best_threshold(np.array([0., 0., 1., 1.]), np.array([False, True, True, True]))
    assert result['errors'] == 1
    # A pure-A result ties with the interior threshold; neither may split x=0.
    predicted = np.array([0., 0., 1., 1.]) >= result['threshold']
    assert predicted[0] == predicted[1]
    assert np.count_nonzero(predicted != [False, True, True, True]) == result['errors']


def test_flat_partition_can_cross_three_times_on_simple_path():
    # A valid simple path in a solid 3x2x1 prism, with an exactly planar A/P cut.
    path = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [2,0,0], [2,1,0]])
    assert len(np.unique(path, axis=0)) == len(path)
    assert np.all(np.abs(np.diff(path, axis=0)).sum(1) == 1)
    labels = path[:, 1]
    assert np.count_nonzero(np.diff(labels)) == 3
