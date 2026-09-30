"""Independent small-complex checks for graph and anatomical topology."""
import itertools

import numpy as np
import pytest

from evaluation.voxel_graph_anatomy import (
    analyze, cubical_stats, gf2_rank, graph_stats, partition_stats,
    square_complex_stats, square_corners, voxel_graph,
)


def betti(stats):
    return [stats[f"beta{i}"] for i in range(3)]


def independent_homology(mask):
    """Enumerate every voxel subcell and explicitly rank boundary matrices."""
    cells = [set() for _ in range(4)]
    for voxel in np.argwhere(mask):
        for offsets in itertools.product((0, 1, 2), repeat=3):
            cell = tuple(2 * voxel + offsets)
            cells[sum(c % 2 for c in cell)].add(cell)
    indices = [{c: i for i, c in enumerate(sorted(group))} for group in cells]
    ranks = [0]
    for dim in range(1, 4):
        columns = []
        for cell in cells[dim]:
            col = 0
            for axis in range(3):
                if cell[axis] % 2:
                    for delta in (-1, 1):
                        boundary = list(cell)
                        boundary[axis] += delta
                        col ^= 1 << indices[dim - 1][tuple(boundary)]
            columns.append(col)
        ranks.append(gf2_rank(columns))
    ranks.append(0)
    return [len(cells[d]) - ranks[d] - ranks[d + 1] for d in range(3)]


def test_face_graph_is_not_cubical_topology():
    mask = np.ones((2, 2, 2), bool)
    coords, edges, axes = voxel_graph(mask)
    stats, _ = graph_stats(coords, edges, axes, np.ones(3))
    assert (stats["nodes"], stats["edges"], stats["cycle_rank"]) == (8, 12, 5)
    assert betti(cubical_stats(mask)) == [1, 0, 0]
    assert np.all(np.abs(coords[edges[:, 0]] - coords[edges[:, 1]]).sum(1) == 1)
    assert len(set(map(tuple, edges))) == len(edges)
    assert stats["bridges"] == stats["articulation_voxels"] == 0


def test_corner_contact_does_not_create_graph_edge():
    mask = np.zeros((2, 2, 2), bool)
    mask[0, 0, 0] = mask[1, 1, 1] = True
    coords, edges, axes = voxel_graph(mask)
    stats, _ = graph_stats(coords, edges, axes, np.ones(3))
    assert stats["components_6"] == 2
    assert len(edges) == 0
    assert betti(cubical_stats(mask)) == [1, 0, 0]


def test_line_bridges_and_metric_spacing():
    coords, edges, axes = voxel_graph(np.ones((4, 1, 1)))
    stats, _ = graph_stats(coords, edges, axes, np.array([2., 1., 1.]))
    assert stats["bridges"] == 3
    assert stats["articulation_voxels"] == 2
    assert stats["diameter_lower_bound_mm"] == 6


def test_tunnel_and_cavity_are_distinct():
    ring = np.ones((3, 3, 1), bool)
    ring[1, 1, 0] = False
    shell = np.ones((3, 3, 3), bool)
    shell[1, 1, 1] = False
    assert betti(cubical_stats(ring)) == [1, 1, 0]
    assert betti(cubical_stats(shell)) == [1, 0, 1]


def test_cubical_duality_against_boundary_matrices():
    rng = np.random.default_rng(2309)
    for _ in range(35):
        mask = rng.random((3, 3, 3)) < rng.uniform(.1, .9)
        assert betti(cubical_stats(mask)) == independent_homology(mask)


def test_interface_disk_annulus_closed_sphere_and_pinch():
    squares = [square_corners(np.array([i, j, 0]), 2) for i in range(3) for j in range(3)]
    assert square_complex_stats(squares)["is_disk"]
    annulus = square_complex_stats(squares[:4] + squares[5:])
    assert betti(annulus) == [1, 1, 0]
    assert annulus["boundary_components"] == 2
    assert not annulus["is_disk"]
    shell = [square_corners(np.eye(3, dtype=int)[axis] * side, axis)
             for axis in range(3) for side in (0, 1)]
    assert betti(square_complex_stats(shell)) == [1, 0, 1]
    pinch = square_complex_stats([square_corners(np.array([i, i, 0]), 2) for i in (0, 1)])
    assert betti(pinch) == [1, 0, 0]
    assert pinch["nonmanifold_vertices"] == 1
    assert not pinch["is_disk"]


def test_induced_regions_and_serialized_edges(tmp_path):
    labels = np.ones((3, 4, 2), dtype=np.uint8)
    labels[:, :2] = 2
    result = analyze(labels, np.ones(3), np.eye(4), tmp_path)
    whole = result["regions"]["whole"]
    a, p = [result["regions"][r] for r in ("anterior", "posterior")]
    assert whole["nodes"] == a["nodes"] + p["nodes"]
    assert whole["edges"] == a["edges"] + p["edges"] + result["interface"]["cut_edges"]
    assert result["interface"]["is_disk"]
    assert result["interface"]["cut_edges"] == 6
    assert result["interface"]["axis_face_counts"] == [0, 6, 0]
    for region in ("whole", "anterior", "posterior"):
        with np.load(tmp_path / f"{region}.npz", allow_pickle=False) as graph:
            coords, edges = graph["coordinates_ijk"], graph["edges"]
            assert len(coords) == result["regions"][region]["nodes"]
            assert np.all(np.abs(coords[edges[:, 0]] - coords[edges[:, 1]]).sum(1) == 1)
            assert np.array_equal(labels[tuple(coords.T)], graph["node_labels"])


def test_misplaced_plane_keeps_topology():
    labels = np.ones((3, 8, 3), dtype=np.uint8)
    labels[:, :3] = 2
    shifted = labels.copy()
    shifted[:, :5] = 2
    first, second = [analyze(v, np.ones(3)) for v in (labels, shifted)]
    for region in ("whole", "anterior", "posterior"):
        assert betti(first["regions"][region]["cubical"]) == betti(second["regions"][region]["cubical"])
    assert first["interface"]["is_disk"] and second["interface"]["is_disk"]
    assert abs(first["interface"]["best_plane_cut"] - second["interface"]["best_plane_cut"]) == 2


def test_empty_and_invalid_masks():
    result = analyze(np.zeros((3, 3, 3), dtype=np.uint8), np.ones(3))
    assert result["regions"]["whole"]["components_6"] == 0
    assert not result["interface"]["is_disk"]
    with pytest.raises(ValueError):
        analyze(np.full((3, 3, 3), 4), np.ones(3))
