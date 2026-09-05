import numpy as np

from surface_normal_ordinal.geometry import Interface, build_surface_samples, signed_distance


def test_signed_distance_is_negative_inside_and_spacing_aware() -> None:
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[3:6, 3:6, 3:6] = True
    sdf = signed_distance(mask, (1.0, 2.0, 3.0))
    assert sdf[4, 4, 4] < 0
    assert sdf[0, 0, 0] > 0
    assert np.isclose(sdf[2, 4, 4], 1.0)
    assert np.isclose(sdf[4, 2, 4], 2.0)


def test_outer_samples_are_oriented_from_foreground_to_background() -> None:
    labels = np.zeros((13, 13, 13), dtype=np.uint8)
    labels[4:9, 4:9, 4:9] = 1
    samples = build_surface_samples(
        labels,
        (1.0, 1.0, 1.0),
        interface=Interface.OUTER,
        radius_mm=2.0,
        max_points=None,
    )
    inside = samples.coordinates_at([-0.6])[:, 0]
    outside = samples.coordinates_at([0.6])[:, 0]
    rounded_inside = np.rint(inside).astype(int)
    rounded_outside = np.rint(outside).astype(int)
    assert np.mean(labels[tuple(rounded_inside.T)] > 0) > 0.9
    assert np.mean(labels[tuple(rounded_outside.T)] == 0) > 0.9


def test_ap_samples_are_oriented_from_anterior_to_posterior() -> None:
    labels = np.zeros((11, 11, 11), dtype=np.uint8)
    labels[3:8, 3:8, 3:5] = 1
    labels[3:8, 3:8, 5:8] = 2
    samples = build_surface_samples(
        labels,
        (1.0, 1.0, 1.0),
        interface=Interface.ANTERIOR_POSTERIOR,
        radius_mm=1.0,
        max_points=None,
    )
    inside = np.rint(samples.coordinates_at([-0.6])[:, 0]).astype(int)
    outside = np.rint(samples.coordinates_at([0.6])[:, 0]).astype(int)
    assert np.all(labels[tuple(inside.T)] == 1)
    assert np.all(labels[tuple(outside.T)] == 2)
