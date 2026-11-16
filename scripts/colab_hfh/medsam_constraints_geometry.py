"""Physical spacing for HFH sampled planes; orientation remains source-axis based."""
import numpy as np

TENSOR_SPACING_MM = (2., 1., 1.)  # [slice, plane-row, plane-column]


def physical_affine(affine):
    value = np.array(affine, dtype=float, copy=True)
    if not np.allclose(np.linalg.norm(value[:3, :3], axis=0), 1):
        raise ValueError('Unexpected reference geometry affine')
    value[:3, :3] *= np.array(TENSOR_SPACING_MM)[None, :]
    return value


def depth_geometry(builder, labels, affine, config, *, fields=False):
    if fields or not np.isin(labels, (0, 1)).all():
        raise ValueError('HFH requires binary depth weights only')
    return builder(labels, physical_affine(affine), config, fields=False)
