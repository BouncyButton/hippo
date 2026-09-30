"""Versioned, single-view augmentation with a checkpointed CPU RNG stream."""

import math
from itertools import product

import torch
import torch.nn.functional as F


MILD_V1 = {
    "name": "mild_v1", "identity_probability": 0.2,
    "spatial_probability_given_nonidentity": 0.7,
    "rotation_degrees": [-10.0, 10.0], "isotropic_scale": [0.9, 1.1],
    "translation_voxels_xyz": [-2.0, 2.0], "reflection": False,
    "contrast_probability_given_nonidentity": 0.5, "contrast_factor": [0.9, 1.1],
    "noise_probability_given_nonidentity": 0.15, "noise_std": [0.01, 0.03],
    "blur_probability_given_nonidentity": 0.15, "blur_sigma_voxels": [0.3, 0.7],
    "image_interpolation": "trilinear", "label_interpolation": "nearest",
    "padding": "zeros", "align_corners": False,
    "foreground_guard": "reject affine if transformed foreground bounding box leaves volume or a class disappears",
    "intensity_support": "nonzero input image after optional affine; no label-dependent intensity mask",
    "rng": "checkpointed translation_generator, CPU; mutually exclusive with translation augmentation",
}


def warp_pair(images, labels, *, angles, scale, translation_xyz):
    """Warp one pair in voxel coordinates; reject foreground-clipping transforms.

    Matrix coordinates are x=W, y=H, z=D, centered on the volume. Labels remain
    integer-valued. Volume changes from scaling/nearest resampling are expected.
    """
    if images.ndim != 5 or images.shape[:2] != (1, 1):
        raise ValueError("mild_v1 requires batch size 1 and one image channel.")
    if labels.shape != images.shape:
        raise ValueError("Image and label shapes must agree.")
    a, b, c = angles
    sa, ca, sb, cb, sc, cc = math.sin(a), math.cos(a), math.sin(b), math.cos(b), math.sin(c), math.cos(c)
    rx = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=torch.float64)
    ry = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=torch.float64)
    rz = torch.tensor([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=torch.float64)
    forward = scale * (rz @ ry @ rx)
    shift = torch.tensor(translation_xyz, dtype=torch.float64)
    size = torch.tensor(images.shape[-3:][::-1], dtype=torch.float64)
    foreground = torch.nonzero(labels[0, 0] != 0, as_tuple=False)
    if foreground.numel() == 0:
        return images, labels, False
    bounds = torch.stack((foreground.amin(0), foreground.amax(0))).cpu().double().flip(1)
    bounds -= (size - 1) / 2
    # Include full voxel cells, not just their centers, in the clipping guard.
    bounds[0] -= 0.5
    bounds[1] += 0.5
    corners = torch.stack([torch.stack([bounds[i, axis] for axis, i in enumerate(indices)])
                           for indices in product((0, 1), repeat=3)])
    moved = corners @ forward.T + shift
    if bool((moved.abs() > size / 2 - 0.5).any()):
        return images, labels, False
    inverse = torch.linalg.inv(forward)
    extent = torch.diag(size / 2)
    theta = torch.cat((torch.linalg.solve(extent, inverse @ extent),
                       torch.linalg.solve(extent, -inverse @ shift)[:, None]), dim=1)
    grid = F.affine_grid(theta.float().to(images.device)[None], images.shape, align_corners=False)
    warped_images = F.grid_sample(images.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_labels = F.grid_sample(labels.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False).to(labels.dtype)
    if not torch.equal(torch.unique(labels), torch.unique(warped_labels)):
        return images, labels, False
    return warped_images.to(images.dtype), warped_labels, True


def augment_mild(images, labels, *, generator):
    """Return one view, preserving global model/data RNG states and input tensors."""
    if images.ndim != 5 or images.shape[:2] != (1, 1) or labels.shape != images.shape:
        raise ValueError("mild_v1 requires matching [1,1,D,H,W] image/label tensors.")
    stats = {"identity": 0, "spatial_applied": 0, "spatial_rejected": 0,
             "contrast": 0, "noise": 0, "blur": 0}
    u = torch.rand(16, generator=generator).tolist()
    if u[0] < 0.2:
        stats["identity"] = 1
        return images, labels, stats
    if u[1] < 0.7:
        images, labels, applied = warp_pair(
            images, labels, angles=[math.radians(20 * x - 10) for x in u[2:5]],
            scale=0.9 + 0.2 * u[5], translation_xyz=[4 * x - 2 for x in u[6:9]],
        )
        stats["spatial_applied" if applied else "spatial_rejected"] = 1
    support = images != 0
    output = images.float()
    if u[9] < 0.5:
        output = output * (0.9 + 0.2 * u[10])
        stats["contrast"] = 1
    if u[11] < 0.15:
        noise = torch.randn(output.shape, generator=generator, dtype=torch.float32).to(output.device)
        output = output + noise * (0.01 + 0.02 * u[12])
        stats["noise"] = 1
    if u[13] < 0.15:
        coordinates = torch.arange(-2, 3, device=output.device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coordinates / (0.3 + 0.4 * u[14])).square())
        kernel /= kernel.sum()
        for axis in range(3):
            shape = [1, 1, 1, 1, 1]
            shape[axis + 2] = 5
            padding = [0, 0, 0]
            padding[axis] = 2
            output = F.conv3d(output, kernel.reshape(shape), padding=tuple(padding))
        stats["blur"] = 1
    output = torch.where(support, output, torch.zeros_like(output)).to(images.dtype)
    return output, labels, stats
