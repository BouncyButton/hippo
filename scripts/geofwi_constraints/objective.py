"""Extend the frozen UNETR region loss with the established binary auxiliaries."""
import math
import numpy as np
import torch
import torch.nn.functional as F
import rescue_objective as rescue
from scripts.adni_constraints import objectives as binary
from scripts.capped_depth_delay import objectives as combined


def depth_loss(logits, labels, weights):
    if tuple(logits.shape) != (1, 2, 96, 96, 96) or tuple(weights.shape) != (96, 96, 96):
        raise ValueError("Expected native seismic depth geometry")
    return (F.cross_entropy(logits.float(), labels[:, 0].long(), reduction="none")[0] * weights).sum()


def training_loss(outputs, labels, arm, weights, scale, sdf=None, criterion=None,
                  depth=None, sdf_scale=0., *, policy="delayed_5_of_9"):
    if criterion is not None:
        raise ValueError("UNETR has no native deep supervision")
    if arm in ("dice", "dice_bands"):
        return rescue.training_loss(outputs, labels, arm,
                                    {"bands": weights["bands"]} if arm == "dice_bands" else {},
                                    scale)
    loss = rescue.training_loss(outputs, labels, "dice", {}, 0.)
    if "bands" in arm:
        loss = loss + float(scale) * weights["bands"] * binary.bands()(outputs.float(), labels).loss
    if "edge" in arm:
        loss = loss + float(scale) * weights["edge"] * binary.BinaryEdge()(outputs.float(), labels)
    if "sdf" in arm:
        if policy == "delayed_5_of_9" and (scale != 1. or not 0 < sdf_scale <= 1.):
            raise ValueError("Delayed SDF must follow the early ramp")
        coefficient = weights["sdf"] * (sdf_scale if policy == "delayed_5_of_9" else 1.)
        loss = loss + float(scale) * coefficient * binary.sdf_loss(outputs.float(), labels, sdf)
    if "depth" in arm:
        loss = loss + (weights["depth"] * scale) * depth_loss(outputs.float(), labels, depth)
    return loss


def calibration_case(outputs, labels, arm, sdf=None, criterion=None, depth=None):
    if arm == "dice_bands":
        return rescue.calibration_case(outputs, labels, arm)
    if criterion is not None:
        raise ValueError("UNETR-only calibration")
    z = outputs.detach().float().requires_grad_(True)
    gradients = {"dice": torch.autograd.grad(training_loss(z, labels, "dice", {}, 0.), z)[0]}
    dr = gradients["dice"].square().mean().sqrt()
    row = dict(dice_rms=float(dr))
    if "bands" in arm:
        band = binary.bands()(z, labels)
        if not bool(band.details["valid"].all()):
            raise ValueError("Invalid calibration bands")
        gradients["bands"] = torch.autograd.grad(band.loss, z)[0]
        br = gradients["bands"].square().mean().sqrt()
        row.update(bands_rms=float(br), bands_over_dice=float(br / dr))
        if "edge" in arm or "sdf" in arm:
            row["dice_bands_cross_mean"] = float((gradients["dice"] * gradients["bands"]).mean())
    if "edge" in arm:
        gradients["edge"] = torch.autograd.grad(binary.BinaryEdge()(z, labels), z)[0]
        er = gradients["edge"].square().mean().sqrt()
        row["edge_rms"] = float(er)
        if "bands" not in arm:
            row["edge_over_base"] = float(er / dr)
    if "sdf" in arm:
        gradients["sdf"] = torch.autograd.grad(binary.sdf_loss(z, labels, sdf), z)[0]
        row["sdf_rms"] = float(gradients["sdf"].square().mean().sqrt())
        if "edge" in arm:
            row["dice_edge_cross_mean"] = float((gradients["dice"] * gradients["edge"]).mean())
            if "bands" in arm:
                row["bands_edge_cross_mean"] = float((gradients["bands"] * gradients["edge"]).mean())
    if "depth" in arm:
        gradients["depth"] = torch.autograd.grad(depth_loss(z, labels, depth), z)[0]
        row["depth_rms"] = float(gradients["depth"].square().mean().sqrt())
        row["_depth_pair_means"] = {a + ":" + b: float((ga * gb).mean())
                                   for a, ga in gradients.items() for b, gb in gradients.items()}
    for key, value in row.items():
        if isinstance(value, dict):
            if not all(math.isfinite(v) for v in value.values()):
                raise ValueError("Nonfinite calibration Gram matrix")
        elif not math.isfinite(value) or (not key.endswith("cross_mean") and value <= 0):
            raise ValueError("Degenerate calibration")
    return row


def xy_affine(affine, seed, update, size):
    """Map D4-augmented voxel coordinates to the original source voxel frame."""
    from xy_augmentation import transform_id
    index = transform_id(seed, update)
    rotation = np.array([[0., 1., 0., 0.], [-1., 0., 0., size - 1.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
    transform = np.linalg.matrix_power(rotation, index % 4)
    if index >= 4:
        flip = np.eye(4)
        flip[0, 0], flip[0, 3] = -1., size - 1.
        transform = transform @ flip
    return np.asarray(affine) @ transform
