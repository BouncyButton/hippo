"""Small coefficient regressor and differentiable cubic B-spline geometry.

Spatial axes are array (i,j,k); vector components follow the same order.
Distances and velocities are in voxels (1 mm for the canonical MSD data).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def cubic_basis(size: int, spacing: float) -> torch.Tensor:
    count = math.ceil((size - 1) / spacing) + 4
    t = (torch.arange(size)[:, None] / spacing - (torch.arange(count)[None] - 1)).abs()
    return torch.where(t < 1, 2 / 3 - t.square() + t.pow(3) / 2,
                       torch.where(t < 2, (2 - t).pow(3) / 6, 0))


class CoefficientHead(nn.Module):
    def __init__(self, control_shape: tuple[int, int, int]):
        super().__init__()
        layers = []
        previous = 5
        for index, channels in enumerate((16, 32, 64)):
            layers += [nn.Conv3d(previous, channels, 3, stride=1 if index == 0 else 2, padding=1),
                       nn.GroupNorm(4, channels), nn.GELU(),
                       nn.Conv3d(channels, channels, 3, padding=1),
                       nn.GroupNorm(4, channels), nn.GELU()]
            previous = channels
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Conv3d(64, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.control_shape = control_shape

    def forward(self, inputs):
        features = F.interpolate(self.encoder(inputs), size=self.control_shape,
                                 mode="trilinear", align_corners=True)
        return 4 * torch.tanh(self.output(features))


class SplineWarp(nn.Module):
    def __init__(self, shape=(64, 64, 64), spacing=4, squarings=6):
        super().__init__()
        self.shape, self.squarings = tuple(shape), squarings
        for axis, size in enumerate(shape):
            self.register_buffer(f"basis_{axis}", cubic_basis(size, spacing))
        self.control_shape = tuple(getattr(self, f"basis_{a}").shape[1] for a in range(3))
        identity = torch.stack(torch.meshgrid(*(torch.arange(n) for n in shape), indexing="ij"), 0).float()
        self.register_buffer("identity", identity[None])
        self.register_buffer("scale", (torch.tensor(shape).float() - 1)[None, :, None, None, None])

    def interpolate(self, coefficients):
        x = torch.einsum("ia,nqabc->nqibc", self.basis_0, coefficients)
        x = torch.einsum("jb,nqibc->nqijc", self.basis_1, x)
        return torch.einsum("kc,nqijc->nqijk", self.basis_2, x)

    def sample(self, field, displacement):
        grid = (2 * (self.identity + displacement) / self.scale - 1).permute(0, 2, 3, 4, 1).flip(-1)
        return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def integrate(self, velocity):
        displacement = velocity / (2 ** self.squarings)
        for _ in range(self.squarings):
            displacement = displacement + self.sample(displacement, displacement)
        return displacement

    def forward(self, coefficients, source_sdf, taper, support):
        velocity = self.interpolate(coefficients) * taper
        backward = self.integrate(-velocity)
        # Exact preservation of the initial shape outside the prediction bands.
        warped = torch.where(support.bool(), self.sample(source_sdf, backward), source_sdf)
        return warped, backward, velocity


def jacobian_determinant(displacement):
    # Central differences, one-sided at volume edges, matching numpy.gradient.
    columns = torch.gradient(displacement, dim=(2, 3, 4), edge_order=1)
    matrix = torch.stack(columns, -1).permute(0, 2, 3, 4, 1, 5)
    matrix = matrix + torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    a, b, c = matrix[..., 0, :], matrix[..., 1, :], matrix[..., 2, :]
    return (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)


def refinement_loss(warped, target_sdf, truth, support, velocity, determinant):
    reduce_axes = (1, 2, 3, 4)
    count = support.sum(reduce_axes).clamp_min(1)
    distance = (F.smooth_l1_loss(warped, target_sdf, reduction="none") * support).sum(reduce_axes) / count
    probability = torch.sigmoid(-warped / .5)
    overlap = 1 - (2 * (probability * truth).sum(reduce_axes) + 1) / (
        probability.sum(reduce_axes) + truth.sum(reduce_axes) + 1)
    movement = velocity.square().sum(reduce_axes) / (3 * count)
    smooth = sum(velocity.diff(dim=a).square().sum(reduce_axes) / (3 * count) for a in (2, 3, 4))
    compression = ((.3 - determinant).relu().square() * support[:, 0]).sum((1, 2, 3)) / count
    total = distance + .25 * overlap + .005 * movement + .02 * smooth + compression
    parts = dict(distance=distance.mean(), overlap=overlap.mean(), movement=movement.mean(),
                 smoothness=smooth.mean(), compression=compression.mean())
    return total.mean(), {k: float(v.detach()) for k, v in parts.items()}
