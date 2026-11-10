"""Plane-position channels on the native grid, without another view or encoder."""
import copy
import numpy as np
import torch
from scripts.depth_supervision.train import model_factory
from scripts.depth_supervision.utils import digest
from .views import FrozenAtlas


def native_plane_channels(shape, affine, center, basis, roi_indices):
    """Return [longitudinal/32 mm, signed depth/16 mm, ROI support].

    Use the exact source coordinates and sample membership used before chart
    rasterization. No MRI resampling, label access, UVs, or return grids.
    """
    shape = tuple(int(x) for x in shape)
    affine, center, basis = [np.asarray(x, dtype=float) for x in (affine, center, basis)]
    indices = np.asarray(roi_indices, dtype=np.int64)
    if len(shape) != 3 or affine.shape != (4, 4) or center.shape != (3,) or basis.shape != (3, 3):
        raise ValueError('Invalid native grid or plane frame')
    if not all(np.isfinite(x).all() for x in (affine, center, basis)):
        raise ValueError('Nonfinite native geometry')
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
        raise ValueError('Expected orthonormal plane basis')
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError('Expected unique nonempty native ROI indices')
    if indices.min() < 0 or indices.max() >= np.prod(shape):
        raise ValueError('ROI outside native model grid')
    ijk = np.stack(np.unravel_index(indices, shape), axis=1)
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    local = (world-center) @ basis
    channels = np.zeros((3, int(np.prod(shape))), dtype=np.float32)
    channels[0, indices] = local[:, 0]/32.
    channels[1, indices] = local[:, 2]/16.
    channels[2, indices] = 1.
    return channels.reshape(3, *shape)


class NativePlaneCache(FrozenAtlas):
    """Read only frozen guide metadata; do not load either chart representation."""
    def __init__(self, root, protocol, fraction):
        super().__init__(root, protocol, fraction)
        self.metadata = protocol['native_geometry']

    def get(self, item):
        path = self.directory/(item['case_name']+'.npz')
        if path.name not in self.verified:
            if digest(path) != self.manifest['artifacts'][path.name]:
                raise ValueError('Frozen atlas changed: '+str(path))
            self.verified.add(path.name)
        shape = tuple(item['image'].shape[1:])
        with np.load(path, allow_pickle=False) as archive:
            channels = native_plane_channels(shape, item['affine'], archive['center'],
                                             archive['basis'], archive['roi_indices'])
        native = np.asarray(self.metadata[item['case_name']]['native_shape'])
        before = (np.asarray(shape)-native)//2
        if (native <= 0).any() or (before < 0).any():
            raise ValueError('Invalid native field of view')
        roi = np.argwhere(channels[2] > 0)
        if (roi < before).any() or (roi >= before+native).any():
            raise ValueError('Guide support includes artificial padding')
        return dict(native_channels=channels)


class InformedSwin(torch.nn.Module):
    """One fully trainable SwinUNETR with four native-grid input channels."""
    def __init__(self, device):
        super().__init__()
        self.native = model_factory(device)
        replaced = []
        for name, layer in list(self.native.named_modules()):
            if not isinstance(layer, torch.nn.Conv3d) or layer.in_channels != 1:
                continue
            if layer.groups != 1:
                raise ValueError('Unexpected grouped input convolution')
            expanded = copy.deepcopy(layer)
            expanded.in_channels = 4
            weight = layer.weight.new_zeros((layer.out_channels, 4, *layer.kernel_size))
            weight[:, :1] = layer.weight.detach()
            expanded.weight = torch.nn.Parameter(weight)
            parent, attribute = name.rsplit('.', 1)
            setattr(self.native.get_submodule(parent), attribute, expanded)
            replaced.append(name)
        expected = {'swinViT.patch_embed.proj', 'encoder1.layer.conv1.conv', 'encoder1.layer.conv3.conv'}
        if set(replaced) != expected:
            raise ValueError('Unexpected SwinUNETR input paths: '+repr(replaced))
        # Preserve baseline initialization throughout the network. The additional
        # weights start at zero but receive gradients and train from the first step.
        self.expanded_inputs = replaced

    def forward(self, image, frame):
        channels = torch.as_tensor(frame['native_channels'], device=image.device, dtype=image.dtype)[None]
        if image.shape[0:2] != (1, 1) or channels.shape != (1, 3, *image.shape[2:]):
            raise ValueError('Expected one native MRI and three aligned channels')
        return self.native(torch.cat((image, channels), dim=1))
