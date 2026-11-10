"""Matched plane/unwrapped branches whose auxiliary convolutions see MRI only."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from scripts.unwrapped_swin.models import DualViewSwin
from scripts.depth_supervision.utils import digest
from .views import FrozenAtlas


MRI_ARMS = {
    'plane_mri_context': 'plane_position', 'unfold_mri_context': 'unfold_position',
    'plane_mri_attention': 'plane_attention', 'unfold_mri_attention': 'unfold_attention',
    'native_mri_context': 'plane_position', 'native_mri_attention': 'plane_attention',
}


class MRIOnlyEncoder(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.fine, self.context, self.combine = original.fine, original.context, original.combine
        self.reduced_inputs = []
        for name, layer in list(self.named_modules()):
            if isinstance(layer, torch.nn.Conv3d) and layer.in_channels == 4:
                reduced = copy.deepcopy(layer)
                reduced.in_channels = 1
                reduced.weight = torch.nn.Parameter(layer.weight[:, :1].detach().clone())
                parent, attribute = name.rsplit('.', 1)
                setattr(self.get_submodule(parent), attribute, reduced)
                self.reduced_inputs.append(name)
        if set(self.reduced_inputs) != {'fine.0.0', 'context.0.0'}:
            raise ValueError('Unexpected auxiliary input convolutions')

    def forward(self, packed):
        # Support defines valid samples for global pooling; it is never a learned
        # input channel. Position/depth planes are not used anywhere in this encoder.
        image, valid = packed[:, :1], packed[:, 3:4]
        fine = self.fine(image)
        coarse = self.context(F.avg_pool3d(image, (1, 3, 3), stride=(1, 3, 3), ceil_mode=True))
        coarse = F.interpolate(coarse, size=fine.shape[2:], mode='trilinear', align_corners=False)
        whole = (coarse*valid).sum((2, 3, 4), keepdim=True)/valid.sum((2, 3, 4), keepdim=True).clamp_min(1.)
        return self.combine(torch.cat([fine, coarse, whole.expand_as(coarse)], 1))


class MRIOnlyViewSwin(DualViewSwin):
    def __init__(self, device, arm):
        super().__init__(device, MRI_ARMS[arm])
        self.encoder = MRIOnlyEncoder(self.encoder)
        if arm.startswith('native_mri_'):
            self.settings = dict(self.settings, view='native')


def native_chart_inputs(image, memberships):
    """Place each existing chart's MRI samples on its original voxel lattice.

    Keep identical chart membership and valid-pooling partitions across all view
    controls. Crop each native chart, add one empty voxel of margin, then pad to
    a multiple of eight (minimum eight), as required by the existing encoders.
    """
    image = np.asarray(image, dtype=np.float32)
    shape = image.shape
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError('Expected finite native 3D MRI')
    pack = dict(charts=np.asarray(len(memberships)))
    seen = []
    for i, raw in enumerate(memberships):
        indices = np.asarray(raw, dtype=np.int64)
        if not len(indices) or indices.min() < 0 or indices.max() >= image.size:
            raise ValueError('Invalid native chart membership')
        points = np.stack(np.unravel_index(indices, shape), axis=1)
        low = points.min(0)-1
        size = np.maximum(8, ((points.max(0)-low+3+7)//8)*8)
        local = points-low
        packed = np.zeros((4, *size), dtype=np.float32)
        # Match the stored precision of both alternative-view atlas inputs.
        packed[(0, *local.T)] = image.ravel()[indices].astype(np.float16).astype(np.float32)
        packed[(3, *local.T)] = 1.
        grid = (2*local/(size-1)-1)[:, ::-1].copy().astype(np.float32)
        prefix = f'native_{i}'
        pack[prefix+'_input'] = packed
        pack[prefix+'_grid'] = grid
        pack[prefix+'_indices'] = indices.astype(np.int32)
        seen.append(indices)
    all_indices = np.concatenate(seen)
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError('Duplicated native chart voxels')
    pack['roi_indices'] = all_indices
    return pack


class NativeViewAtlas(FrozenAtlas):
    """Use only frozen source membership; no UV coordinates or resampled MRI."""
    def get(self, item):
        path = self.directory/(item['case_name']+'.npz')
        if path.name not in self.verified:
            if digest(path) != self.manifest['artifacts'][path.name]:
                raise ValueError('Frozen atlas changed: '+str(path))
            self.verified.add(path.name)
        with np.load(path, allow_pickle=False) as archive:
            memberships = [archive[f'plane_{i}_indices'] for i in range(int(archive['charts']))]
            expected = archive['roi_indices']
        pack = native_chart_inputs(item['image'][0].numpy(), memberships)
        if not np.array_equal(np.sort(pack['roi_indices']), np.sort(expected)):
            raise ValueError('Native-view control changed source ROI membership')
        return pack
