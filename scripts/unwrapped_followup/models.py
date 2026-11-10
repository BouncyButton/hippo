"""Parameter-matched residual gates using the native path's uncertainty."""
import math
import torch
from torch.nn import functional as F
from scripts.unwrapped_swin.models import DualViewSwin


def uncertainty_gate(logits, mode):
    """Detached normalized entropy, or its spatial mean, in [0, 1]."""
    if mode not in ('entropy', 'mean_entropy'):
        raise ValueError('Unknown uncertainty gate: ' + str(mode))
    with torch.no_grad():
        logp = F.log_softmax(logits.detach().float(), dim=1)
        value = -(logp.exp() * logp).sum(1, keepdim=True) / math.log(logits.shape[1])
        value = value.clamp(0., 1.)
        if mode == 'mean_entropy':
            value = value.mean((2, 3, 4), keepdim=True).expand_as(value)
    return value


class FollowupSwin(DualViewSwin):
    """Keep the previous architecture; gate only its final feature residual."""
    def __init__(self, device, arm, gate_mode='none'):
        if gate_mode not in ('none', 'entropy', 'mean_entropy'):
            raise ValueError('Unknown gate mode')
        super().__init__(device, arm)
        if gate_mode != 'none' and not self.settings:
            raise ValueError('Uncertainty fusion requires an auxiliary branch')
        self.gate_mode = gate_mode

    def forward(self, x, atlas=None, *, suppress_aux=False, native_only=False, unit_gate=False):
        if native_only:
            return self.native(x)
        if self.gate_mode == 'none':
            return super().forward(x, atlas, suppress_aux=suppress_aux)
        native = self.native_features(x)
        view = self.settings['view']
        features = torch.zeros((1, 16, x[0, 0].numel()), device=x.device, dtype=torch.float32)
        valid = torch.zeros((1, 1, x[0, 0].numel()), device=x.device, dtype=torch.float32)
        for i in range(int(atlas['charts'])):
            prefix = f'{view}_{i}'
            image = torch.as_tensor(atlas[prefix + '_input'], device=x.device, dtype=x.dtype)[None].clone()
            if not self.settings['position']:
                image[:, 1:3] = 0
            encoded = self.encoder(image)
            grid = torch.as_tensor(atlas[prefix + '_grid'], device=x.device, dtype=torch.float32)[None, :, None, None]
            pulled = F.grid_sample(encoded.float(), grid, align_corners=True, padding_mode='zeros')[:, :, :, 0, 0]
            indices = torch.as_tensor(atlas[prefix + '_indices'].astype('int64'), device=x.device)
            features = features.index_copy(2, indices, pulled)
            valid[:, :, indices] = 1
        mask = valid.reshape(1, 1, *native.shape[2:])
        mapped = self.norm(features.reshape(1, 16, *native.shape[2:])) * mask
        if suppress_aux:
            mapped = mapped * 0
        both = torch.cat([native.float(), mapped], 1)
        if self.settings['attention']:
            weighted = both * self.channel_gate(both.mean((2, 3, 4), keepdim=True))
            descriptor = torch.cat([weighted.mean(1, keepdim=True), weighted.amax(1, keepdim=True)], 1)
            both = both + weighted * self.spatial_gate(descriptor)
        residual = self.fusion(torch.cat([both, mask], 1))
        gate = 1. if unit_gate else uncertainty_gate(self.native.out(native), self.gate_mode)
        return self.native.out(native + gate * residual)
