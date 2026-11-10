"""SwinUNETR with a shared 3D chart encoder and native-space feature fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from scripts.depth_supervision.train import model_factory


ARMS = {
    'dice': {}, 'depth': {},
    'plane_local': dict(view='plane', context=False, position=False, attention=False),
    'unfold_local': dict(view='unfold', context=False, position=False, attention=False),
    'plane_context': dict(view='plane', context=True, position=False, attention=False),
    'unfold_context': dict(view='unfold', context=True, position=False, attention=False),
    'plane_position': dict(view='plane', context=True, position=True, attention=False),
    'unfold_position': dict(view='unfold', context=True, position=True, attention=False),
    'plane_attention': dict(view='plane', context=True, position=True, attention=True),
    'unfold_attention': dict(view='unfold', context=True, position=True, attention=True),
}


def block(in_channels, out_channels, kernel=(3, 3, 3)):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel, padding=tuple(k // 2 for k in kernel)),
                         nn.GroupNorm(4, out_channels), nn.GELU())


class ChartEncoder(nn.Module):
    def __init__(self, context):
        super().__init__()
        self.fine = nn.Sequential(block(4, 8, (1, 3, 3)), block(8, 8))
        self.context = nn.Sequential(block(4, 8), block(8, 8)) if context else None
        self.combine = block(24 if context else 8, 16, (1, 1, 1))

    def forward(self, image):
        fine = self.fine(image)
        if self.context is not None:
            coarse = self.context(F.avg_pool3d(image, (1, 3, 3), stride=(1, 3, 3), ceil_mode=True))
            coarse = F.interpolate(coarse, size=fine.shape[2:], mode='trilinear', align_corners=False)
            valid = image[:, -1:]
            whole = (coarse * valid).sum((2, 3, 4), keepdim=True) / valid.sum((2, 3, 4), keepdim=True).clamp_min(1.)
            fine = torch.cat([fine, coarse, whole.expand_as(coarse)], 1)
        return self.combine(fine)


class DualViewSwin(nn.Module):
    def __init__(self, device, arm):
        super().__init__()
        self.native = model_factory(device)
        self.settings = ARMS[arm]
        self.encoder = ChartEncoder(self.settings['context']) if self.settings else None
        if self.settings:
            self.norm = nn.GroupNorm(4, 16)
            self.fusion = nn.Sequential(block(24 + 16 + 1, 24, (1, 1, 1)), block(24, 24))
            if self.settings['attention']:
                self.channel_gate = nn.Sequential(nn.Conv3d(40, 8, 1), nn.GELU(), nn.Conv3d(8, 40, 1), nn.Sigmoid())
                self.spatial_gate = nn.Sequential(nn.Conv3d(2, 1, 3, padding=1), nn.Sigmoid())
        self.to(device)

    def native_features(self, x):
        n = self.native
        hidden = n.swinViT(x, n.normalize)
        enc0, enc1 = n.encoder1(x), n.encoder2(hidden[0])
        enc2, enc3 = n.encoder3(hidden[1]), n.encoder4(hidden[2])
        dec3 = n.decoder5(n.encoder10(hidden[4]), hidden[3])
        dec2 = n.decoder4(dec3, enc3)
        dec1 = n.decoder3(dec2, enc2)
        dec0 = n.decoder2(dec1, enc1)
        return n.decoder1(dec0, enc0)

    def forward(self, x, atlas=None, *, suppress_aux=False):
        if not self.settings:
            return self.native(x)
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
            # Sparse pull to source voxel centres avoids mixing the two cut sides.
            pulled = F.grid_sample(encoded.float(), grid, align_corners=True, padding_mode='zeros')[:, :, :, 0, 0]
            indices = torch.as_tensor(atlas[prefix + '_indices'].astype('int64'), device=x.device)
            features = features.index_copy(2, indices, pulled)
            valid[:, :, indices] = 1
        mapped = self.norm(features.reshape(1, 16, *native.shape[2:]))
        mask = valid.reshape(1, 1, *native.shape[2:])
        mapped = mapped * mask
        if suppress_aux:
            mapped = mapped * 0
        both = torch.cat([native.float(), mapped], 1)
        if self.settings['attention']:
            weighted = both * self.channel_gate(both.mean((2, 3, 4), keepdim=True))
            descriptor = torch.cat([weighted.mean(1, keepdim=True), weighted.amax(1, keepdim=True)], 1)
            both = both + weighted * self.spatial_gate(descriptor)
        fused = self.fusion(torch.cat([both, mask], 1))
        return self.native.out(native + fused)
