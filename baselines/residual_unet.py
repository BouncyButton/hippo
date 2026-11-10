"""Width-48 residual U-Net with a linear final class head.

Instance normalization and PReLU remain in every hidden feature block. They
are omitted only after the final three-channel upsampling convolution, before
the final linear residual unit. The small foregrounds otherwise exhibited
long false-positive plateaus in the fixed-compute hippocampus pilots.
"""
import torch
from monai.networks.nets import UNet

MODEL_KWARGS = dict(spatial_dims=3, in_channels=1, out_channels=3,
                    channels=[48, 96, 192, 384, 768], strides=[2, 2, 2, 2], num_res_units=2)
PARAMETERS = 43234201


def build_unet48() -> UNet:
    """Build the tested residual network; return raw, unnormalized class logits."""
    model = UNet(**MODEL_KWARGS)
    top = model.model[2][0]
    if (not isinstance(top.adn.N, torch.nn.InstanceNorm3d) or top.adn.N.affine
            or not isinstance(top.adn.A, torch.nn.PReLU) or top.conv.out_channels != 3):
        raise ValueError('MONAI output structure differs from the verified residual U-Net')
    top.adn = torch.nn.Identity()
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Residual U-Net parameter count changed')
    return model


def absolute_learning_rate(step: int, initial: float, warmup_updates: int = 100) -> float:
    """Absolute update warmup followed by constant LR; independent of budget."""
    if step < 0 or warmup_updates < 1:
        raise ValueError('Invalid update or warmup')
    return initial * min(1., (step + 1) / warmup_updates)
