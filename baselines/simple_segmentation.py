"""Predeclared classical CNN candidates for the fixed-compute baseline screen."""
from monai.networks.nets import BasicUNet, SegResNet, VNet, UNETR


CANDIDATES = {
    'unetr_vit_b16': {
        'class': 'monai.networks.nets.UNETR',
        'kwargs': {'in_channels': 1, 'out_channels': 3, 'img_size': (64, 64, 64),
                   'feature_size': 16, 'hidden_size': 768, 'mlp_dim': 3072, 'num_heads': 12,
                   'proj_type': 'perceptron', 'norm_name': 'instance', 'conv_block': True,
                   'res_block': True, 'dropout_rate': 0.0, 'spatial_dims': 3, 'qkv_bias': False},
        'source': 'https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV',
        'rationale': 'Published feature16/ViT-B16 UNETR preset in MONAI 1.6.0, adapted to 64 cubed inputs and three labels; twelve global-attention layers, no pretrained weights.',
    },
    'vnet': {
        'class': 'monai.networks.nets.VNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'act': 'prelu', 'dropout_prob_down': .5, 'dropout_prob_up': (.5, .5)},
        'source': 'https://arxiv.org/abs/1606.04797',
        'rationale': 'Classical fully convolutional volumetric segmentation with large kernels and residual blocks; PReLU as in the original architecture, MONAI dropout defaults retained.',
    },
    'basic_unet': {
        'class': 'monai.networks.nets.BasicUNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'features': (32, 32, 64, 128, 256, 32), 'upsample': 'deconv'},
        'source': 'https://arxiv.org/abs/1606.06650',
        'rationale': 'Plain convolutional U-Net with pooling and concatenated skips; MONAI default feature widths, no residual units, attention, or deep supervision. This is an implementation preset, not an exact reproduction of the original paper.',
    },
    'basic_unet16': {
        'class': 'monai.networks.nets.BasicUNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'features': (16, 16, 32, 64, 128, 16), 'upsample': 'deconv'},
        'source': 'https://arxiv.org/abs/1606.06650',
        'rationale': 'Plain 3D U-Net with sixteen initial channels, pooling and concatenated skips; no residual units, attention or deep supervision. Same BasicUNet topology as the width32 pilot, reduced channel capacity.',
    },
    'basic_unet8': {
        'class': 'monai.networks.nets.BasicUNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'features': (8, 8, 16, 32, 64, 8), 'upsample': 'deconv'},
        'source': 'https://arxiv.org/abs/1606.06650',
        'rationale': 'Plain 3D U-Net with eight initial channels; a capacity ablation of the same BasicUNet topology, retaining all pooling stages and skips. No validation performance has been assumed.',
    },
    'segresnet': {
        'class': 'monai.networks.nets.SegResNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'init_filters': 8, 'upsample_mode': 'deconv'},
        'source': 'https://arxiv.org/abs/1810.11654',
        'rationale': 'Compact convolutional residual encoder-decoder, default eight initial filters; trainable deconvolution upsampling; no autoencoder branch.',
    },
    'segresnet12': {
        'class': 'monai.networks.nets.SegResNet',
        'kwargs': {'spatial_dims': 3, 'in_channels': 1, 'out_channels': 3,
                   'init_filters': 12, 'norm': ('GROUP', {'num_groups': 4}),
                   'upsample_mode': 'deconv'},
        'source': 'https://arxiv.org/abs/1810.11654',
        'rationale': 'Segmentation-only residual encoder-decoder with twelve initial filters, four GroupNorm groups and trainable deconvolution; no autoencoder branch or deep supervision.',
    },
}


def build_candidate(name):
    constructors = {'vnet': VNet, 'basic_unet': BasicUNet,
                    'basic_unet16': BasicUNet, 'basic_unet8': BasicUNet, 'segresnet': SegResNet,
                    'segresnet12': SegResNet, 'unetr_vit_b16': UNETR}
    return constructors[name](**CANDIDATES[name]['kwargs'])
