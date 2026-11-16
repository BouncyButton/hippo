"""MONAI model definitions for the 3D U-Net and UNETR experiments."""
from monai.networks.nets import BasicUNet, UNETR


CANDIDATES = {'basic_unet16': {'class': 'monai.networks.nets.BasicUNet',
                  'kwargs': {'spatial_dims': 3,
                             'in_channels': 1,
                             'out_channels': 3,
                             'features': (16, 16, 32, 64, 128, 16),
                             'upsample': 'deconv'},
                  'source': 'https://arxiv.org/abs/1606.06650',
                  'rationale': 'Plain 3D U-Net with sixteen initial channels, pooling and '
                               'concatenated skips; no residual units, attention or deep '
                               'supervision. Same BasicUNet topology as the width32 pilot, reduced '
                               'channel capacity.'},
 'unetr_vit_b16': {'class': 'monai.networks.nets.UNETR',
                   'kwargs': {'in_channels': 1,
                              'out_channels': 3,
                              'img_size': (64, 64, 64),
                              'feature_size': 16,
                              'hidden_size': 768,
                              'mlp_dim': 3072,
                              'num_heads': 12,
                              'proj_type': 'perceptron',
                              'norm_name': 'instance',
                              'conv_block': True,
                              'res_block': True,
                              'dropout_rate': 0.0,
                              'spatial_dims': 3,
                              'qkv_bias': False},
                   'source': 'https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV',
                   'rationale': 'Published feature16/ViT-B16 UNETR preset in MONAI 1.6.0, adapted '
                                'to 64 cubed inputs and three labels; twelve global-attention '
                                'layers, no pretrained weights.'}}


def build_candidate(name):
    constructors = {"basic_unet16": BasicUNet, "unetr_vit_b16": UNETR}
    return constructors[name](**CANDIDATES[name]["kwargs"])
