"""GeoFWI winning loss/dropout and stateless paired voxel-plane symmetries."""
import hashlib
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR
import torch
from scripts.unetr_subfields.core import model_kwargs as base_kwargs

LOSS_ARGUMENTS = dict(include_background=False, to_onehot_y=True, softmax=True,
    squared_pred=True, smooth_nr=0., smooth_dr=1e-6, lambda_dice=1., lambda_ce=1.)
POLICY = 'xy_d4_v1'


def model_kwargs(classes, shape):
    return dict(base_kwargs(classes, shape), dropout_rate=.1)


def build_model(classes, shape):
    return UNETR(**model_kwargs(classes, shape))


def training_loss(outputs, labels, arm, weights, scale, target_sdf=None,
                  criterion=None, depth_weights=None, sdf_scale=0.):
    if arm not in ('dice', 'dice_ce') or weights or any(v is not None for v in (target_sdf, criterion, depth_weights)):
        raise ValueError('Only the unconstrained Geo Dice+CE baseline is allowed')
    return DiceCELoss(**LOSS_ARGUMENTS)(outputs.float(), labels)


def transform_id(seed, update):
    if seed < 0 or update < 0: raise ValueError('Negative seed/update')
    return hashlib.sha256(f'geofwi-xy-d4-v1:{seed}:{update}'.encode()).digest()[0] % 8


def apply_transform(image, label, index):
    if image.ndim != 5 or image.shape != label.shape or image.shape[2] != image.shape[3] or index not in range(8):
        raise ValueError('Expected aligned single-channel volumes with square first two voxel axes')
    def apply(value):
        value = torch.rot90(value, index % 4, dims=(2, 3))
        if index >= 4: value = torch.flip(value, dims=(2,))
        return value.contiguous()
    return apply(image), apply(label)


def augment(image, label, seed, update):
    return apply_transform(image, label, transform_id(seed, update))
