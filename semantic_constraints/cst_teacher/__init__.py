"""Convolutional Set Transformer teachers for conditional segmentation constraints."""

from .descriptors import DESCRIPTOR_NAMES, sampled_slice_profiles, soft_descriptors
from .losses import (
    anomaly_constraint,
    conditional_descriptor_constraint,
    contextual_profile_constraint,
)
from .model import CSTDescriptorTeacher, CSTMaskAnomalyTeacher, SetConv2d

__all__ = [
    "CSTDescriptorTeacher",
    "CSTMaskAnomalyTeacher",
    "DESCRIPTOR_NAMES",
    "SetConv2d",
    "anomaly_constraint",
    "conditional_descriptor_constraint",
    "contextual_profile_constraint",
    "sampled_slice_profiles",
    "soft_descriptors",
]

