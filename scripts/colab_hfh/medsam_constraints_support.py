"""Use the exact archived control samples and learning rates for every arm."""
from scripts.cross_dataset_campaign.foundation import SupportSlabs


class PairedSlabs(SupportSlabs):
    def __init__(self, volumes, protocol, processor):
        from unittest.mock import patch
        from scripts.depth_supervision import geometry
        from scripts.colab_hfh.medsam_constraints_geometry import TENSOR_SPACING_MM, depth_geometry
        sdf, depth = geometry.signed_distance, geometry.build_geometry
        with patch.object(geometry, 'signed_distance',
                          lambda mask, spacing, clip: sdf(mask, TENSOR_SPACING_MM, clip)), \
             patch.object(geometry, 'build_geometry',
                          lambda labels, affine, config, **kw: depth_geometry(depth, labels, affine, config, **kw)):
            super().__init__(volumes, protocol, processor)

    def schedule(self, protocol):
        order = protocol['frozen_sample_schedule']
        if len(order) != protocol['updates'] * protocol['accumulation']:
            raise ValueError('Paired sampling length changed')
        if any(case not in self.indices or not 0 <= z < len(self.indices[case]) for case, z in order):
            raise ValueError('Paired sampling includes an invalid training slice')
        return order


def bind_learning_rates(original, protocol):
    import math
    rates = protocol['frozen_learning_rates']
    if len(rates) != 200 or not all(math.isfinite(x) and x > 0 for x in rates):
        raise ValueError('Invalid paired learning rates')
    original.lr_for_update = lambda update, *args: rates[update]
