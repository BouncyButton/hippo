"""MNI geometry adaptation; preserve subfield targets and paired sampling."""
import numpy as np


def depth_geometry(builder, labels, affine, config, *, fields=False):
    if fields:
        raise ValueError('MNI adapter is only for class-independent depth weights')
    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError('Invalid MNI subfield labels')
    # With fields=False the reference algorithm uses foreground occupancy only.
    # Retain the original multiclass labels everywhere else, including depth CE.
    return builder((labels > 0).astype(np.uint8), affine, config, fields=False)


def install(protocol):
    from unittest.mock import patch
    from scripts.colab_mni_medsam3.train import SupportSlabs as MNI
    from scripts.depth_supervision import geometry
    from scripts.cross_dataset_campaign import foundation
    reference = geometry.build_geometry

    class PairedSupport(MNI):
        def __init__(self, volumes, current, processor):
            with patch.object(geometry, 'build_geometry',
                    lambda labels, affine, config, **kw: depth_geometry(reference, labels, affine, config, **kw)):
                super().__init__(volumes, current, processor)

        def schedule(self, current):
            order = current['frozen_sample_schedule']
            if len(order) != current['updates'] * current['accumulation']:
                raise ValueError('Paired sample schedule length differs')
            if any(c not in self.indices or not 0 <= z < len(self.indices[c]) for c, z in order):
                raise ValueError('Paired sample schedule includes an invalid support slice')
            return order

    foundation.SupportSlabs = PairedSupport
    rates = protocol['frozen_learning_rates']
    if len(rates) != protocol['updates'] or not all(np.isfinite(x) and x > 0 for x in rates):
        raise ValueError('Invalid matched learning-rate trajectory')
    foundation.lr_for_update = lambda update, *args: rates[update]
