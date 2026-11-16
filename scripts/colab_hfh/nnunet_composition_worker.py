"""Isolated worker for the two frozen nnU-Net composition tests."""
from functools import partial

from scripts.colab_hfh import nnunet_long_constraints_worker as long_worker
from scripts.colab_hfh.nnunet_composition_calibration import select_fixed_weights, check_anchor, ANCHOR


def install(trainer, protocol, arm):
    long_worker.install(trainer, protocol, arm)
    spec = protocol['composition']
    trainer.select_weights = partial(select_fixed_weights, specification=spec)
    original = trainer.calibrate
    def calibrate(*args, **kwargs):
        report = original(*args, **kwargs)
        # Initial training calibration is additionally checked in the existing
        # update60 integration. Final read-only probes retain these fixed weights.
        if report['anchor_model_sha256'] == ANCHOR['model_state_sha256']:
            check_anchor(report, spec)
            report['composition_anchor_verified'] = True
        report.update(composition_variant=spec['variant'], coefficient_rule=spec['rule'])
        return report
    trainer.calibrate = calibrate


def source():
    value = long_worker.source()
    old = 'from scripts.colab_hfh.nnunet_long_constraints_worker import install'
    if value.count(old) != 1:
        raise ValueError('Long worker integration changed')
    return value.replace(old, 'from scripts.colab_hfh.nnunet_composition_worker import install')


if __name__ == '__main__':
    exec(compile(source(), __file__, 'exec'), {'__name__': '__main__', '__file__': __file__})
