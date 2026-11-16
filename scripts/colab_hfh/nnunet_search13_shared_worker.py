"""Matched 90-update native recipe with earlier constraint activation."""
import inspect
import json
import os
from pathlib import Path
import time

from scripts.colab_hfh.nnunet_search13_worker import ActiveLoss,adherence,audit_source,installing
from scripts.colab_hfh import nnunet_composition_a100_worker as a100
from scripts.colab_hfh import constraint_worker
from scripts.colab_hfh import nnunet_seed0_plateau_worker as plateau
from scripts.colab_hfh.nnunet_search13_shared_policy import SCHEDULE
from scripts.colab_hfh.runtime import read,write


class NativeAnchors(plateau.HeadProbeLoss):
    def finish(self,image,label,device,amp):
        super().finish(image,label,device,amp)
        step = self.rows[-1]['update']
        if step in (20,30,45,60):
            from scripts.cross_dataset_campaign.capped_core import model_digest
            write(self.output/f'native_update{step}.json',dict(after_update=step,model_state_sha256=model_digest(self.model)))


def native_anchor(protocol):
    path = Path(protocol['a100_native_anchor'])
    deadline = time.monotonic()+300
    while not path.exists():
        if time.monotonic()>deadline:raise RuntimeError('Matched native anchor unavailable')
        time.sleep(.25)
    anchor = read(path)
    if anchor['after_update'] != protocol['prefix_anchor_updates']:
        raise ValueError('Wrong native prefix anchor')
    return anchor


def install_native(trainer,protocol,arm):
    installing(trainer,protocol,arm,plateau.install)


def install(trainer,protocol,arm):
    if protocol['seed0_plateau_schedule'] != SCHEDULE or protocol['lr_horizon_updates'] != 90:
        raise ValueError('Shared-cap native LR recipe changed')
    original_get = inspect.getsource
    target = trainer.run
    def source_of(obj):
        value = original_get(obj)
        if obj is not target:return value
        value = audit_source(value)
        old = '            report.update(config_sha256=base.digest(config_path), anchor_successful_updates=step,'
        if value.count(old) != 1:raise ValueError('Calibration hook changed')
        value = value.replace(old,"            from scripts.colab_hfh.nnunet_search13_shared_worker import native_anchor\n            expected = native_anchor(protocol)\n            if step != expected['after_update'] or report['anchor_model_sha256'] != expected['model_state_sha256']:\n                raise ValueError('Shared-cap matched native prefix differs')\n"+old)
        return value
    inspect.getsource = source_of
    try:constraint_worker.install(trainer,protocol,arm)
    finally:inspect.getsource = original_get
    trainer.base.learning_rate = lambda name,step,maximum,initial:plateau.plateau_lr(step)


def source(is_native):
    if is_native:
        value = a100.source(True)
        replacements = {
            'from scripts.colab_hfh.nnunet_seed0_long_worker import install':
            'from scripts.colab_hfh.nnunet_search13_shared_worker import install_native as install',
            'from scripts.colab_hfh.nnunet_composition_a100_worker import NativeAnchorLoss as ReadyLoss':
            'from scripts.colab_hfh.nnunet_search13_shared_worker import NativeAnchors as ReadyLoss'}
    else:
        value = constraint_worker.worker_source()
        replacements = {'from scripts.colab_hfh.constraint_worker import install':
                        'from scripts.colab_hfh.nnunet_search13_shared_worker import install',
                        '    builder = exp.base.build_model':
                        '    from scripts.colab_hfh.nnunet_search13_worker import ActiveLoss\n    exp.trainer.training_loss = ActiveLoss(exp.trainer.training_loss,p,args.output)\n    builder = exp.base.build_model'}
    # Preserve the original CPU reduction/preprocessing setting. Changing this
    # to one thread changed the native update60 hash in the superseded pilot.
    for old,new in replacements.items():
        if value.count(old) != 1:raise ValueError('Shared worker integration changed: '+old)
        value = value.replace(old,new)
    return value


if __name__ == '__main__':
    import sys
    import torch
    if 'HFH_SEARCH_GPU_FRACTION' in os.environ:
        torch.cuda.set_per_process_memory_fraction(float(os.environ['HFH_SEARCH_GPU_FRACTION']))
    path = Path(sys.argv[sys.argv.index('--protocol-dir')+1])
    is_native = read(path/'protocol.json')['active_objective'] == 'dice'
    exec(compile(source(is_native),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
