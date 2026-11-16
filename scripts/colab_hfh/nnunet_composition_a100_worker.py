"""Same long recipes on A100, with a fresh, shared native update-60 anchor."""
from functools import partial
from pathlib import Path
import time

from scripts.colab_hfh import nnunet_seed0_long_worker as native
from scripts.colab_hfh import nnunet_long_constraints_worker as constrained
from scripts.colab_hfh import nnunet_composition_calibration as composition
from scripts.colab_hfh.runtime import read, write


def native_anchor(protocol):
    path = Path(protocol['a100_native_anchor'])
    deadline = time.monotonic()+180
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Matched native anchor did not arrive; no constraint updates permitted')
        time.sleep(.25)
    anchor = read(path)
    if anchor['after_update'] != 60 or len(anchor['model_state_sha256']) != 64:
        raise ValueError('Invalid A100 native anchor')
    return anchor


class NativeAnchorLoss(native.LongLoss):
    def finish(self, image, label, device, amp):
        super().finish(image,label,device,amp)
        if self.rows[-1]['update'] == 60:
            from scripts.cross_dataset_campaign.capped_core import model_digest
            write(self.output/'native_update60.json',dict(after_update=60,model_state_sha256=model_digest(self.model)))


def install(trainer, protocol, arm):
    original = constrained.modified_run
    def with_live_anchor(source):
        result = original(source)
        old = "            expected = protocol['matched_readiness_anchor']"
        if result.count(old) != 1:
            raise ValueError('Native anchor integration changed')
        return result.replace(old,"            from scripts.colab_hfh.nnunet_composition_a100_worker import native_anchor\n            expected = native_anchor(protocol)")
    constrained.modified_run = with_live_anchor
    try:
        constrained.install(trainer,protocol,arm)
    finally:
        constrained.modified_run = original
    if 'composition' in protocol:
        anchor = native_anchor(protocol)
        if protocol['matched_readiness_anchor'] != anchor:
            raise ValueError('Composition protocol does not match fresh native anchor')
        composition.ANCHOR = anchor
        specification = protocol['composition']
        trainer.select_weights = partial(composition.select_fixed_weights,specification=specification)
        original_calibrate = trainer.calibrate
        def calibrate(*args, **kwargs):
            report = original_calibrate(*args,**kwargs)
            if report['anchor_model_sha256'] == anchor['model_state_sha256']:
                composition.check_anchor(report,specification)
                report['composition_anchor_verified'] = True
            report.update(composition_variant=specification['variant'],coefficient_rule=specification['rule'])
            return report
        trainer.calibrate = calibrate


def source(is_native):
    if is_native:
        value = native.source()
        old = 'from scripts.colab_hfh.nnunet_seed0_long_worker import LongLoss as ReadyLoss'
        new = 'from scripts.colab_hfh.nnunet_composition_a100_worker import NativeAnchorLoss as ReadyLoss'
    else:
        value = constrained.source()
        old = 'from scripts.colab_hfh.nnunet_long_constraints_worker import install'
        new = 'from scripts.colab_hfh.nnunet_composition_a100_worker import install'
    if value.count(old) != 1:
        raise ValueError('Worker binding changed')
    return value.replace(old,new)


if __name__ == '__main__':
    import sys
    import torch
    # Three workers fit comfortably within the 40 GiB A100; reserve headroom.
    torch.cuda.set_per_process_memory_fraction(.28)
    protocol = Path(sys.argv[sys.argv.index('--protocol-dir')+1])
    is_native = read(protocol/'protocol.json')['active_objective'] == 'dice'
    exec(compile(source(is_native),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
