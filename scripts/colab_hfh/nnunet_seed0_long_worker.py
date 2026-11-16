"""No-augmentation, doubled-cap matched nnU-Net worker."""
import hashlib
import inspect
import json
from pathlib import Path
from scripts.colab_hfh import nnunet_warm_ready_worker as ready
from scripts.colab_hfh.nnunet_seed0_balance_worker import HeadProbeLoss

SCHEDULE=dict(original_prefix_updates=20,plateau_lr=.07,decay_start=120,horizon=180,power=.9)


def learning_rate(step):
    if step < 20:return .1*(1-step/90)**.9
    if step < 120:return .07
    return .07*max(0.,1-(step-120)/60)**.9


class LongLoss(HeadProbeLoss):
    def finish(self,image,label,device,amp):
        before=self.ready
        super().finish(image,label,device,amp)
        if self.ready and not before:
            from scripts.cross_dataset_campaign.capped_core import model_digest
            (self.output/'readiness_model.json').write_text(json.dumps(dict(
                after_update=self.rows[-1]['update'],model_state_sha256=model_digest(self.model)))+'\n')


def installer_source():
    source=inspect.getsource(ready.install)
    source=source.replace('if step == 89:', 'if step == 179:').replace('validation_update89.json','validation_update179.json').replace('min(89,max_steps)','min(179,max_steps)')
    source=source.replace('base.atomic_json(args.output / "validation_update179.json", result)',
        'base.atomic_json(args.output / "validation_update179.json", result)\\n            base.atomic_json(args.output / "training_update179.json", base.evaluate(model,train,device,amp))')
    return source


def install(trainer,p,arm):
    if p['long_schedule']!=SCHEDULE or p['augmentation']!='none' or p['lr_horizon_updates']!=180:
        raise ValueError('Frozen longer schedule differs')
    for name,digest in p['long_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen longer source differs: '+name)
    ns=dict(ready.__dict__)
    exec(compile(installer_source(),__file__,'exec'),ns)
    ns['install'](trainer,p,arm)
    from scripts.colab_hfh import residual_band_recall_balance as balanced
    balanced.PROBE_UPDATES=(75,120,150,179)
    trainer.base.learning_rate=lambda name,step,maximum,initial:learning_rate(step)


def source():
    value=ready.worker_source()
    for old,new in {
        'from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss':
        'from scripts.colab_hfh.nnunet_seed0_long_worker import LongLoss as ReadyLoss',
        'from scripts.colab_hfh.nnunet_warm_ready_worker import install':
        'from scripts.colab_hfh.nnunet_seed0_long_worker import install',
    }.items():
        if value.count(old)!=1:raise ValueError('Long worker integration changed')
        value=value.replace(old,new)
    return value


if __name__=='__main__':exec(compile(source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
