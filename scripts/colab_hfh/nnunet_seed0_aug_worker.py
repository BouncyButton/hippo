"""Matched augmentation binding; native loss, LR and balanced BAND unchanged."""
import hashlib
import inspect
import json
from pathlib import Path

from scripts.colab_hfh.nnunet_seed0_balance_worker import HeadProbeLoss as OriginalLoss
from scripts.colab_hfh.nnunet_seed0_balance_worker import install as install_balance
from scripts.colab_hfh.nnunet_warm_ready_worker import worker_source
from scripts.colab_hfh.nnunet_augmentation import NAME,POLICY,augment


class AugmentedLoss(OriginalLoss):
    def bind_cases(self,train):
        super().bind_cases(train)
        self.allowed_cases={case['case_name'] for case in train}
        self.augmentation_record=None

    def bind_augmented_case(self,case_name,label,record):
        from scripts.colab_hfh.residual_band import geometry
        if case_name not in self.allowed_cases:raise ValueError('Non-training augmentation')
        annotation=label.detach().cpu().numpy()[0,0].astype('uint8').tobytes()
        key=hashlib.sha256(annotation).hexdigest()
        self.names[key]=case_name
        geometry(annotation)  # Recompute from transformed annotation, never stale shells.
        self.augmentation_record=dict(case_name=case_name,annotation_sha256=key,**record)

    def finish(self,image,label,device,amp):
        before=self.ready
        super().finish(image,label,device,amp)
        if self.augmentation_record is None:raise ValueError('Missing augmentation evidence')
        row=dict(update=self.rows[-1]['update'],**self.augmentation_record)
        with (self.output/'augmentation_trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if self.ready and not before:
            from scripts.cross_dataset_campaign.capped_core import model_digest
            (self.output/'readiness_model.json').write_text(json.dumps(dict(
                after_update=row['update'],model_state_sha256=model_digest(self.model)))+'\n')
        self.augmentation_record=None


def validate_policy(protocol):
    if protocol['augmentation']!=NAME or protocol['augmentation_policy']!=POLICY:
        raise ValueError('Frozen augmentation policy differs')
    for name,expected in protocol['augmentation_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Augmentation source differs: '+name)


def patched_run_source(source):
    old='    check_no_augmentation(protocol)'
    new='    from scripts.colab_hfh.nnunet_seed0_aug_worker import validate_policy\n    validate_policy(protocol)'
    if source.count(old)!=1:raise ValueError('Augmentation guard anchor differs')
    source=source.replace(old,new)
    old='        image, label = batch["image"][None], batch["label"][None]'
    new=old+'''
        from scripts.colab_hfh.nnunet_augmentation import augment
        image,label,augmentation_record=augment(image,label,augmentation_rng)
        training_loss.bind_augmented_case(batch['case_name'],label,augmentation_record)
'''
    if source.count(old)!=1:raise ValueError('Training-only augmentation anchor differs')
    source=source.replace(old,new)
    old='        latest_validation = result'
    new=old+'''
        if step == 89:
            base.atomic_json(args.output/'training_update89.json',base.evaluate(model,train,device,amp))
'''
    if source.count(old)!=1:raise ValueError('Equal-update evaluation anchor differs')
    return source.replace(old,new)


def install(trainer,protocol,arm):
    original=inspect.getsource
    target=trainer.run
    modified=patched_run_source(original(target))
    # Let the existing readiness installer compose its audited changes with
    # this source, in one compile. Do not mutate shared source files on disk.
    inspect.getsource=lambda obj:modified if obj is target else original(obj)
    try:install_balance(trainer,protocol,arm)
    finally:inspect.getsource=original


def source():
    value=worker_source()
    changes={
        'from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss':
            'from scripts.colab_hfh.nnunet_seed0_aug_worker import AugmentedLoss as ReadyLoss',
        'from scripts.colab_hfh.nnunet_warm_ready_worker import install':
            'from scripts.colab_hfh.nnunet_seed0_aug_worker import install',
        "augmentation='none'":"augmentation=p['augmentation']",
    }
    for old,new in changes.items():
        if value.count(old)!=1:raise ValueError('Augmented worker anchor differs: '+old)
        value=value.replace(old,new)
    return value


if __name__=='__main__':exec(compile(source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
