"""Preserve the native optimizer; instrument and guard all included losses."""
import inspect
import json
from pathlib import Path
import numpy as np

from scripts.colab_hfh import nnunet_composition_a100_worker as a100
from scripts.colab_hfh.nnunet_search13_policy import active_coefficients, probe_allowance
from scripts.colab_hfh.constraint_calibration import components


def terms(logits, labels, arm, sdf=None, depth=None):
    from scripts.adni_constraints import objectives as binary
    available = dict(bands=lambda:binary.bands()(logits,labels).loss,
                     edge=lambda:binary.BinaryEdge()(logits,labels),
                     sdf=lambda:binary.sdf_loss(logits,labels,sdf),
                     depth=lambda:binary.depth_loss(logits,labels,depth))
    return {name:available[name]() for name in components(arm)}


def measure(outputs, labels, arm, sdf, criterion, depth):
    import torch
    from scripts.nnunet_band_objective import native_loss
    with torch.autocast(outputs[0].device.type,enabled=False):
        detached = [x.detach().float() for x in outputs]
        z = detached[0].requires_grad_(True)
        losses = terms(z,labels,arm,sdf,depth)
        gradients = [torch.autograd.grad(native_loss(detached,labels,criterion),z)[0]]
        gradients += [torch.autograd.grad(value,z)[0] for value in losses.values()]
        gram = [[float((a*b).mean()) for b in gradients] for a in gradients]
    return gram,{n:float(v.detach()) for n,v in losses.items()}


class ActiveLoss:
    def __init__(self,original,protocol,output):
        self.original,self.protocol,self.output = original,protocol,Path(output)

    def __call__(self,outputs,labels,arm,weights,scale,target_sdf=None,
                 criterion=None,depth_weights=None,sdf_scale=0.):
        if arm == 'dice':
            return self.original(outputs,labels,arm,weights,scale,target_sdf,criterion,depth_weights,sdf_scale)
        if arm != self.protocol['active_objective'] or criterion is None:
            raise ValueError('A declared component was removed or native criterion changed')
        if tuple(outputs[0].shape) != (1,2,64,64,64):
            raise ValueError('Activity cost requires the declared grid')
        gram,losses = measure(outputs,labels,arm,target_sdf,criterion,depth_weights)
        effective,stats = active_coefficients(gram,weights,components(arm),self.protocol['activity_policy'])
        ledger = json.loads((self.output/'budget_ledger.json').read_text())
        row = dict(charged_attempt=ledger['charged_attempts'],arm=arm,scale=scale,
                   anchor_weights=weights,effective_weights=effective,gradient_gram=gram,
                   raw_losses=losses,probe_flops=probe_allowance(arm),**stats)
        with (self.output/'constraint_activity.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        return self.original(outputs,labels,arm,effective,scale,target_sdf,criterion,depth_weights,sdf_scale)


def adherence(model,train,validation,device,amp,protocol,output,schedule):
    """Post-training raw canonical losses, including the same native comparator."""
    import torch
    from scripts.colab_hfh.constraint_geometry import sdf_targets,depth_weights,DEPTH
    from scripts.cross_dataset_campaign.capped_core import model_digest
    from scripts.colab_hfh.runtime import write
    before = model_digest(model)
    mode = model.training
    records = []
    devices = [torch.cuda.current_device()] if amp else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            for split,patients in [('training',train),('validation',validation)]:
                for patient in patients:
                    image,label = patient['image'][None].to(device),patient['label'][None].to(device)
                    sdf = sdf_targets(label)
                    depth = depth_weights(label,protocol['native_geometry'][patient['case_name']]['model_affine'],DEPTH)
                    with torch.no_grad(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                        raw = model(image)
                    z = raw[0] if isinstance(raw,(list,tuple)) else raw
                    with torch.no_grad(),torch.autocast(device.type,enabled=False):
                        losses = {n:float(v) for n,v in terms(z.float(),label,'dice_bands_edge_sdf_depth',sdf,depth).items()}
                    if not all(np.isfinite(v) for v in losses.values()):
                        raise ValueError('Nonfinite adherence evaluation')
                    records.append(dict(split=split,case_name=patient['case_name'],losses=losses))
    finally:
        model.train(mode)
    if model_digest(model) != before:
        raise ValueError('Adherence evaluation changed model')
    write(output/'constraint_adherence.json',dict(status='complete',records=records,
        model_sha256=before,optimization_updates=0,
        diagnostic_flops_allowance=20*(schedule['baseline_flops_per_update']+probe_allowance('dice_bands_edge_sdf_depth')),
        interpretation='Lower raw canonical penalty means better surrogate adherence; padded training grid; no claim of hard anatomical feasibility; final endpoint only'))


def audit_source(value):
    old = '    base.atomic_json(args.output / "training_final.json", training_evaluation)'
    if value.count(old) != 1:
        raise ValueError('Final audit integration changed')
    return value.replace(old,old+'\n    from scripts.colab_hfh.nnunet_search13_worker import adherence\n    adherence(model,train,validation,device,amp,protocol,args.output,schedule)\n')


def installing(trainer,protocol,arm,original_install):
    original_get = inspect.getsource
    target = trainer.run
    def source_of(obj):
        value = original_get(obj)
        return audit_source(value) if obj is target else value
    inspect.getsource = source_of
    try:
        original_install(trainer,protocol,arm)
    finally:
        inspect.getsource = original_get


def install(trainer,protocol,arm):
    installing(trainer,protocol,arm,a100.install)


def install_native(trainer,protocol,arm):
    installing(trainer,protocol,arm,a100.native.install)


def source(is_native):
    value = a100.source(is_native)
    old = ('from scripts.colab_hfh.nnunet_seed0_long_worker import install' if is_native else
           'from scripts.colab_hfh.nnunet_composition_a100_worker import install')
    new = ('from scripts.colab_hfh.nnunet_search13_worker import install_native as install' if is_native else
           'from scripts.colab_hfh.nnunet_search13_worker import install')
    if value.count(old) != 1:
        raise ValueError('Worker source binding changed')
    value = value.replace(old,new)
    # CPU oversubscription would slow the 12-worker A100 search. Native
    # initialization and update60 must still match the archived control exactly.
    old_threads = "4, True"
    if value.count(old_threads) != 1:
        raise ValueError('Worker thread argument changed')
    value = value.replace(old_threads,"1, True")
    if not is_native:
        old = '    builder = exp.base.build_model'
        if value.count(old) != 1:
            raise ValueError('Loss binding changed')
        value = value.replace(old,'    from scripts.colab_hfh.nnunet_search13_worker import ActiveLoss\n    exp.trainer.training_loss = ActiveLoss(exp.trainer.training_loss,p,args.output)\n'+old)
    return value
