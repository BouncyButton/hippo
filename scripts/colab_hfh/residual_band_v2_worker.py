"""Isolated HFH residual-band worker; no changes to other running recipes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.colab_hfh.residual_band import ResidualBandLoss, policy_dict, geometry, selection


def calibration_case(outputs, labels, arm, target_sdf=None, criterion=None, depth_weights=None):
    import torch
    from scripts.nnunet_band_objective import native_loss
    if arm!='dice_bands' or criterion is None:
        raise ValueError('Residual BAND calibration requires native nnU-Net supervision')
    detached=[x.detach().float() for x in outputs]
    z=detached[0].requires_grad_(True)
    native=native_loss(detached,labels,criterion)
    envelope=ResidualBandLoss(envelope=True)(z,labels).loss
    gradients=[torch.autograd.grad(x,z)[0] for x in (native,envelope)]
    gram=[[float((a*b).mean()) for b in gradients] for a in gradients]
    if not np.isfinite(gram).all() or min(gram[0][0],gram[1][1])<=0:
        raise ValueError('Invalid native/envelope calibration gradient')
    return dict(gradient_names=['native','bands'],gradient_gram=gram,
        native_rms=gram[0][0]**.5,auxiliary='Fixed-normalization maximum-width BCE envelope')


def install(trainer,protocol,arm):
    import inspect
    from functools import partial
    from scripts.colab_hfh.constraint_calibration import select_joint_weights
    from scripts.adni_constraints import objectives as binary
    if (protocol.get('residual_band_policy')!=policy_dict()
            or protocol['model_identity']!='nnunet' or arm!='dice_bands'):
        raise ValueError('Conditional BAND policy or model differs')
    for name,digest in protocol['residual_band_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Conditional source changed: '+name)
    for name,digest in protocol['residual_band_v2_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('V2 source changed: '+name)
    binary.OuterBoundaryBandLoss=ResidualBandLoss
    binary.bind()
    trainer.calibration_case=calibration_case
    trainer.select_weights=partial(select_joint_weights,
        targets=protocol['constraint_component_targets'],component_cap=protocol['calibration_p95_cap'],
        joint_target=protocol['constraint_joint_target'],joint_cap=protocol['constraint_joint_cap'])
    original=trainer.calibrate
    def calibrate(*args,**kwargs):
        report=original(*args,**kwargs)
        report.update(method=protocol['constraint_calibration'],
            scope='Ten training crops only; maximum-width fixed-normalization envelope bounds gated residual gradients at the anchor; no parameter updates')
        return report
    trainer.calibrate=calibrate
    source=inspect.getsource(trainer.run)
    anchor='    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0'
    end='    base.atomic_json(args.output / "training_final.json", training_evaluation)'
    if source.count(anchor)!=1 or source.count(end)!=1:
        raise ValueError('Trainer integration anchor changed')
    source=source.replace(anchor,'    if not ledger_path.exists():\n        base.atomic_json(ledger_path, ledger)\n'+anchor)
    source=source.replace(end,end+'''
    from scripts.colab_hfh.residual_band_v2_worker import final_spatial_audit
    final_spatial_audit(model,train,validation,device,amp,args.output)
''')
    trainer.finish_band_step=lambda image,label,device,amp: trainer.training_loss.finish(image,label,device,amp)
    step_anchor='        optimizer.step()'
    if source.count(step_anchor)!=1:raise ValueError('Optimizer integration anchor changed')
    source=source.replace(step_anchor,step_anchor+'\n        finish_band_step(image,label,device,amp)')
    exec(compile(source,trainer.__file__,'exec'),trainer.__dict__)


from scripts.colab_hfh.residual_band_v2 import bind_trace


def final_spatial_audit(model,train,validation,device,amp,output):
    """Record final error support; held-out measurements cannot change the rule."""
    import torch
    from contextlib import nullcontext
    rows=[];mode=model.training
    model.eval()
    try:
        with torch.no_grad():
            for group,patients in (('train',train),('validation',validation)):
                for item in patients:
                    with torch.autocast('cuda',dtype=torch.bfloat16) if amp else nullcontext():
                        raw=model(item['image'][None].to(device))
                    z=raw[0] if isinstance(raw,(list,tuple)) else raw
                    values=item['label'].detach().cpu().numpy()[0].astype(np.uint8)
                    truth,roi,inner,outer=geometry(values.tobytes())
                    pred=z.detach().float().argmax(1)[0].cpu().numpy().astype(bool)
                    _,row=selection(truth,pred,roi,inner,outer)
                    row.update(split=group,case_name=item['case_name'],
                        annotation_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        interpretation='Post-training counterfactual support; never used for fitting')
                    rows.append(row)
    finally:model.train(mode)
    report=dict(optimization_updates=0,extra_forward_passes=len(rows),
        accounting='Separate post-training inference diagnostic, outside optimization cap',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=rows)
    (output/'residual_band_spatial_final.json').write_text(json.dumps(report,indent=2)+'\n')


def worker_source():
    from scripts.colab_hfh.constraint_worker import worker_source as original
    source=original()
    replacements=[
        ('from scripts.colab_hfh.constraint_worker import install',
         'from scripts.colab_hfh.residual_band_v2_worker import install'),
        ("    exp.trainer.training_loss = partial(binary.training_loss, policy=p['sdf_policy'][p['active_objective']])",
         "    exp.trainer.training_loss = partial(binary.training_loss, policy=p['sdf_policy'][p['active_objective']])\n    from scripts.colab_hfh.residual_band_v2_worker import bind_trace\n    exp.trainer.training_loss = bind_trace(exp.trainer.training_loss,args.output,p)"),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Worker integration anchor changed')
        source=source.replace(old,new)
    # Precompute label geometry before optimization. No predictions or held-out
    # measurements enter these cached physical-distance fields.
    anchor='    exp.base.PreparedPatients = PreparedPatients'
    injection='''    from scripts.colab_hfh.residual_band import geometry
    class GeometryPreparedPatients(PreparedPatients):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            for item in self.items:
                geometry(item['label'].detach().cpu().numpy()[0].astype('uint8').tobytes())
    exp.base.PreparedPatients = GeometryPreparedPatients
'''
    if source.count(anchor)!=1:raise ValueError('Data binding anchor changed')
    source=source.replace(anchor,injection)
    model_anchor='        return model\n    exp.base.build_model = build'
    if source.count(model_anchor)!=1:raise ValueError('Model binding anchor changed')
    return source.replace(model_anchor,'        exp.trainer.training_loss.model=model\n'+model_anchor)


if __name__=='__main__':
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
