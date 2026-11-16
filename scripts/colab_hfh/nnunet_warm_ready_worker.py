"""Isolated worker: unchanged native optimizer/loss, new LR and readiness gate."""
import hashlib
import inspect
import json
from pathlib import Path

from scripts.colab_hfh.nnunet_warm_ready import readiness, READY_POLICY


class ReadyLoss:
    def __init__(self, original, output, protocol):
        from scripts.colab_hfh.residual_band_v2 import TrainingProbe
        self.original, self.output, self.protocol = original, output, protocol
        self.probe = TrainingProbe(original,output,protocol)
        self.path = output/'readiness_trace.jsonl'
        self.rows = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        self.ready = any(x['ready_after_step'] for x in self.rows)
        self.names = {}; self.pending = None

    @property
    def model(self):return self.probe.model

    @model.setter
    def model(self,model):self.probe.model=model

    def bind_cases(self,train):
        for case in train:
            key=hashlib.sha256(case['label'].detach().cpu().numpy().astype('uint8').tobytes()).hexdigest()
            if key in self.names:raise ValueError('Training annotations not uniquely identifiable')
            self.names[key]=case['case_name']

    def __call__(self, outputs, labels, arm, weights, scale, *args, **kwargs):
        import torch
        attempt=json.loads((self.output/'budget_ledger.json').read_text())['charged_attempts']
        if self.pending is not None or (self.rows and attempt != self.rows[-1]['update']+1):
            raise ValueError('Readiness trace cannot replay or skip charged work')
        z=outputs[0] if isinstance(outputs,(list,tuple)) else outputs
        key=hashlib.sha256(labels.detach().cpu().numpy().astype('uint8').tobytes()).hexdigest()
        if key not in self.names:raise ValueError('Readiness received a non-training crop')
        with torch.no_grad(),torch.autocast(z.device.type,enabled=False):
            prob=z.detach().float().softmax(1)[0,1,:,16:48,:]
            truth=labels[0,0,:,16:48,:].bool(); pred=prob>.5
            tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
            row=dict(update=attempt,case_name=self.names[key],annotation_sha256=key,
                hard_dice=2*tp/max(1,2*tp+fp+fn),
                soft_dice=float(2*(prob*truth).sum()/(prob.sum()+truth.sum()).clamp_min(1e-12)),
                mean_foreground_probability=float(prob[truth].mean()),
                mean_background_probability=float(prob[~truth].mean()),
                predicted_foreground=int(pred.sum()),truth_foreground=int(truth.sum()),fp=fp,fn=fn,
                ready_before_step=self.ready, band_call=arm=='dice_bands')
        self.pending=row
        if arm=='dice_bands':
            if not self.ready:raise ValueError('BAND enabled before training-panel readiness')
            return self.probe(outputs,labels,arm,weights,scale,*args,**kwargs)
        return self.original(outputs,labels,arm,weights,scale,*args,**kwargs)

    def finish(self,image,label,device,amp):
        self.probe.finish(image,label,device,amp)
        row=self.pending
        if row is None:raise ValueError('Missing training readiness observation')
        if row['update']%10==0:
            panel=(self.rows+[row])[-10:]
            row['panel_mean_dice']=sum(x['hard_dice'] for x in panel)/len(panel)
            row['panel_ready_cases']=sum(x['hard_dice']>=.5 for x in panel)
            self.ready=self.ready or readiness(panel)
        row['ready_after_step']=self.ready
        with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.rows.append(row);self.pending=None


def install(trainer,p,arm):
    from functools import partial
    from scripts.colab_hfh.residual_band_v2_worker import calibration_case
    from scripts.colab_hfh.constraint_calibration import select_joint_weights
    from scripts.colab_hfh import residual_band_v2
    if p['readiness_policy']!=READY_POLICY:raise ValueError('Readiness policy changed')
    for name,sha in p['warm_ready_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Frozen source differs: '+name)
    residual_band_v2.PROBE_UPDATES=(60,75,89)
    trainer.calibration_case=calibration_case
    trainer.select_weights=partial(select_joint_weights,targets={'bands':.1},component_cap=.5,joint_target=.1,joint_cap=.5)
    original=trainer.calibrate
    def calibrate(*a,**kw):
        report=original(*a,**kw)
        report.update(method=p['constraint_calibration'],scope='Ten training crops at first completed-panel readiness; no validation input')
        return report
    trainer.calibrate=calibrate
    source=inspect.getsource(trainer.run)
    replacements=[
        ('    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0',
         '    if not ledger_path.exists():\n        base.atomic_json(ledger_path, ledger)\n    training_loss.bind_cases(train)\n    step, best_score, best_step, epoch, cursor = 0, -1., 0, 0, 0'),
        ('charged >= schedule["prefix_updates"] and not calibration_path.exists()',
         'training_loss.ready and not calibration_path.exists()'),
        ('active_arm = "dice" if charged < schedule["prefix_updates"] else args.objective',
         'active_arm = "dice" if args.objective == "dice" or not calibration_path.exists() else args.objective'),
        ('        optimizer.step()', '        optimizer.step()\n        training_loss.finish(image,label,device,amp)'),
        ('        latest_validation = result','        latest_validation = result\n        if step == 89:\n            base.atomic_json(args.output / "validation_update89.json", result)'),
        ('    milestones = base.validation_steps(max_steps, protocol["validation_milestones"])',
         '    milestones = sorted(set(base.validation_steps(max_steps, protocol["validation_milestones"])) | {min(89,max_steps)})'),
        ('args.objective != "dice" and not calibration_path.exists()',
         'args.objective != "dice" and training_loss.ready and not calibration_path.exists()'),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Readiness integration changed: '+old)
        source=source.replace(old,new)
    end='    base.atomic_json(args.output / "training_final.json", training_evaluation)'
    source=source.replace(end,end+'''
    from scripts.colab_hfh.residual_band_v2_worker import final_spatial_audit
    final_spatial_audit(model,train,validation,device,amp,args.output)
    if not calibration_path.exists():
        base.atomic_json(calibration_path, dict(status='not_activated', weights={}, config_sha256=base.digest(config_path)))
''')
    exec(compile(source,trainer.__file__,'exec'),trainer.__dict__)


def worker_source():
    from scripts.colab_hfh.constraint_worker import worker_source as original
    source=original()
    replacements=[
        ('from scripts.colab_hfh.constraint_worker import install','from scripts.colab_hfh.nnunet_warm_ready_worker import install'),
        ("    exp.trainer.training_loss = partial(binary.training_loss, policy=p['sdf_policy'][p['active_objective']])",
         "    exp.trainer.training_loss = partial(binary.training_loss, policy=p['sdf_policy'][p['active_objective']])\n    from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss\n    exp.trainer.training_loss = ReadyLoss(exp.trainer.training_loss,args.output,p)"),
        ('        return model\n    exp.base.build_model = build','        exp.trainer.training_loss.model=model\n        return model\n    exp.base.build_model = build')]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Worker anchor changed: '+old)
        source=source.replace(old,new)
    return source


if __name__=='__main__':exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
