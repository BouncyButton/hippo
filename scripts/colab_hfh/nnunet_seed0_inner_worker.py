"""Seed-zero LR diagnosis with detached, per-head foreground measurements."""
import json
from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss, worker_source, install as install_ready


class HeadProbeLoss(ReadyLoss):
    def __init__(self,original,output,protocol):
        super().__init__(original,output,protocol)
        from scripts.colab_hfh.residual_band_inner import TrainingProbe
        self.probe=TrainingProbe(original,output,protocol)

    def __call__(self, outputs, labels, *args, **kwargs):
        import torch
        import torch.nn.functional as F
        update = json.loads((self.output/'budget_ledger.json').read_text())['charged_attempts']
        if update % 10 == 0 or update == 89:
            heads=[]
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                for z in outputs:
                    z=z.detach().float()
                    truth=F.interpolate(labels.float(),size=z.shape[2:],mode='nearest')[:,0].bool()
                    prob=z.softmax(1)[:,1];pred=prob>.5
                    tp=(pred&truth).sum();fp=(pred&~truth).sum();fn=(~pred&truth).sum()
                    heads.append(dict(shape=list(z.shape),foreground_probability=float(prob[truth].mean()) if truth.any() else None,
                        hard_dice=float(2*tp/(2*tp+fp+fn).clamp_min(1)),
                        soft_dice=float(2*(prob*truth).sum()/(prob.sum()+truth.sum()).clamp_min(1)),
                        ce=float(F.cross_entropy(z,truth.long()))))
            with (self.output/'head_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=update,heads=heads))+'\n')
        return super().__call__(outputs,labels,*args,**kwargs)


def plateau_lr(step):
    if step < 20:
        return .1*(1-step/90)**.9
    if step < 60:
        return .07
    return .07*(1-(step-60)/30)**.9


def inner_calibration_case(outputs,labels,arm,target_sdf=None,criterion=None,depth_weights=None):
    import torch
    import torch.nn.functional as F
    from scripts.nnunet_band_objective import native_loss
    from scripts.colab_hfh.residual_band import geometry
    if arm!='dice_bands' or criterion is None:raise ValueError('Expected native supervised BAND calibration')
    detached=[x.detach().float() for x in outputs];z=detached[0].requires_grad_(True)
    truth,roi,inner,outer=geometry(labels.detach().cpu().numpy()[0,0].astype('uint8').tobytes())
    mask=torch.as_tensor(inner[-1].copy(),device=z.device)
    target=torch.as_tensor(truth.astype('float32'),device=z.device)
    native=native_loss(detached,labels,criterion)
    envelope=(F.binary_cross_entropy_with_logits(z[0,1]-z[0,0],target,reduction='none')*mask).sum()/int(truth.sum())
    gradients=[torch.autograd.grad(t,z)[0] for t in (native,envelope)]
    gram=[[float((a*b).mean()) for b in gradients] for a in gradients]
    if not all(__import__('math').isfinite(x) for row in gram for x in row) or min(gram[0][0],gram[1][1])<=0:raise ValueError('Invalid inner envelope calibration')
    return dict(gradient_names=['native','bands'],gradient_gram=gram,native_rms=gram[0][0]**.5,
        auxiliary='Inner maximum-width BCE envelope only; fixed foreground normalization; outer penalty zero')


def install(trainer, protocol, arm):
    import hashlib
    from pathlib import Path
    for name,expected_hash in protocol['recall_balance_source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('Recall-balance source differs: '+name)
    install_ready(trainer, protocol, arm)
    trainer.calibration_case=inner_calibration_case
    expected={'original_prefix_updates':20,'plateau_lr':.07,'decay_start':60,'horizon':90,'power':.9}
    if protocol['seed0_plateau_schedule']!=expected:raise ValueError('Plateau schedule differs')
    trainer.base.learning_rate=lambda name,step,maximum,initial:plateau_lr(step)


if __name__=='__main__':
    source=worker_source().replace('from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss',
        'from scripts.colab_hfh.nnunet_seed0_inner_worker import HeadProbeLoss as ReadyLoss')
    source=source.replace('from scripts.colab_hfh.nnunet_warm_ready_worker import install','from scripts.colab_hfh.nnunet_seed0_inner_worker import install')
    exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
