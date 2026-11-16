"""Seed-zero LR diagnosis with detached, per-head foreground measurements."""
import json
from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss, worker_source


class HeadProbeLoss(ReadyLoss):
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


if __name__=='__main__':
    source=worker_source().replace('from scripts.colab_hfh.nnunet_warm_ready_worker import ReadyLoss',
        'from scripts.colab_hfh.nnunet_seed0_repair_worker import HeadProbeLoss as ReadyLoss')
    exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
