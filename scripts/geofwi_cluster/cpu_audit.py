"""Read-only architecture/initialization and optimizer audit; no training updates."""
import argparse
from pathlib import Path
import sys

ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--bundle',required=True);a=ap.parse_args()
root=a.stage.resolve();child=root/'bundles'/a.bundle
sys.path.insert(0,str(child))
import worker
import torch
torch.set_num_threads(2)
rows=[]
models=['unetr_vit_b16'] if a.bundle=='unetr' else ['nnunet','swinunetr','basic_unet16']
for model_name in models:
    for seed in range(3):
        folder=root/'protocols'/f'GeoFWI3D_{model_name}_baseline_f1s{seed}'
        p=worker.io.read(folder/'protocol.json')
        worker.exp.configure(p,'dice')
        worker.exp.base.seed_everything(seed)
        if a.bundle=='unetr':
            import rescue_objective
            model=rescue_objective.build_model(model_name,worker.io.read(folder/'nnunet_architecture.json'))
            rescue_objective.dropout_audit(model)
        else:
            model=worker.exp.build_model(p,worker.io.read(folder/'nnunet_architecture.json'))
        actual=worker.io.model_digest(model)
        assert actual==p['reference_initializations'][str(seed)],(model_name,seed,actual)
        assert sum(v.numel() for v in model.parameters())==p['models'][p['trainer_alias']]['parameters']
        loss,opt=worker.exp.ORIGINAL_LOSS_OPTIMIZER(p['trainer_alias'],model,p)
        assert opt.param_groups[0]['lr']==p['initial_lrs'][p['trainer_alias']]
        rows.append(dict(model=model_name,seed=seed,sha256=actual,optimizer=type(opt).__name__,lr=opt.param_groups[0]['lr']))
        del model,loss,opt
worker.io.write(root/f'cpu_audit_{a.bundle}.json',dict(status='passed',updates=0,rows=rows))
print(a.bundle,'CPU initialization/architecture/optimizer audit passed',flush=True)
