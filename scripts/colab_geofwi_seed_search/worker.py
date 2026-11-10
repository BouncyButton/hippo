"""Bind frozen baseline recipes to the existing capped trainer and Drive recovery."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).parent))
import storage_runtime as io
io.activate(Path(__file__).parent)
from scripts.cross_dataset_campaign import conventional as exp
from scripts.adni_constraints import objectives as binary
import torch


def configure(p,output):
    exp.configure(p,'dice')
    exp.base.build_loss_and_optimizer=exp.ORIGINAL_LOSS_OPTIMIZER
    if p['weight_ce'] and p['model_identity']!='nnunet':
        from scripts.cross_dataset_campaign.msd_ce_constraints import training_loss
        exp.trainer.training_loss=training_loss
    else:
        exp.trainer.training_loss=binary.training_loss
    builder=exp.base.build_model
    def build(name,architecture):
        model=builder(name,architecture)
        record=dict(model_state_sha256=io.model_digest(model))
        path=output/'initialization.json'
        if path.exists() and io.read(path)!=record:raise ValueError('Initialization changed on resume')
        io.write(path,record)
        return model
    exp.base.build_model=build
    io.write(output/'runtime_recipe.json',dict(dataset='GeoFWI3D',model=p['model_identity'],
        augmentation='none',weight_dice=1,weight_ce=p['weight_ce'],
        dice_include_background=p['model_identity']!='nnunet',
        native_deep_supervision=p['model_identity']=='nnunet',
        primary_checkpoint='final_budget',budget_flops=p['budget_flops'],
        protocol_sha256=io.digest(output/'protocol.json') if (output/'protocol.json').exists() else None))


def preflight(p,folder,output):
    """Real 96³ forward/backward plus exact replay of a saved model/optimizer/RNG."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Native BF16 CUDA GPU required')
    device=torch.device('cuda');exp.base.seed_everything(0);torch.set_num_threads(2)
    model=exp.base.build_model(p['trainer_alias'],io.read(folder/'nnunet_architecture.json')).to(device)
    if sum(x.numel() for x in model.parameters())!=p['models'][p['trainer_alias']]['parameters']:
        raise ValueError('Architecture differs from profile')
    criterion,optimizer=exp.base.build_loss_and_optimizer(p['trainer_alias'],model,p)
    native=criterion if p['model_identity']=='nnunet' else None
    split=io.read(folder/'splits_10patients.json')[0]
    data=exp.SeismicDataset(io.read(folder/'dataset.json'),split['train'][:1])
    x=data[0]['image'][None].to(device);y=data[0]['label'][None].to(device)
    def step():
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(x)
            loss=exp.trainer.training_loss(logits,y,'dice',{},.2,criterion=native)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError('Nonfinite gradient')
        if native is not None:torch.nn.utils.clip_grad_norm_(model.parameters(),12,foreach=False,error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    first=step()
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        path=Path(tmp)/'resume.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),path)
        expected_loss=step();expected=io.model_digest(model)
        expected_opt=deepcopy(optimizer.state_dict());expected_rng=torch.cuda.get_rng_state_all()
        expected_cpu_rng=torch.get_rng_state()
        # Let load_state_dict place parameters/moments; AdamW's noncapturable step stays on CPU.
        cp=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer'])
        torch.set_rng_state(cp['cpu_rng'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
        actual_loss=step()
        if expected_loss!=actual_loss or expected!=io.model_digest(model):raise ValueError('Resume model/loss mismatch')
        def equal(a,b):
            if isinstance(a,torch.Tensor):return a.device==b.device and torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        if (not equal(expected_opt,optimizer.state_dict()) or not equal(expected_rng,torch.cuda.get_rng_state_all())
                or not torch.equal(expected_cpu_rng,torch.get_rng_state())):
            raise ValueError('Resume optimizer/RNG mismatch')
    evaluated=exp.evaluate(model,data,device,True,p['classes'])
    if len(evaluated['cases'])!=1:raise ValueError('Evaluation failed')
    io.write(output/'gpu_preflight.json',dict(status='passed',model=p['model_identity'],
        first_loss=first,resume_loss=actual_loss,shape=[1,1,96,96,96],
        exact_model_optimizer_rng_replay=True,diagnostic_optimizer_updates=3,
        diagnostic_estimated_flops=3*p['models'][p['trainer_alias']]['objectives']['dice']['flops_per_update'],
        scientific_result=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fold',type=int,required=True,choices=[0,1,2])
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    p=io.read(args.protocol_dir/'protocol.json')
    if args.seed<0 or p['augmentation']!='none' or p['dataset']!='GeoFWI3D':raise ValueError('Invalid request')
    configure(p,output)
    if args.preflight:return preflight(p,args.protocol_dir,output)
    recovery=Path(os.environ['HIPPO_COLAB_RECOVERY'])
    if (recovery/'charged_ledger.json').exists():
        if not (output/'budget_ledger.json').exists() or io.read(output/'budget_ledger.json')!=io.read(recovery/'charged_ledger.json'):
            raise RuntimeError('Charged updates newer than recovery snapshot: preserve recovery and audit; do not reject seed')
    writer=io.RecoveryWriter(output,recovery)
    original_json,original_save=exp.base.atomic_json,exp.base.save_checkpoint
    def atomic(path,value):
        if Path(path).name=='budget_ledger.json':
            io.write(recovery/'charged_ledger.json',value)
            if io.read(recovery/'charged_ledger.json')!=value:raise IOError('Drive ledger readback failed')
        return original_json(path,value)
    def save(path,value):
        original_save(path,value)
        if Path(path).name=='checkpoint_latest.pt':writer.capture()
    exp.base.atomic_json,exp.base.save_checkpoint=atomic,save
    config=argparse.Namespace(model=p['trainer_alias'],objective='dice',fold=args.fold,seed=args.seed,
        pkl=Path(p['data']),output=output,protocol_dir=args.protocol_dir,device='cuda',
        threads=2,resume=True,smoke_steps=0)
    coherent=False
    try:
        with (output/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            exp.trainer.run(config)
        coherent=True
    except SystemExit as exc:
        coherent=exc.code in (None,0,75);raise
    finally:writer.close(force=coherent)


if __name__=='__main__':main()
