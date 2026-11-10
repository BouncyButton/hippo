"""Colab worker for the frozen five-volume GeoFWI3D SAM3 exploration."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from runtime import (RecoveryWriter, activate, binding, check_ledger, digest,
                     fold_calibration, protocol, read, write)


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                python=random.getstate(),numpy=np.random.get_state())


def set_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    random.setstate(state['python']);np.random.set_state(state['numpy'])


def load(root,p):
    from scripts.cross_dataset_campaign import foundation as original
    from scripts.cross_dataset_campaign.foundation_msd import configure_retained_backward
    from scripts.cross_dataset_campaign.foundation_fast import TransformCache
    configure_retained_backward(); torch.set_num_threads(4)
    original.reset_seed(p['seed'])
    model,processor=original.load_foundation(p,root/'weights/sam3.pt')
    processor.transform=TransformCache(processor.transform,max_bytes=1024**3)
    for name,value in model.named_parameters():value.requires_grad_(original.is_adapter(name))
    parameters=[v for v in model.parameters() if v.requires_grad]
    if sum(v.numel() for v in parameters)!=18497664:
        raise ValueError('LoRA architecture changed')
    criterion=original.native_loss(model);volumes=original.Volumes(p)
    slabs=original.SupportSlabs(volumes,p,processor)
    model.train()
    return original,model,processor,parameters,criterion,volumes,slabs


def project_rows(shared,names):
    indices=[shared['names'].index(n) for n in names]
    return [dict(case=r['case'],anchor=r['anchor'],
                 gram=[[r['gram'][i][j] for j in indices] for i in indices],
                 losses={n:r['losses'][n] for n in names}) for r in shared['rows']]


def preflight(root,durable):
    task=next(t for t in read(root/'plan.json')['tasks'] if t['scientific_arm']=='dice_bands_edge')
    p=protocol(root,task);folder=durable/'calibration/fold0';folder.mkdir(parents=True,exist_ok=True)
    write(folder/'protocol.json',p)
    original,model,processor,parameters,criterion,volumes,slabs=load(root,p)
    from scripts.cross_dataset_campaign.foundation_msd import native_training_check
    native_training_check(model,criterion,parameters,slabs,p,folder)
    # Raw Gram measurements depend on initialization, support and losses, not
    # auxiliary targets or stopping updates. Compute all three losses only once.
    calibration=original.calibrate(model,criterion,parameters,slabs,p,'dice_bands_edge',folder)
    calibration['initialization']=dict(frozen=original.parameter_digest(model,False),
                                        adapter=original.parameter_digest(model,True))
    calibration['inventory']=slabs.inventory
    if len(calibration['rows'])!=20:raise ValueError('Expected20 support slabs')
    write(folder/'calibration.json',calibration)
    write(folder/'receipt.json',dict(status='passed',binding=binding(root,durable),
        calibration_sha256=digest(folder/'calibration.json'),optimizer_updates=0,
        policy='Shared fresh native/bands/edge Gram measurements; target-specific weights are rederived by the frozen calibrator'))
    print('GPU backward checks and20 shared calibration slabs passed; no optimizer updates.',flush=True)


def evaluate(original,model,processor,volumes,p,output,root,durable):
    from fast_eval import EvaluationAudit
    processor.transform.clear();processor.transform.enabled=False
    model.eval().requires_grad_(False)
    identity=dict(checkpoint_sha256=digest(output/'checkpoint_latest.pt'),
        protocol_sha256=digest(output/'protocol.json'),binding=read(durable/'environment_binding.json'),
        extension_sha256=digest(root/'fast_eval.py'))
    reference=SimpleNamespace(forward_plane=lambda m,pr,pixels:original.forward_plane(m,pr,pixels,p))
    helpers=SimpleNamespace(rng_state=rng_state,set_rng=set_rng,write=write,read=read)
    audit=EvaluationAudit(helpers,reference,model,processor,output,output/'evaluation_audit.json',identity)
    if os.environ.get('HIPPO_AUDITED_FAST_EVAL')!='1':audit.fallback('Disabled by notebook setting')
    path=output/'evaluation.json';records=read(path)['records'] if path.exists() else []
    partial=output/'evaluation_partial.npz';marker=output/'evaluation_partial.json'
    try:
        for split,cases in [('training_support',p['train_cases']),('validation',p['evaluation_cases'])]:
            for case in cases:
                if any(r['split']==split and r['case_name']==case for r in records):continue
                if original.stopping():raise SystemExit(75)
                image,truth=volumes.load(case)
                prediction=np.zeros(image.shape,dtype=np.uint8);start=0
                if marker.exists():
                    state=read(marker)
                    if state['identity']!=identity:raise ValueError('Partial evaluation belongs to another checkpoint')
                    if state['case']==case and state['split']==split:
                        if digest(partial)!=state['sha256']:raise ValueError('Partial prediction changed')
                        with np.load(partial,allow_pickle=False) as archive:prediction=archive['prediction']
                        start=state['next_slice']
                audit.begin_volume(case,image.shape[2])
                if start:
                    # A partly audited volume cannot establish a full-volume pass.
                    if audit.mode=='audit':audit.fallback('Resumed a partial volume; original evaluation retained for this checkpoint')
                    audit.slice=start
                    if audit.mode=='fast':audit.slice=0  # Check first resumed slice too.
                with torch.inference_mode():
                    for z in range(start,image.shape[2]):
                        if original.stopping():
                            temp=partial.with_suffix('.tmp')
                            with temp.open('wb') as stream:np.savez(stream,prediction=prediction)
                            temp.replace(partial)
                            write(marker,dict(identity=identity,case=case,split=split,next_slice=z,sha256=digest(partial)))
                            raise SystemExit(75)
                        pixels=np.round(image[:,:,z].T*255).astype(np.uint8)
                        logp,_=audit.plane(model,processor,pixels)
                        audit.slice=z+1
                        probability=logp.exp()
                        if (not torch.isfinite(probability).all() or
                            not torch.allclose(probability.sum(0),torch.ones_like(probability[0]),atol=1e-5)):
                            raise ValueError('Invalid semantic probabilities')
                        prediction[:,:,z]=probability.argmax(0).cpu().numpy().T
                n=len(p['classes'])
                matrix=np.bincount((truth.astype(np.int64)*n+prediction).ravel(),minlength=n*n).reshape(n,n).tolist()
                records.append(dict(case_name=case,split=split,**original.confusion_metrics(matrix,p['classes']),
                    prediction_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                    truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),shape=list(truth.shape)))
                write(path,dict(records=records));partial.unlink(missing_ok=True);marker.unlink(missing_ok=True)
                print('EVALUATED',split,case,records[-1]['macro_hard_dice'],flush=True)
        result=dict(records=records,mean_dice={s:float(np.mean([r['macro_hard_dice'] for r in records if r['split']==s]))
                                              for s in ('training_support','validation')})
        write(path,result);return result
    finally:audit.fast.clear();audit.save()


def train(root,durable,task):
    p=protocol(root,task);output=root/'work'/task['id'];recovery=durable/'recovery'/task['id']
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists() and read(output/'protocol.json')!=p:raise ValueError('Resume protocol differs')
    write(output/'protocol.json',p);check_ledger(output,recovery)
    original,model,processor,parameters,criterion,volumes,slabs=load(root,p)
    shared=fold_calibration(root,durable,0)
    if shared['initialization']!=dict(frozen=original.parameter_digest(model,False),adapter=original.parameter_digest(model,True)):
        raise ValueError('Initial parameters differ from shared calibration')
    if slabs.inventory!=shared['inventory']:raise ValueError('Support inventory changed')
    write(output/'training_inventory.json',slabs.inventory)
    names=['native']+[n for n in ('bands','edge') if n in p['arm']]
    rows=project_rows(shared,names)
    write(output/'calibration_partial.json',dict(names=names,rows=rows))
    calibration=original.calibrate(model,criterion,parameters,slabs,p,p['arm'],output)
    # Preserve the cluster's full loss graph and coefficient insertion order.
    calibration['weights']={'bands':0.,'edge':0.,**calibration['weights']}
    calibration['shared_calibration_sha256']=digest(durable/'calibration/fold0/calibration.json')
    write(output/'calibration.json',calibration)
    write(output/'runtime_recipe.json',dict(updates=p['updates'],learning_rate_horizon_updates=200,
        augmentation='none',precision='CUDA BF16 autocast',transform_cache_cpu_gib=1,
        shared_calibration=True,learned_features_cached_during_training=False,
        training_source_sha256=digest(root/'training_loop.py'),binding=read(durable/'environment_binding.json')))
    latest=output/'checkpoint_latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if state['step']!=read(output/'checkpoint_ledger.json')['step'] or len(state['history'])!=state['step']:
            raise ValueError('Checkpoint and ledger disagree')
    writer=RecoveryWriter(output,recovery)
    coherent=False
    try:
        from training_loop import train as training
        try:
            training(model,criterion,parameters,slabs,p,calibration,p['arm'],output,writer,recovery)
            coherent=True
        except SystemExit as error:
            if error.code==75:
                check_ledger(output,recovery);coherent=True
            raise
        writer.capture(force=True)
        result=evaluate(original,model,processor,volumes,p,output,root,durable)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        history=read(output/'training_history.json')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for name in ('native','total'):axes[0].plot([r['update'] for r in history],[r[name] for r in history],label=name)
        for name in ('bands','edge'):axes[1].plot([r['update'] for r in history],[r[name] for r in history],label=name)
        for ax in axes:ax.set_xlabel('Update');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(output/'curves.png',dpi=140);plt.close(fig)
        validation=[r for r in result['records'] if r['split']=='validation']
        write(output/'completion.json',dict(status='complete',updates=p['updates'],
            validation_dice=result['mean_dice']['validation'],training_dice=result['mean_dice']['training_support'],
            foreground_fn=sum(r['foreground_fn'] for r in validation),foreground_fp=sum(r['foreground_fp'] for r in validation),
            primary_checkpoint='final_update',checkpoint_sha256=digest(latest),
            evaluation_sha256=digest(output/'evaluation.json'),protocol_sha256=digest(output/'protocol.json')))
    finally:writer.close(force=coherent)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('preflight','train'))
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--durable',type=Path,required=True)
    parser.add_argument('--task');parser.add_argument('--deadline',type=float)
    args=parser.parse_args();root=args.bundle.resolve();durable=args.durable.resolve();activate(root)
    from scripts.cross_dataset_campaign.common import install_signals
    install_signals()
    if args.deadline:os.environ['HIPPO_DEADLINE']=str(args.deadline)
    if args.command=='preflight':preflight(root,durable)
    else:
        task=next(t for t in read(root/'plan.json')['tasks'] if t['id']==args.task)
        train(root,durable,task)


if __name__=='__main__':main()
