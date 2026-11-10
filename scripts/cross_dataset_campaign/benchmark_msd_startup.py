"""Compare independent interpreter launches with isolated preloaded children."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.cross_dataset_campaign.common import MODELS,read,write
from scripts.cross_dataset_campaign.runtime_worker import TrainerProcess


def fingerprint(value):
    import torch
    h=hashlib.sha256()
    def walk(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());walk(x[k])
        elif isinstance(x,(tuple,list)):
            for item in x:walk(item)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for backend in ('forkserver','interpreter'):
        for identity in MODELS:
            folder=args.stage/'protocols'/identity;p=read(folder/'protocol.json')
            output=args.output/backend/identity;output.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','scripts.cross_dataset_campaign.msd_mixed_losses',
                '--model',p['trainer_alias'],'--objective','dice','--fold','0','--seed','0',
                '--pkl',p['data'],'--protocol-dir',str(folder),'--output',str(output),'--smoke-steps','3']
            print(dict(phase='start',backend=backend,model=identity,time=time.time()),flush=True)
            started=time.perf_counter()
            if backend=='interpreter':
                with (output/'train.log').open('a') as log:
                    subprocess.run(command,cwd=args.stage/'source',stdout=log,stderr=subprocess.STDOUT,check=True)
            else:
                worker=TrainerProcess(command,args.stage/'source',output/'train.log')
                if worker.wait()!=0:raise RuntimeError('Fork-server worker failed; inspect '+str(output/'train.log'))
            elapsed=time.perf_counter()-started
            # Importing torch here happens only in the parent, not in the clean
            # fork server. It does not create a CUDA context.
            import torch
            latest=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            if torch.cuda.is_initialized():raise ValueError('Benchmark parent unexpectedly initialized CUDA')
            row=dict(backend=backend,model=identity,wall_seconds=elapsed,
                state_sha256=fingerprint(latest),validation=read(output/'validation_latest.json'),
                training=read(output/'training_final.json'),calibration=read(output/'calibration.json'),
                completion=read(output/'completion.json'))
            rows.append(row);write(args.output/'results.json',rows)
            print(dict(phase='complete',backend=backend,model=identity,wall_seconds=elapsed),flush=True)
            del latest
    comparisons=[]
    for identity in MODELS:
        old=next(r for r in rows if r['model']==identity and r['backend']=='interpreter')
        new=next(r for r in rows if r['model']==identity and r['backend']=='forkserver')
        for key in ('state_sha256','validation','training','calibration'):
            if old[key]!=new[key]:raise ValueError('Worker isolation mismatch: '+identity+'/'+key)
        comparisons.append(dict(model=identity,exact_checkpoint_rng_optimizer_model_and_metrics=True,
            interpreter_seconds=old['wall_seconds'],forkserver_seconds=new['wall_seconds'],
            speedup=old['wall_seconds']/new['wall_seconds']))
    write(args.output/'summary.json',dict(status='passed',models=comparisons,
        first_forkserver_launch_includes_preload=True,scope='Three-update diagnostic smoke runs only; original four-model recipes, fresh initialization, no augmentation'))


if __name__=='__main__':main()
