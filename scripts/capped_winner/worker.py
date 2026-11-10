"""One sequential allocation, resumable progress and verified result retention."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from scripts.campaign90 import BUDGET, MODELS, digest, write_json
from scripts.capped_replication_worker import verify_checkpoints, store_metadata, verify_stored
from scripts.capped_depth_delay.schedule import training_flops
from .objectives import ARM
from .prepare import CAMPAIGN

STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def check_frozen(stage, pkl=None):
    source = stage/'source'
    manifest = read(stage/'source_manifest.json')
    for name, sha in manifest.items():
        assert digest(source/name) == sha, name
    plan = read(source/'experiments'/CAMPAIGN/'plan.json')
    expected = {(m,f,s) for m in MODELS for f in range(3) for s in range(3)}
    assert {(r['model'],r['fold'],r['seed']) for r in plan['runs']} == expected
    assert len(plan['runs']) == plan['total_runs'] == 36
    assert plan['budget_flops'] == BUDGET and not plan['calibration_in_cap']
    if pkl is not None:
        assert digest(pkl) == plan['dataset_sha256']
    assert digest(source/'paired_references.json') == plan['references_sha256']
    assert digest(source/'winner_loss_profile.json') == plan['loss_profile_sha256']
    for model, spec in MODELS.items():
        directory = source/'experiments'/CAMPAIGN/model
        p = read(directory/'protocol.json')
        assert p['requested_arm_order'] == [ARM]
        assert p['folds'] == p['seeds'] == [0,1,2]
        assert p['lr_horizon_updates'] == spec['lr_horizon'] and not p['early_stopping']
        for name, sha in p['training_source_sha256'].items():
            assert manifest[name] == sha, name
        for filename,key in [('splits_10patients.json','splits_sha256'),
                             ('nnunet_architecture.json','nnunet_architecture_sha256'),
                             ('native_geometry.json','native_geometry_sha256')]:
            assert digest(directory/filename) == p[key]
        for split in read(directory/'splits_10patients.json'):
            assert len(set(split['train'])) == 10 and len(set(split['val'])) == 52
            assert not set(split['train']) & set(split['val'])
        schedule = p['models'][spec['alias']]['objectives'][ARM]
        n = schedule['max_steps']
        assert training_flops(n,schedule) <= BUDGET < training_flops(n+1,schedule)
    return plan


def report(root, plan, rows, state, current=None):
    refs = read(root/'source/paired_references.json')
    reference = {(r['model'],r['fold'],r['seed'],r['objective']):r for r in refs}
    for row in rows:
        for objective,name in [('dice','baseline'),('dice_bands','original_bands')]:
            other = reference[(row['model'],row['fold'],row['seed'],objective)]
            row[name+'_final_dice'] = other['final_dice']
            row['gain_vs_'+name+'_pp'] = 100*(row['final_dice']-other['final_dice'])
    summary = {}
    for model in plan['models']:
        values = [r['final_dice'] for r in rows if r['model']==model]
        summary[model] = dict(completed=len(values),planned=9,
            mean_final_dice=statistics.mean(values) if values else None,
            sd_final_dice=statistics.stdev(values) if len(values)>1 else None)
    write_json(root/'status.json',dict(state=state,completed=len(rows),planned=36,current=current,
        job_id=os.environ.get('SLURM_JOB_ID')))
    write_json(root/'results.json',dict(status=state,budget_flops=BUDGET,
        budget_scope='optimization_only',calibration_in_cap=False,runs=rows,by_model=summary,
        comparison=plan['primary_comparison'],inference=plan['inference']))
    if rows:
        with (root/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stop(signum, frame):
    global STOP
    STOP=True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--pkl',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    plan=check_frozen(args.stage,args.pkl)
    if args.check_only:
        print(json.dumps(dict(status='passed',runs=36,strict_budget_verified=True)),flush=True)
        return
    import torch,monai
    assert os.getuid()==1395 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert torch.__version__==plan['runtime_versions']['torch'] and monai.__version__==plan['runtime_versions']['monai']
    assert args.root is not None and args.pkl is not None
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source=args.stage/'source'
        if not (root/'source').exists():(root/'source').symlink_to(source.resolve(),target_is_directory=True)
        assert (root/'source').resolve()==source.resolve()
        binding=dict(manifest_sha256=digest(args.stage/'source_manifest.json'))
        if (root/'binding.json').exists():assert read(root/'binding.json')==binding
        else:write_json(root/'binding.json',binding)
        signal.signal(signal.SIGUSR1,stop)
        started=time.monotonic()
        rows=[]
        for entry in plan['runs']:
            dest=root/'runs'/entry['model']/f"fold{entry['fold']}_seed{entry['seed']}"/'run'
            if dest.exists():rows.append(verify_stored(dest))
        report(root,plan,rows,'preflight')
        if not (root/'gpu_preflight.json').exists():
            from .tests import production_preflight
            production_preflight(args.stage,args.pkl,root/'gpu_preflight.json')
        from .verify import verify
        for entry in plan['runs']:
            model,fold,seed=entry['model'],entry['fold'],entry['seed']
            if any((r['model'],r['fold'],r['seed'])==(model,fold,seed) for r in rows):continue
            if STOP or time.monotonic()-started>23*3600:
                report(root,plan,rows,'paused_walltime');return
            quota=subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
            match=re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB',quota)
            if not match or float(match[2])-float(match[1])<6:
                report(root,plan,rows,'paused_storage');return
            output=root/'active'/model/f'fold{fold}_seed{seed}'/'run'
            destination=root/'runs'/model/f'fold{fold}_seed{seed}'/'run'
            directory=source/'experiments'/CAMPAIGN/model
            report(root,plan,rows,'training',entry)
            try:
                command=[sys.executable,'-m','scripts.capped_winner.train','--model',entry['trainer_alias'],
                    '--objective',ARM,'--fold',str(fold),'--seed',str(seed),'--protocol-dir',str(directory),
                    '--output',str(output),'--pkl',str(args.pkl),'--threads','4']
                if (output/'budget_ledger.json').exists():command.append('--resume')
                subprocess.run(command,check=True)
                row=verify(output,directory,model,fold,seed)
                proofs=verify_checkpoints(output,row)
                destination.parent.mkdir(parents=True,exist_ok=True)
                row=store_metadata(output,destination,row,proofs)
                rows.append(row);report(root,plan,rows,'running')
                # Results and both checkpoint hashes verified in permanent
                # storage before removing this completed run's scratch copy.
                assert output.is_relative_to(root/'active') and not output.is_symlink()
                shutil.rmtree(output)
                print(json.dumps(dict(completed=len(rows),planned=36,result=row)),flush=True)
            except BaseException as error:
                write_json(root/'failure.json',dict(run=entry,error=repr(error)))
                report(root,plan,rows,'failed',entry)
                raise
        report(root,plan,rows,'complete')


if __name__=='__main__':
    main()
