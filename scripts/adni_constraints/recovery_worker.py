"""Resume the frozen pilot with verified local archives and node-local writes.

Training code/protocols remain frozen. Authorized replay after the interrupted
checkpoint is an explicit extra-compute charge, never erased from provenance.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_completed(stage, root, run, archived):
    relative = Path(run['model'])/run['objective']/'fold0/seed0/run'
    output = root/relative
    done, config = read(output/'completion.json'), read(output/'config.json')
    hashes = {'config_sha256':'config.json','metrics_sha256':'metrics.csv',
              'budget_ledger_sha256':'budget_ledger.json','validation_best_sha256':'validation_best.json',
              'calibration_sha256':'calibration.json'}
    for key, filename in hashes.items():
        if done[key] != digest(output/filename):
            raise ValueError('Metadata changed: '+str(output/filename))
    checkpoint = output/'checkpoint_best.pt'
    if checkpoint.exists():
        actual = digest(checkpoint)
    else:
        entry = archived[str(relative/'checkpoint_best.pt')]
        if entry['completion_sha256'] != digest(output/'completion.json'):
            raise ValueError('Archive belongs to a different completed run')
        actual = entry['sha256']
    if actual != done['checkpoint_best_sha256']:
        raise ValueError('Best checkpoint binding differs')
    latest = output/'checkpoint_latest.pt'
    if latest.exists():
        if digest(latest) != done['checkpoint_latest_sha256']:
            raise ValueError('Latest checkpoint changed')
    else:
        retention = read(output/'retention.json')
        if (retention['completion_sha256'] != digest(output/'completion.json')
                or retention['verified_latest_sha256'] != done['checkpoint_latest_sha256']):
            raise ValueError('Retired checkpoint binding differs')
    protocol = read(stage/'protocols'/run['model']/'protocol.json')
    schedule = protocol['models'][run['alias']]['objectives'][run['objective']]
    if (done['status'] != 'complete' or done['updates'] != run['updates'] or done['discarded_attempts'] != 0
            or config['augmentation'] != 'none' or config['smoke_test']
            or config['model_identity'] != run['model'] or config['objective'] != run['objective']
            or config['protocol_sha256'] != digest(stage/'protocols'/run['model']/'protocol.json')
            or done['consumed_training_flops'] != schedule['consumed_training_flops']
            or done['consumed_training_flops'] > protocol['budget_flops']):
        raise ValueError('Completed identity/accounting mismatch')
    validation = read(output/'validation_latest.json')
    retention = read(output/'retention.json')
    if digest(output/'validation_latest.json') != retention['validation_latest_sha256']:
        raise ValueError('Final validation changed')
    with (output/'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 20 or float(rows[-1]['macro_hard_dice']) != validation['summary']['hippocampus_hard_dice']:
        raise ValueError('Validation history differs')
    return dict(model=run['model'],objective=run['objective'],updates=done['updates'],
                best_dice=done['best_validation_hard_dice'],final_dice=validation['summary']['hippocampus_hard_dice'],
                selected_step=done['selected_step'],optimization_gflops=done['consumed_training_flops']/1e9,
                calibration_gflops=done['calibration_flops']/1e9,
                completion_sha256=digest(output/'completion.json'),
                validation_latest_sha256=digest(output/'validation_latest.json'))


def replay_ledger(old, saved_step, schedule):
    from scripts.capped_depth_delay.schedule import training_flops
    charged = old['charged_attempts']
    if not 0 <= saved_step <= charged or old['training_flops'] != training_flops(charged, schedule):
        raise ValueError('Original interrupted ledger is inconsistent')
    new = {**old, 'charged_attempts':saved_step, 'training_flops':training_flops(saved_step,schedule)}
    return new, dict(discarded_updates=charged-saved_step,
                     extra_training_flops=old['training_flops']-new['training_flops'],
                     original_charged_attempts=charged,resume_update=saved_step)


def quota_free_gib():
    result = subprocess.check_output(['beegfs','quota','list-usage','--uids','current','--pool','pool_home'],text=True)
    found = re.search(r'(\d+(?:\.\d+)?)GiB/(\d+(?:\.\d+)?)GiB', result)
    if not found:
        raise ValueError('Cannot read actual quota headroom')
    return float(found[2])-float(found[1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recovery',type=Path,required=True)
    args=parser.parse_args()
    recovery=args.recovery
    control=read(recovery/'recovery_plan.json')
    for name, sha in read(recovery/'manifest.json').items():
        if digest(recovery/name)!=sha:
            raise ValueError('Recovery source or control changed: '+name)
    stage,root=Path(control['original_stage']),Path(control['result_root'])
    from scripts.adni_constraints.worker import verify_stage, verify_run
    plan=verify_stage(stage)
    if digest(stage/'manifest.json')!=control['original_manifest_sha256']:
        raise ValueError('Frozen training source changed')
    archive=read(recovery/'checkpoint_archive.json')
    entries={r['relative_path']:r for r in archive['files']}
    import monai
    import torch
    if (torch.__version__!=plan['runtime_versions']['torch'] or monai.__version__!=plan['runtime_versions']['monai']
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError('Wrong training runtime')
    scratch=Path('/tmp')/f'hippo-adni-recovery-{os.getuid()}-{os.environ["SLURM_JOB_ID"]}'
    scratch.mkdir(mode=0o700,exist_ok=False)
    if shutil.disk_usage(scratch).free < 12*1024**3:
        raise ValueError('Need 12 GiB free node-local storage')
    write(recovery/'launch.json',dict(job_id=os.environ['SLURM_JOB_ID'],scratch=str(scratch),
          gpu=torch.cuda.get_device_name(0),augmentation='none',extra_compute=control['replay']))
    results=[]
    for run in plan['runs']:
        relative=Path(run['model'])/run['objective']/'fold0/seed0/run'
        target=root/relative
        if (target/'completion.json').exists():
            row=verify_completed(stage,root,run,entries)
        else:
            if quota_free_gib()<1.5:
                raise RuntimeError('Stop before training: insufficient quota for the next best-model export')
            output=scratch/relative
            output.parent.mkdir(parents=True,exist_ok=True)
            is_resume=str(relative)==control['interrupted_relative_path']
            if is_resume:
                source=root/control['interrupted_archive_relative_path']
                if digest(source/'checkpoint_latest.pt')!=control['checkpoint_sha256']:
                    raise ValueError('Resume checkpoint differs from verified backup')
                shutil.copytree(source,output)
                checkpoint=torch.load(output/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
                if (checkpoint['step']!=control['replay']['resume_update']
                        or checkpoint['charged_attempts']!=checkpoint['step']
                        or checkpoint['configuration']!=read(output/'config.json')):
                    raise ValueError('Checkpoint resume identity differs')
                del checkpoint
                protocol=read(stage/'protocols'/run['model']/'protocol.json')
                schedule=protocol['models'][run['alias']]['objectives'][run['objective']]
                old=read(output/'budget_ledger.json')
                ledger,replay=replay_ledger(old,control['replay']['resume_update'],schedule)
                if replay!=control['replay']:
                    raise ValueError('Authorized replay allowance changed')
                write(output/'original_interrupted_ledger.json',old)
                write(output/'budget_ledger.json',ledger)
                write(output/'recovery_provenance.json',dict(**replay,
                    original_failed_job=698935,recovery_plan_sha256=digest(recovery/'recovery_plan.json'),
                    accounting='Original failed-attempt compute retained here; replay allowance explicitly authorized. Training ledger tracks the restored trajectory; physical total adds extra_training_flops.'))
            print(json.dumps(dict(starting=run,resume=is_resume,scratch=str(output))),flush=True)
            subprocess.run([sys.executable,'-m','scripts.adni_constraints.train','--model',run['alias'],
                '--objective',run['objective'],'--fold','0','--seed','0','--pkl',plan['remote_pkl'],
                '--protocol-dir',str(stage/'protocols'/run['model']),'--output',str(output),
                '--threads','4','--resume'],check=True)
            verify_run(stage,output,run)  # Verify and retire only the node-local optimizer state.
            if target.exists():
                raise FileExistsError('Refusing to replace any prior result: '+str(target))
            pending=target.with_name('run.recovery-export')
            if pending.exists():
                raise FileExistsError('Previous export needs inspection')
            shutil.copytree(output,pending)
            for item in output.iterdir():
                if item.is_file() and digest(item)!=digest(pending/item.name):
                    raise ValueError('Export checksum mismatch')
            pending.rename(target)
            row=verify_completed(stage,root,run,entries)
            shutil.rmtree(output)
        overhead=control['replay']['extra_training_flops'] if str(relative)==control['interrupted_relative_path'] else 0
        row.update(recovery_extra_gflops=overhead/1e9,
                   physical_training_gflops=row['optimization_gflops']+overhead/1e9)
        baseline=next((r for r in results if r['model']==row['model'] and r['objective']=='dice'),row)
        row['final_gain_vs_unaugmented_baseline_pp']=100*(row['final_dice']-baseline['final_dice'])
        row['best_gain_vs_unaugmented_baseline_pp']=100*(row['best_dice']-baseline['best_dice'])
        results.append(row)
        write(root/'summary.json',dict(status='complete' if len(results)==56 else 'in_progress',
            completed_runs=len(results),total_runs=56,dataset='ADNI',augmentation='none',fold=0,seed=0,
            recovery_job_id=os.environ['SLURM_JOB_ID'],recovery_extra_gflops=control['replay']['extra_training_flops']/1e9,
            metric='mean whole-hippocampus Dice across 54 crops',runs=results))
        print(json.dumps(dict(completed=row)),flush=True)
    write(recovery/'completion.json',dict(status='complete',completed_runs=56,
        summary_sha256=digest(root/'summary.json'),extra_compute=control['replay']))


if __name__=='__main__':
    main()
