"""Run the four frozen binary baselines sequentially and verify outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .core import BUDGET, MODELS, base


def verify_stage(stage, check_data=True):
    for name, sha in json.loads((stage/'manifest.json').read_text()).items():
        if base.digest(stage/name) != sha:
            raise ValueError('Frozen file changed: '+name)
    plan = json.loads((stage/'plan.json').read_text())
    if (plan['num_runs'] != 4 or plan['fold'] != 0 or plan['seed'] != 0
            or plan['budget_flops_per_model'] != BUDGET
            or [r['model'] for r in plan['runs']] != list(MODELS)):
        raise ValueError('Campaign identity differs')
    if check_data and base.digest(Path(plan['remote_pkl'])) != plan['pkl_sha256']:
        raise ValueError('Cluster dataset differs from local verified ADNI copy')
    return plan


def verify_run(stage, root, run):
    output = root/run['model']/'fold0/seed0/run'
    done = json.loads((output/'completion.json').read_text())
    config = json.loads((output/'config.json').read_text())
    protocol_dir = stage/'protocols'/run['model']
    protocol = json.loads((protocol_dir/'protocol.json').read_text())
    split = json.loads((protocol_dir/'splits_10patients.json').read_text())[0]
    for key, name in [('config_sha256','config.json'),('checkpoint_best_sha256','checkpoint_best.pt'),
                      ('checkpoint_latest_sha256','checkpoint_latest.pt'),('metrics_sha256','metrics.csv'),
                      ('budget_ledger_sha256','budget_ledger.json'),('validation_best_sha256','validation_best.json')]:
        if done[key] != base.digest(output/name):
            raise ValueError('Completion artifact changed: '+name)
    if (done['status'] != 'complete' or done['updates'] != run['updates']
            or done['training_attempts_charged'] != run['updates'] or done['discarded_attempts'] != 0
            or config['train_cases'] != split['train'] or config['validation_cases'] != split['val']
            or config['smoke_test'] or config['dataset_sha256'] != protocol['pkl_sha256']
            or config['protocol_sha256'] != base.digest(protocol_dir/'protocol.json')):
        raise ValueError('Incomplete or mismatched ADNI run')
    cost = protocol['models'][run['alias']]['flops_per_update']
    if done['consumed_training_flops'] != run['updates']*cost or not run['updates']*cost <= BUDGET < (run['updates']+1)*cost:
        raise ValueError('Compute accounting differs')
    latest = json.loads((output/'validation_latest.json').read_text())
    if [r['case_name'] for r in latest['cases']] != split['val']:
        raise ValueError('Validation cases differ')
    return dict(model=run['model'], updates=done['updates'],
                best_dice=done['best_validation_hard_dice'],
                final_dice=latest['summary']['hippocampus_hard_dice'],
                selected_step=done['selected_step'], gflops=done['consumed_training_flops']/1e9,
                completion_sha256=base.digest(output/'completion.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    plan = verify_stage(args.stage)
    import monai
    import torch
    if (monai.__version__ != plan['runtime_versions']['monai']
            or torch.__version__ != plan['runtime_versions']['torch']
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('Expected frozen MSD CUDA/BF16 environment')
    args.root.mkdir(parents=True,exist_ok=True)
    base.atomic_json(args.root/'launch.json', dict(plan_sha256=base.digest(args.stage/'plan.json'),
                     manifest_sha256=base.digest(args.stage/'manifest.json'),
                     gpu=torch.cuda.get_device_name(0),torch=torch.__version__,monai=monai.__version__))
    results = []
    for run in plan['runs']:
        verify_stage(args.stage,check_data=False)
        output = args.root/run['model']/'fold0/seed0/run'
        print(json.dumps(dict(starting=run['model'],updates=run['updates'])),flush=True)
        subprocess.run([sys.executable,'-m','scripts.adni_baselines.core','--model',run['alias'],
            '--fold','0','--seed','0','--pkl',plan['remote_pkl'],
            '--protocol-dir',str(args.stage/'protocols'/run['model']),
            '--output',str(output),'--threads','4','--resume'],check=True)
        results.append(verify_run(args.stage,args.root,run))
        base.atomic_json(args.root/'summary.json',dict(status='complete' if len(results)==4 else 'in_progress',
                         dataset='ADNI',fold=0,seed=0,metric='whole-hippocampus Dice',runs=results,
                         msd_reference_note=plan['comparison']))
        print(json.dumps(dict(completed=results[-1])),flush=True)


if __name__ == '__main__':
    main()
