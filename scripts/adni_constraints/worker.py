"""Sequential execution with verified results, no augmentation, bounded storage."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from scripts.adni_baselines.core import MODELS, BUDGET, base
from scripts.adni_constraints.objectives import ARMS
from scripts.capped_depth_delay.core import ARTIFACTS


def verify_stage(stage, check_data=True):
    for name, sha in json.loads((stage/'manifest.json').read_text()).items():
        if base.digest(stage/name) != sha:
            raise ValueError('Frozen file changed: '+name)
    plan = json.loads((stage/'plan.json').read_text())
    if (plan['num_runs'] != 56 or plan['fold'] != 0 or plan['seed'] != 0
            or plan['augmentation'] != 'none' or plan['budget_flops_per_run'] != BUDGET
            or plan['objectives'] != list(ARMS) or plan['models'] != list(MODELS)):
        raise ValueError('Campaign identity changed')
    if check_data and base.digest(Path(plan['remote_pkl'])) != plan['pkl_sha256']:
        raise ValueError('ADNI dataset differs')
    return plan


def verify_run(stage, output, run):
    done = json.loads((output/'completion.json').read_text())
    config = json.loads((output/'config.json').read_text())
    directory = stage/'protocols'/run['model']
    protocol = json.loads((directory/'protocol.json').read_text())
    split = json.loads((directory/'splits_10patients.json').read_text())[0]
    schedule = protocol['models'][run['alias']]['objectives'][run['objective']]
    receipt_path = output/'retention.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    for key, name in ARTIFACTS.items():
        if name == 'checkpoint_latest.pt' and not (output/name).exists() and receipt:
            if (receipt['completion_sha256'] != base.digest(output/'completion.json')
                    or receipt['verified_latest_sha256'] != done[key]):
                raise ValueError('Retention receipt changed')
        elif base.digest(output/name) != done[key]:
            raise ValueError('Completed artifact changed: '+name)
    if (done['status'] != 'complete' or done['updates'] != run['updates']
            or done['training_attempts_charged'] != run['updates'] or done['discarded_attempts'] != 0
            or config['augmentation'] != 'none' or config['smoke_test']
            or config['model_identity'] != run['model'] or config['objective'] != run['objective']
            or config['fold'] != 0 or config['seed'] != 0
            or config['protocol_sha256'] != base.digest(directory/'protocol.json')
            or config['dataset_sha256'] != protocol['pkl_sha256']
            or config['train_cases'] != split['train'] or config['validation_cases'] != split['val']
            or done['consumed_training_flops'] != schedule['consumed_training_flops']
            or done['calibration_flops'] != schedule['calibration_flops']
            or not 0 <= BUDGET-done['consumed_training_flops'] < schedule['flops_per_update']):
        raise ValueError('Run identity or compute accounting mismatch')
    latest = json.loads((output/'validation_latest.json').read_text())
    if [r['case_name'] for r in latest['cases']] != split['val']:
        raise ValueError('Validation population changed')
    with (output/'metrics.csv').open(newline='') as stream:
        history = list(csv.DictReader(stream))
    if (len(history) != 20 or int(history[-1]['step']) != run['updates']
            or float(history[-1]['macro_hard_dice']) != latest['summary']['hippocampus_hard_dice']
            or max(float(row['macro_hard_dice']) for row in history) != done['best_validation_hard_dice']
            or done['calibration_attempts'] != int(run['objective'] != 'dice')):
        raise ValueError('Validation history or calibration count differs')
    result = dict(model=run['model'], objective=run['objective'], updates=done['updates'],
                  best_dice=done['best_validation_hard_dice'],
                  final_dice=latest['summary']['hippocampus_hard_dice'],
                  selected_step=done['selected_step'], optimization_gflops=done['consumed_training_flops']/1e9,
                  calibration_gflops=done['calibration_flops']/1e9,
                  completion_sha256=base.digest(output/'completion.json'),
                  validation_latest_sha256=base.digest(output/'validation_latest.json'))
    # This only releases a newly generated, fully verified optimizer snapshot.
    # Best model and every metadata file remain on the cluster.
    if receipt is None:
        base.atomic_json(receipt_path, dict(completion_sha256=result['completion_sha256'],
                         verified_latest_sha256=done['checkpoint_latest_sha256'],
                         best_checkpoint_sha256=done['checkpoint_best_sha256'],
                         validation_latest_sha256=result['validation_latest_sha256'],
                         policy='Verified completed latest optimizer checkpoint removed; best FP32 model retained'))
    (output/'checkpoint_latest.pt').unlink(missing_ok=True)
    return result


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
        raise RuntimeError('Expected frozen CUDA/BF16 environment')
    args.root.mkdir(parents=True, exist_ok=True)
    base.atomic_json(args.root/'launch.json', dict(plan_sha256=base.digest(args.stage/'plan.json'),
        manifest_sha256=base.digest(args.stage/'manifest.json'), gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,monai=monai.__version__,augmentation='none'))
    results = []
    for run in plan['runs']:
        verify_stage(args.stage, check_data=False)
        output = args.root/run['model']/run['objective']/'fold0/seed0/run'
        if not (output/'completion.json').exists():
            print(json.dumps(dict(starting=run)), flush=True)
            subprocess.run([sys.executable,'-m','scripts.adni_constraints.train',
                '--model',run['alias'],'--objective',run['objective'],'--fold','0','--seed','0',
                '--pkl',plan['remote_pkl'],'--protocol-dir',str(args.stage/'protocols'/run['model']),
                '--output',str(output),'--threads','4','--resume'], check=True)
        result = verify_run(args.stage, output, run)
        baseline = next((r for r in results if r['model']==result['model'] and r['objective']=='dice'), result)
        result['final_gain_vs_unaugmented_baseline_pp'] = 100*(result['final_dice']-baseline['final_dice'])
        result['best_gain_vs_unaugmented_baseline_pp'] = 100*(result['best_dice']-baseline['best_dice'])
        results.append(result)
        base.atomic_json(args.root/'summary.json', dict(status='complete' if len(results)==56 else 'in_progress',
            completed_runs=len(results), total_runs=56, dataset='ADNI',augmentation='none',fold=0,seed=0,
            metric='mean whole-hippocampus Dice across 54 crops',runs=results))
        print(json.dumps(dict(completed=result)), flush=True)


if __name__ == '__main__':
    main()
