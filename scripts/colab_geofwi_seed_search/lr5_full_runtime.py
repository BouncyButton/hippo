"""One higher-LR trial and the previously authorized conditional matched BANDS."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time


def bind(campaign):
    campaign.ARMS = (*campaign.ARMS, 'lr5', 'lr5_bands')
    campaign.LRS.update(lr5=5e-4, lr5_bands=5e-4)


def run(root, durable, hours):
    sys.path.insert(0, str(root))
    import lr3_runtime as campaign
    io = campaign.io
    root = io.activate(root)
    campaign.verify(root)
    # Same frozen implementation and GPU environment; reuse verified replay checks.
    campaign.preflight(root, durable, 'lr3')
    reference = campaign.final_result(root, durable, 'lr3')
    if not reference or abs(reference['final_dice'] - .4855053508265538) > 1e-12:
        raise ValueError('Verified completed LR3 reference required')
    bind(campaign)
    deadline = time.time() + hours * 3600
    work = root / 'work'
    work.mkdir(exist_ok=True)
    with (work / 'augmentation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        authorization = dict(
            instruction='Improve UNETR by testing one higher learning rate; conditional matched BANDS only above 50% final Dice',
            peak_lr=5e-4, control_lr=3e-4, lr_horizon_updates=477,
            halfway_gate=False, cap_flops_per_arm=campaign.CAP,
            maximum_new_optimization_flops=campaign.CAP * 2,
            only_training_change='Initial/peak LR: 3e-4 -> 5e-4',
            driver_sha256=io.digest(__file__), time=time.time())
        io.write(durable / 'lr5_full_budget_authorization.json', authorization)
        runs = {}
        for arm, total, preflight_arm in [('lr5', 414, 'lr3'), ('lr5_bands', 407, 'xy_bands')]:
            if arm == 'lr5_bands' and not campaign.qualifies(runs['lr5']):
                break
            campaign.preflight(root, durable, preflight_arm)
            if campaign.settings(root, arm)[2] != total:
                raise ValueError('Full-budget update count changed')
            result = campaign.final_result(root, durable, arm)
            if result is None:
                out = work / campaign.task(arm)['id']
                out.mkdir(parents=True, exist_ok=True)
                io.write(out / 'experiment_authorization.json', authorization)
                io.write(durable / 'lr5_full_budget_status.json', dict(
                    status='running', active_arm=arm, target_updates=total,
                    peak_lr=5e-4, halfway_gate=False, reference_lr3=reference,
                    completed_runs=runs, cap_flops_per_arm=campaign.CAP))
                print('RUN', arm, 'to', total, 'total updates; peak LR 5e-4; no halfway gate', flush=True)
                if not campaign.execute(root, durable, arm, total, deadline):
                    io.write(durable / 'lr5_full_budget_status.json', dict(
                        status='paused', active_arm=arm, completed_runs=runs, target_updates=total))
                    return
                result = campaign.final_result(root, durable, arm)
            if not result:
                raise RuntimeError('Full-budget archive not verified')
            runs[arm] = result
        baseline = runs['lr5']
        bands = runs.get('lr5_bands')
        value = dict(status='complete', reference_lr3=reference, lr5_baseline=baseline,
            final_dice_delta_vs_lr3_percentage_points=100 * (baseline['final_dice'] - reference['final_dice']),
            threshold_passed=campaign.qualifies(baseline), bands=bands,
            bands_status='complete' if bands else 'not_triggered',
            bands_delta_percentage_points=campaign.comparison(root, durable, baseline, bands, 'lr5', 'lr5_bands') if bands else None,
            cap_flops_per_arm=campaign.CAP, halfway_gate=False,
            note='One fold/seed development trial; only peak LR differs from LR3 control')
        io.write(durable / 'lr5_full_budget_status.json', value)
        io.write(durable / 'LR5_FULL_BUDGET_FINISHED.json', value)
        print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.hours <= 2:
        parser.error('Session limit must be in (0,2] hours')
    run(args.bundle.resolve(), args.durable.resolve(), args.hours)
