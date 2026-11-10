"""User-authorized continuation of the existing LR 3e-4 baseline; no halfway gate."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time


def run(root, durable, hours):
    sys.path.insert(0,str(root))
    import lr3_runtime as campaign
    io=campaign.io
    root=io.activate(root)
    campaign.verify(root)
    campaign.preflight(root,durable,'lr3')
    arm='lr3';task=campaign.task(arm)
    work=root/'work';work.mkdir(exist_ok=True)
    with (work/'augmentation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io.write(durable/'lr3_full_budget_authorization.json',dict(
            instruction='resume the 3e-4 baseline from halfway to full budget; remove only the halfway gate',arm=arm,halfway_gate=False,
            cap_flops=campaign.CAP,total_updates=414,recipe_changed=False,
            driver_sha256=io.digest(__file__),time=time.time()))
        result=campaign.final_result(root,durable,arm)
        if result is None:
            out=work/task['id'];recovery=durable/'recovery'/task['id']
            if not (out/'checkpoint_latest.pt').exists():io.restore_recovery(out,recovery)
            if not (out/'checkpoint_latest.pt').exists():
                raise RuntimeError('No verified checkpoint available. Refusing to repeat the first 207 updates.')
            campaign.check_recipe(out,arm)
            ledger=io.read(out/'budget_ledger.json')
            if ledger['charged_attempts']<207:raise ValueError('Recovery predates the halfway checkpoint')
            full=campaign.settings(root,arm)[2]
            if full!=414:raise ValueError('Full-budget schedule changed')
            io.write(durable/'lr3_full_budget_status.json',dict(status='running',
                starting_updates=ledger['charged_attempts'],target_updates=full,halfway_gate=False,
                cap_flops=campaign.CAP,recipe_changed=False))
            print('RESUME LR3',ledger['charged_attempts'],'->',full,'updates; no halfway gate',flush=True)
            if not campaign.execute(root,durable,arm,full,time.time()+hours*3600):
                io.write(durable/'lr3_full_budget_status.json',dict(status='paused',target_updates=full))
                return
            result=campaign.final_result(root,durable,arm)
        if not result:raise RuntimeError('Full-budget archive not verified')
        baseline=campaign.reference_result(root)
        delta=100*(result['final_dice']-baseline['final_dice'])
        value=dict(status='complete',reference_lr1_baseline=baseline,lr3_baseline=result,
            final_dice_delta_percentage_points=delta,threshold_passed=campaign.qualifies(result),
            cap_flops=campaign.CAP,halfway_gate=False,
            note='Full-budget LR3 continuation; prior pruning history retained separately; no additional training arms')
        io.write(durable/'lr3_full_budget_status.json',value)
        io.write(durable/'LR3_FULL_BUDGET_FINISHED.json',value)
        print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--durable',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=2)
    args=parser.parse_args()
    if not 0<args.hours<=2:parser.error('Session limit must be in (0,2] hours')
    run(args.bundle.resolve(),args.durable.resolve(),args.hours)
